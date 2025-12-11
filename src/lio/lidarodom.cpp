#include <yaml-cpp/yaml.h>
#include "lio/lidarodom.h"

#include <tbb/tbb.h>

#define P_USE_TBB_ 1 //   是否使用tbb加速

namespace zjloc
{

     lidarodom::lidarodom(/* args */) : index_frame(1)
     {
          current_state = new state(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

          points_world.reset(new pcl::PointCloud<pcl::PointXYZI>());
          std::cout << "original loam normal" << std::endl;
     }

     lidarodom::~lidarodom() {}

     void lidarodom::loadOptions()
     {
          auto yaml = YAML::LoadFile(config_yaml_);
          options_.surf_res = yaml["odometry"]["surf_res"].as<double>();
          options_.max_iteration = yaml["odometry"]["max_iteration"].as<int>();
          options_.log_print = yaml["odometry"]["log_print"].as<bool>();

          options_.size_voxel_map = yaml["odometry"]["size_voxel_map"].as<double>();
          options_.min_distance_points = yaml["odometry"]["min_distance_points"].as<double>();
          options_.max_num_points_in_voxel = yaml["odometry"]["max_num_points_in_voxel"].as<int>();
          options_.max_distance = yaml["odometry"]["max_distance"].as<double>();
          options_.weight_alpha = yaml["odometry"]["weight_alpha"].as<double>();
          options_.weight_neighborhood = yaml["odometry"]["weight_neighborhood"].as<double>();
          options_.max_dist_to_plane_icp = yaml["odometry"]["max_dist_to_plane_icp"].as<double>();
          options_.init_num_frames = yaml["odometry"]["init_num_frames"].as<int>();
          options_.voxel_neighborhood = yaml["odometry"]["voxel_neighborhood"].as<int>();
          options_.max_number_neighbors = yaml["odometry"]["max_number_neighbors"].as<int>();
          options_.threshold_voxel_occupancy = yaml["odometry"]["threshold_voxel_occupancy"].as<int>();
          options_.estimate_normal_from_neighborhood = yaml["odometry"]["estimate_normal_from_neighborhood"].as<bool>();
          options_.min_number_neighbors = yaml["odometry"]["min_number_neighbors"].as<int>();
          options_.power_planarity = yaml["odometry"]["power_planarity"].as<double>();
          options_.num_closest_neighbors = yaml["odometry"]["num_closest_neighbors"].as<int>();

          options_.sampling_rate = yaml["odometry"]["sampling_rate"].as<double>();
          options_.max_num_residuals = yaml["odometry"]["max_num_residuals"].as<int>();
          options_.min_num_residuals = yaml["odometry"]["min_num_residuals"].as<int>();

          options_.thres_orientation_norm = yaml["odometry"]["thres_orientation_norm"].as<double>();
          options_.thres_translation_norm = yaml["odometry"]["thres_translation_norm"].as<double>();
          adapt_sample_res = options_.sampling_rate;
     }

     bool lidarodom::init(const std::string &config_yaml)
     {
          config_yaml_ = config_yaml;
          StaticIMUInit::Options imu_init_options;
          imu_init_ = StaticIMUInit(imu_init_options);

          auto yaml = YAML::LoadFile(config_yaml_);

          // lidar和IMU外参
          std::vector<double> ext_t = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
          std::vector<double> ext_r = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();
          Vec3d lidar_T_wrt_IMU = math::VecFromArray(ext_t);
          Mat3d lidar_R_wrt_IMU = math::MatFromArray(ext_r);
          std::cout << yaml["mapping"]["extrinsic_R"] << std::endl;
          Eigen::Quaterniond q_IL(lidar_R_wrt_IMU);
          q_IL.normalized();
          lidar_R_wrt_IMU = q_IL;
          // init TIL
          TIL_ = SE3(q_IL, lidar_T_wrt_IMU);
          R_imu_lidar = lidar_R_wrt_IMU;
          t_imu_lidar = lidar_T_wrt_IMU;
          std::cout << "RIL:\n"
                    << R_imu_lidar << std::endl;
          std::cout << "tIL:" << t_imu_lidar.transpose() << std::endl;

          loadOptions();

          return true;
     }

     void lidarodom::pushData(std::vector<point3D> msg, std::pair<double, double> data)
     {
          if (data.first < last_timestamp_lidar_)
          {
               LOG(ERROR) << "lidar loop back, clear buffer";
               lidar_buffer_.clear();
               time_buffer_.clear();
          }

          mtx_buf.lock();
          lidar_buffer_.push_back(msg);
          time_buffer_.push_back(data);
          last_timestamp_lidar_ = data.first;
          mtx_buf.unlock();
          cond.notify_one();
     }

     void lidarodom::pushData(IMUPtr imu)
     {
          double timestamp = imu->timestamp_;
          if (timestamp < last_timestamp_imu_)
          {
               LOG(WARNING) << "imu loop back, clear buffer";
               imu_buffer_.clear();
          }

          last_timestamp_imu_ = timestamp;

          mtx_buf.lock();
          imu_buffer_.emplace_back(imu);
          mtx_buf.unlock();
          cond.notify_one();
     }

     void lidarodom::run()
     {
          while (true)
          {
               std::vector<MeasureGroup> measurements;
               std::unique_lock<std::mutex> lk(mtx_buf);
               cond.wait(lk, [&]
                         { return (measurements = getMeasureMents()).size() != 0; });
               lk.unlock();

               for (auto &m : measurements)
               {
                    zjloc::common::Timer::Evaluate([&]()
                                                   { ProcessMeasurements(m); },
                                                   "processMeasurement");

                    {
                         auto real_time = std::chrono::high_resolution_clock::now();
                         static std::chrono::system_clock::time_point prev_real_time = real_time;

                         if (real_time - prev_real_time > std::chrono::seconds(5))
                         {
                              auto data_time = m.lidar_end_time_;
                              static double prev_data_time = data_time;
                              auto delta_real = std::chrono::duration_cast<std::chrono::milliseconds>(real_time - prev_real_time).count() * 0.001;
                              auto delta_sim = data_time - prev_data_time;
                              printf("Processing the rosbag at %.1fX speed.", delta_sim / delta_real);

                              prev_data_time = data_time;
                              prev_real_time = real_time;
                         }
                    }
               }
          }
     }

     /**
      * [功能描述]: 处理一组传感器测量数据（IMU + LiDAR），执行完整的激光里程计流程
      *            包括：IMU预测、状态初始化、点云帧构建、位姿估计、ESKF观测更新、位姿发布
      * @param meas: 测量数据组，包含同步后的IMU数据和LiDAR点云数据
      */
     void lidarodom::ProcessMeasurements(MeasureGroup &meas)
     {
          // 保存当前测量数据到成员变量
          measures_ = meas;

          // 如果IMU尚未初始化，则尝试初始化IMU并返回
          if (imu_need_init_)
          {
               TryInitIMU();
               return;
          }

          // 打印当前处理的帧序号
          std::cout << ANSI_COLOR_GREEN << "============== process frame: "
                    << index_frame << ANSI_COLOR_RESET << std::endl;

          // 清空IMU状态缓存，为新一帧的IMU积分做准备
          imu_states_.clear(); //   need clear here

          // 步骤1: IMU预测 - 利用IMU数据进行状态预测（位置、速度、姿态）
          zjloc::common::Timer::Evaluate([&]()
                                         { Predict(); },
                                         "predict");

          // 步骤2: 状态初始化 - 初始化当前帧的优化状态
          zjloc::common::Timer::Evaluate([&]()
                                         { stateInitialization(); },
                                         "state init");

          // 步骤3: 构建点云帧
          // 将测量数据中的LiDAR点云复制到const_surf中
          std::vector<point3D> const_surf;
          const_surf.insert(const_surf.end(), meas.lidar_.begin(), meas.lidar_.end());

          // 构建点云帧对象，包含点云数据、当前状态、起止时间戳
          cloudFrame *p_frame;
          zjloc::common::Timer::Evaluate([&]()
                                         { p_frame = buildFrame(const_surf, current_state,
                                                                meas.lidar_begin_time_,
                                                                meas.lidar_end_time_); },
                                         "build frame");

          // 步骤4: 位姿估计 - 通过点云配准优化当前帧位姿
          zjloc::common::Timer::Evaluate([&]()
                                         { poseEstimation(p_frame); },
                                         "poseEstimate");

          // 从优化后的状态中提取位姿（旋转 + 平移），构建SE3李群表示
          SE3 pose_of_lo_ = SE3(current_state->rotation, current_state->translation);

          // 步骤5: ESKF观测更新 - 将激光里程计位姿作为观测量更新误差状态卡尔曼滤波器
          // 参数：位姿观测、位置噪声协方差(1e-2)、姿态噪声协方差(1e-2)
          zjloc::common::Timer::Evaluate([&]()
                                         { eskf_.ObserveSE3(pose_of_lo_, 1e-2, 1e-2); },
                                         "eskf_obs");

          // 步骤6: 发布位姿到ROS
          zjloc::common::Timer::Evaluate([&]()
                                         {
               // 发布激光里程计位姿（base_link相对于map的位姿）
               std::string laser_topic = "laser";
               pub_pose_to_ros(laser_topic, pose_of_lo_, meas.lidar_end_time_);

               // 发布世界坐标系位姿（用于TF变换：world -> map）
               laser_topic = "world";
               SE3 pose_of_world = SE3(RIG_,Eigen::Vector3d(0,0,0));
               pub_pose_to_ros(laser_topic, pose_of_world, meas.lidar_end_time_); },
                                         "pub cloud");

          // 步骤7: 状态管理与内存清理
          // 将当前状态深拷贝保存到点云帧中
          p_frame->p_state = new state(current_state, true);
          // 将当前状态深拷贝保存到全局状态历史列表中
          state *tmp_state = new state(current_state, true);
          all_state_frame.push_back(tmp_state);
          // 创建新的状态对象用于下一帧（浅拷贝，共享部分数据）
          current_state = new state(current_state, false);

          // 帧计数器递增
          index_frame++;
          // 释放点云帧资源
          p_frame->release();
          // 使用swap技巧释放vector内存，避免内存泄漏
          std::vector<point3D>().swap(meas.lidar_);
          std::vector<point3D>().swap(const_surf);
     }

     /**
      * [功能描述]: 位姿估计函数，执行点云配准优化和地图更新
      *            包括：迭代优化位姿、增量式地图更新、视场角分割
      * @param p_frame: 当前点云帧指针，包含点云数据和状态信息
      */
     void lidarodom::poseEstimation(cloudFrame *p_frame)
     {
          // 从第2帧开始才进行优化（第1帧用于初始化地图，无需优化）
          if (index_frame > 1)
          {
               // 执行位姿优化：通过点到面残差最小化，迭代优化当前帧位姿
               zjloc::common::Timer::Evaluate([&]()
                                              { optimize(p_frame); },
                                              "optimize");
          }

          // 是否将当前帧点云添加到全局地图中
          bool add_points = true;
          if (add_points)
          {
               // 增量式地图更新：将当前帧的点云添加到体素地图中
               zjloc::common::Timer::Evaluate([&]()
                                              { map_incremental(p_frame); },
                                              "map update");
          }

          // 视场角分割：移除超出当前传感器视场范围的地图点，控制地图规模
          zjloc::common::Timer::Evaluate([&]()
                                         { lasermap_fov_segment(); },
                                         "fov segment");
     }

     /**
      * [功能描述]: 位姿优化函数，使用Ceres求解器进行点到面ICP迭代优化
      *            通过最小化点到平面的残差，优化当前帧的旋转和平移
      * @param p_frame: 当前点云帧指针，包含待优化的点云和状态
      */
     void lidarodom::optimize(cloudFrame *p_frame)
     {
          // 获取当前帧状态，提取旋转四元数和平移向量作为优化变量
          state *curr_state = p_frame->p_state;
          Eigen::Quaterniond end_quat = curr_state->rotation;   // 待优化的旋转四元数
          Eigen::Vector3d end_t = curr_state->translation;      // 待优化的平移向量

          // ==================== 自适应降采样 ====================
          std::vector<point3D> surf_keypoints;
          // 对点云进行网格降采样，获取关键点用于优化
          // FIXME: adaptive filter, res_new = res_old*(N_scan/N_last);
          gridSampling(p_frame->point_surf, surf_keypoints, adapt_sample_res * options_.surf_res);
          
          // 根据关键点数量自适应调整采样分辨率
          double tt = surf_keypoints.size() / options_.max_num_residuals;
          if (tt < 1)
          {
               // 关键点数量不足，降低采样分辨率以获取更多点
               adapt_sample_res = 0.7 * adapt_sample_res; // 采样率范围：0.5 ~ options_.sample_rate
               if (adapt_sample_res < 0.5)
                    adapt_sample_res = 0.5;
          }
          else
               // 关键点数量充足，恢复默认采样率
               adapt_sample_res = options_.sampling_rate;

          // ==================== 定义点云变换Lambda函数 ====================
          size_t num_size = p_frame->point_surf.size();
          /**
           * transformKeypoints: 将点云从LiDAR坐标系变换到世界坐标系
           * 变换公式: P_world = R * (T_IL * P_lidar) + t
           * 其中: T_IL为LiDAR到IMU的外参, R和t为IMU到世界的位姿
           */
          auto transformKeypoints = [&](std::vector<point3D> &point_frame)
          {
               Eigen::Matrix3d R = end_quat.normalized().toRotationMatrix();  // 旋转矩阵
               Eigen::Vector3d t = end_t;                                      // 平移向量
#ifdef P_USE_TBB_
               // 使用TBB并行加速点云变换
               tbb::parallel_for(size_t(0), point_frame.size(), [&](size_t id)
                                 { point_frame[id].point = R * (TIL_ * point_frame[id].raw_point) + t;
                                   point_frame[id].normal = R * TIL_.rotationMatrix() * point_frame[id].raw_normal; });
#else
               // 串行方式进行点云变换
               for (auto &keypoint : point_frame)
               {
                    // 变换点坐标: 先通过外参TIL_变换到IMU系，再通过R,t变换到世界系
                    keypoint.point = R * (TIL_ * keypoint.raw_point) + t;
                    // 变换法向量: 只需旋转，不需要平移
                    keypoint.normal = R * TIL_.rotationMatrix() * keypoint.raw_normal;
               }
#endif
          };

          // ==================== ICP迭代优化主循环 ====================
          for (int iter(0); iter < options_.max_iteration; iter++)
          {
               // 创建Huber鲁棒核函数，阈值0.5，用于抑制外点影响
               ceres::LossFunction *loss_function = new ceres::HuberLoss(0.5);
               ceres::Problem::Options problem_options;
               ceres::Problem problem(problem_options);

               // 为四元数创建参数化器，确保优化过程中四元数保持单位范数
               auto *parameterization = new ceres::EigenQuaternionParameterization();

               // 添加优化变量：四元数(4维)和平移向量(3维)
               problem.AddParameterBlock(&end_quat.x(), 4, parameterization);
               problem.AddParameterBlock(&end_t.x(), 3);

               // 构建点到面残差因子
               std::vector<ceres::CostFunction *> surfFactor;
               addSurfCostFactor(surfFactor, surf_keypoints, p_frame);

               // 将所有残差因子添加到优化问题中
               int surf_num = 0;
               if (options_.log_print)
                    std::cout << "get factor: " << surfFactor.size() << std::endl;
               for (auto &e : surfFactor)
               {
                    surf_num++;
                    // 添加残差块：代价函数、鲁棒核、平移参数、旋转参数
                    problem.AddResidualBlock(e, loss_function, &end_t.x(), &end_quat.x());
               }
               // 释放临时vector内存
               std::vector<ceres::CostFunction *>().swap(surfFactor);

               // 检查残差数量是否足够，不足则终止优化
               if (surf_num < options_.min_num_residuals)
               {
                    std::stringstream ss_out;
                    ss_out << "[Optimization] Error : not enough keypoints selected in ct-icp !" << std::endl;
                    ss_out << "[Optimization] number_of_residuals : " << surf_num << std::endl;
                    std::cout << "ERROR: " << ss_out.str();
                    return;
               }

               // ==================== 配置Ceres求解器 ====================
               ceres::Solver::Options options;
               options.max_num_iterations = 5;                    // 每次ICP迭代中的最大优化次数
               options.num_threads = 6;                           // 并行线程数
               options.minimizer_progress_to_stdout = false;      // 不打印优化过程
               // 使用Levenberg-Marquardt信赖域策略
               options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;

               ceres::Solver::Summary summary;

               // 执行优化求解
               ceres::Solve(options, &problem, &summary);

               // ==================== 收敛性检查 ====================
               // 计算本次迭代的位姿变化量
               double diff_trans = 0, diff_rot = 0;
               diff_trans += (current_state->translation - end_t).norm();      // 平移变化量(米)
               diff_rot += AngularDistance(current_state->rotation, end_quat); // 旋转变化量(弧度)

               // 更新点云帧状态和全局当前状态
               p_frame->p_state->translation = end_t;
               p_frame->p_state->rotation = end_quat;

               current_state->translation = end_t;
               current_state->rotation = end_quat;

               // 如果旋转和平移变化量都小于阈值，认为已收敛，提前退出
               if (diff_rot < options_.thres_orientation_norm &&
                   diff_trans < options_.thres_translation_norm)
               {
                    if (options_.log_print)
                         std::cout << "Optimization: Finished with N=" << iter << " ICP iterations" << std::endl;
                    break;
               }
          }

          // 释放关键点内存
          std::vector<point3D>().swap(surf_keypoints);
          // 将优化后的位姿应用到原始点云，变换到世界坐标系（用于后续地图更新）
          transformKeypoints(p_frame->point_surf);
     }

     /**
      * [功能描述]: 计算邻域点云的分布特性（PCA主成分分析）
      *            通过协方差矩阵的特征值分解，提取法向量和平面性特征
      * @param points: 邻域点集合（世界坐标系下的3D点）
      * @return Neighborhood: 包含中心点、法向量、协方差矩阵、平面性等信息的结构体
      */
     Neighborhood lidarodom::computeNeighborhoodDistribution(const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &points)
     {
          Neighborhood neighborhood;
          
          // ==================== 步骤1: 计算质心（重心） ====================
          Eigen::Vector3d barycenter(Eigen::Vector3d(0, 0, 0));
          for (auto &point : points)
          {
               barycenter += point;
          }
          // 质心 = 所有点坐标的算术平均值
          barycenter /= (double)points.size();
          neighborhood.center = barycenter;

          // ==================== 步骤2: 计算协方差矩阵 ====================
          // 协方差矩阵描述点云的分布形状
          // Cov(i,j) = Σ(p_i - mean_i)(p_j - mean_j)
          Eigen::Matrix3d covariance_Matrix(Eigen::Matrix3d::Zero());
          for (auto &point : points)
          {
               // 只计算上三角部分（因为协方差矩阵是对称的）
               for (int k = 0; k < 3; ++k)
                    for (int l = k; l < 3; ++l)
                         covariance_Matrix(k, l) += (point(k) - barycenter(k)) *
                                                    (point(l) - barycenter(l));
          }
          // 填充下三角部分（对称矩阵）
          covariance_Matrix(1, 0) = covariance_Matrix(0, 1);
          covariance_Matrix(2, 0) = covariance_Matrix(0, 2);
          covariance_Matrix(2, 1) = covariance_Matrix(1, 2);
          neighborhood.covariance = covariance_Matrix;
          
          // ==================== 步骤3: 特征值分解（PCA） ====================
          // 使用自伴随特征值求解器（适用于对称矩阵）
          // 特征值按升序排列：eigenvalues[0] < eigenvalues[1] < eigenvalues[2]
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(covariance_Matrix);
          
          // 最小特征值对应的特征向量即为法向量方向
          // 因为平面上点的分布在法向量方向上方差最小
          Eigen::Vector3d normal(es.eigenvectors().col(0).normalized());
          neighborhood.normal = normal;

          // ==================== 步骤4: 计算平面性特征 a2D ====================
          // sigma_1, sigma_2, sigma_3 分别对应最大、中间、最小特征值的平方根
          // 它们表示点云在三个主方向上的分布程度（标准差）
          double sigma_1 = sqrt(std::abs(es.eigenvalues()[2]));  // 最大特征值 -> 第一主方向
          double sigma_2 = sqrt(std::abs(es.eigenvalues()[1]));  // 中间特征值 -> 第二主方向
          double sigma_3 = sqrt(std::abs(es.eigenvalues()[0]));  // 最小特征值 -> 法向量方向
          
          // a2D = (σ₂ - σ₃) / σ₁
          // 平面性指标：
          // - 理想平面: σ₃ ≈ 0, a2D ≈ σ₂/σ₁ (较大)
          // - 线状分布: σ₂ ≈ σ₃ ≈ 0, a2D ≈ 0
          // - 球状分布: σ₁ ≈ σ₂ ≈ σ₃, a2D ≈ 0
          neighborhood.a2D = (sigma_2 - sigma_3) / sigma_1;

          // 检查NaN（当a2D != a2D时表示出现NaN）
          if (neighborhood.a2D != neighborhood.a2D)
          {
               throw std::runtime_error("error");
          }

          return neighborhood;
     }

     /**
      * [功能描述]: 构建点到面残差因子，用于Ceres优化
      *            为每个关键点在体素地图中搜索邻近点，计算局部平面，构建点到面约束
      * @param surf: 输出的残差因子列表（Ceres代价函数指针）
      * @param keypoints: 当前帧的关键点集合
      * @param p_frame: 当前点云帧指针，包含位姿状态信息
      */
     void lidarodom::addSurfCostFactor(std::vector<ceres::CostFunction *> &surf,
                                       std::vector<point3D> &keypoints, const cloudFrame *p_frame)
     {
          /**
           * Lambda函数：估计点的邻域平面特性
           * @param vector_neighbors: 邻近点集合
           * @param location: 当前点在IMU坐标系下的位置
           * @param planarity_weight: 输出的平面性权重（平面越平，权重越大）
           * @return neighborhood: 邻域分布特性（包含法向量、平面性等）
           */
          auto estimatePointNeighborhood = [&](std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> &vector_neighbors,
                                               Eigen::Vector3d &location, double &planarity_weight)
          {
               // 计算邻域点的分布特性（PCA分析得到法向量和平面性）
               auto neighborhood = computeNeighborhoodDistribution(vector_neighbors);
               // 根据平面性计算权重，power_planarity控制权重的敏感度
               planarity_weight = std::pow(neighborhood.a2D, options_.power_planarity);

               // 确保法向量指向传感器方向（法向量一致性校正）
               if (neighborhood.normal.dot(p_frame->p_state->translation - location) < 0)
               {
                    neighborhood.normal = -1.0 * neighborhood.normal;
               }
               return neighborhood;
          };

          // ==================== 权重参数归一化 ====================
          double lambda_weight = std::abs(options_.weight_alpha);           // 平面性权重系数
          double lambda_neighborhood = std::abs(options_.weight_neighborhood); // 邻域距离权重系数
          const double kMaxPointToPlane = options_.max_dist_to_plane_icp;   // 点到面最大距离阈值
          const double sum = lambda_weight + lambda_neighborhood;

          // 归一化权重，使两个权重之和为1
          lambda_weight /= sum;
          lambda_neighborhood /= sum;

          // ==================== 自适应参数设置 ====================
          // 初始化阶段使用更小的搜索范围，后续使用配置值
          const short nb_voxels_visited = p_frame->frame_id < options_.init_num_frames
                                              ? 2  // 初始化阶段：搜索2个邻近体素
                                              : options_.voxel_neighborhood;  // 正常阶段：使用配置的邻域大小

          // 体素占用阈值：初始化阶段要求更低
          const int kThresholdCapacity = p_frame->frame_id < options_.init_num_frames
                                             ? 1  // 初始化阶段：体素内至少1个点
                                             : options_.threshold_voxel_occupancy;  // 正常阶段：使用配置阈值

          size_t num = keypoints.size();
          int num_residuals = 0;  // 已添加的残差数量计数器

          // ==================== 遍历所有关键点，构建残差因子 ====================
          for (int k = 0; k < num; k++)
          {
               auto &keypoint = keypoints[k];
               auto &raw_point = keypoint.raw_point;
               
               // 将原始点从LiDAR坐标系变换到世界坐标系
               // P_world = R * (T_IL * P_lidar) + t
               keypoint.point = p_frame->p_state->rotation * (TIL_ * raw_point) + p_frame->p_state->translation;
               // 变换法向量（只旋转，不平移）
               keypoint.normal = p_frame->p_state->rotation * TIL_.unit_quaternion() * keypoint.raw_normal;

               // 在体素地图中搜索当前点的邻近点
               std::vector<voxel> voxels;
               auto vector_neighbors = searchNeighbors(voxel_map, keypoint.point,
                                                       keypoint.normal, nb_voxels_visited,
                                                       options_.size_voxel_map,
                                                       options_.max_number_neighbors,
                                                       kThresholdCapacity,
                                                       options_.estimate_normal_from_neighborhood
                                                           ? nullptr
                                                           : &voxels);

               // 邻近点数量不足，跳过该关键点
               if (vector_neighbors.size() < options_.min_number_neighbors)
                    continue;

               double weight;

               // 计算点在IMU坐标系下的位置（用于法向量方向校正）
               Eigen::Vector3d location = TIL_ * raw_point;

               // 估计邻域平面特性，获取法向量和平面性权重
               auto neighborhood = estimatePointNeighborhood(vector_neighbors, location, weight);

               // 计算综合权重：平面性权重 + 邻域距离权重
               // 距离越近、平面性越好，权重越大
               weight = lambda_weight * weight + lambda_neighborhood *
                                                     std::exp(-(vector_neighbors[0] -
                                                                keypoint.point)
                                                                   .norm() /
                                                              (kMaxPointToPlane *
                                                               options_.min_number_neighbors));

               // ==================== 构建点到面残差 ====================
               double point_to_plane_dist;
               std::set<voxel> neighbor_voxels;
               
               // 对最近的几个邻近点分别构建残差
               for (int i(0); i < options_.num_closest_neighbors; ++i)
               {
                    // 计算点到平面的距离：|（P - Q）· n|
                    // P为当前点，Q为邻近点，n为平面法向量
                    point_to_plane_dist = std::abs((keypoint.point - vector_neighbors[i]).transpose() * neighborhood.normal);

                    // 距离小于阈值才添加残差（过滤外点）
                    if (point_to_plane_dist < options_.max_dist_to_plane_icp)
                    {

                         num_residuals++;

                         // 归一化法向量
                         Eigen::Vector3d norm_vector = neighborhood.normal;
                         norm_vector.normalize();

                         // 计算平面方程的偏移量 d = -n · Q
                         double norm_offset = -norm_vector.dot(vector_neighbors[i]);

                         // 将点变换回body坐标系（用于优化，因为优化变量是body系位姿）
                         // P_body = R^(-1) * P_world - R^(-1) * t
                         Eigen::Vector3d point_end = p_frame->p_state->rotation.inverse() * keypoints[k].point -
                                                     p_frame->p_state->rotation.inverse() * p_frame->p_state->translation;

                         // 创建点到面代价函数
                         // 参数：参考点（地图点）、当前点（body系）、法向量、权重
                         auto *cost_function = zjloc::PointToPlaneFunctor::Create(vector_neighbors[0],
                                                                                  point_end, norm_vector, weight);

                         surf.push_back(cost_function);
                    }
               }

               // 残差数量达到上限，提前退出
               if (num_residuals >= options_.max_num_residuals)
                    break;
          }
     }

     ///  ===================  for search neighbor  ===================================================
     using pair_distance_t = std::tuple<double, Eigen::Vector3d, voxel>;

     struct comparator
     {
          bool operator()(const pair_distance_t &left, const pair_distance_t &right) const
          {
               return std::get<0>(left) < std::get<0>(right);
          }
     };

     using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, comparator>;

     /**
      * [功能描述]: 在体素地图中搜索给定点的邻近点
      *            使用法向量一致性过滤，只返回法向量方向相似的邻近点
      *            利用优先队列维护距离最近的K个邻近点
      * @param map: 体素哈希地图，存储所有地图点
      * @param point: 查询点的位置（世界坐标系）
      * @param normal: 查询点的法向量，用于法向量一致性过滤
      * @param nb_voxels_visited: 搜索邻域大小（以体素为单位），搜索范围为 [-n, n]
      * @param size_voxel_map: 体素大小（米）
      * @param max_num_neighbors: 返回的最大邻近点数量
      * @param threshold_voxel_capacity: 体素内最小点数阈值，点数不足的体素被跳过
      * @param voxels: 可选输出，返回邻近点所属的体素索引
      * @return 邻近点坐标列表，按距离从近到远排序
      */
     std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
     lidarodom::searchNeighbors(const voxelHashMap2 &map, const Eigen::Vector3d &point,
                                const Eigen::Vector3d &normal, int nb_voxels_visited,
                                double size_voxel_map, int max_num_neighbors,
                                int threshold_voxel_capacity, std::vector<voxel> *voxels)
     {
          // 预分配体素索引输出容器
          if (voxels != nullptr)
               voxels->reserve(max_num_neighbors);

          // ==================== 步骤1: 计算查询点所属的体素索引 ====================
          short kx = static_cast<short>(point[0] / size_voxel_map);
          short ky = static_cast<short>(point[1] / size_voxel_map);
          short kz = static_cast<short>(point[2] / size_voxel_map);

          /**
           * Lambda函数：计算两个向量之间的夹角（单位：度）
           * @param v1, v2: 两个归一化向量
           * @return 夹角（0° ~ 180°）
           */
          auto calculateAngle = [&](const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
          {
               double dot_product = v1.dot(v2);
               return std::acos(dot_product) * 180.0 / M_PI; // 转换为角度
          };

          // 优先队列：按距离排序，存储(距离, 点坐标, 体素索引)
          // 大顶堆：队首是距离最大的元素，方便淘汰最远的点
          priority_queue_t priority_queue;

          // ==================== 步骤2: 遍历邻域体素，搜索邻近点 ====================
          voxel voxel_temp(kx, ky, kz);
          // 三重循环遍历以(kx,ky,kz)为中心的立方体邻域
          for (short kxx = kx - nb_voxels_visited; kxx < kx + nb_voxels_visited + 1; ++kxx)
          {
               for (short kyy = ky - nb_voxels_visited; kyy < ky + nb_voxels_visited + 1; ++kyy)
               {
                    for (short kzz = kz - nb_voxels_visited; kzz < kz + nb_voxels_visited + 1; ++kzz)
                    {
                         voxel_temp.x = kxx;
                         voxel_temp.y = kyy;
                         voxel_temp.z = kzz;

                         // 在哈希表中查找该体素
                         auto search = map.find(voxel_temp);
                         if (search != map.end())
                         {
                              const auto &voxel_block = search.value();
                              
                              // ========== 处理主方向点集（points） ==========
                              // 先判断体素的主法向量与查询点法向量的夹角
                              double angle = calculateAngle(voxel_block.normal, normal);
                              if (angle < 90.0)  // 夹角小于90°才考虑（法向量方向大致相同）
                              {
                                   // 体素内点数不足阈值则跳过
                                   if (voxel_block.NumPoints() < threshold_voxel_capacity)
                                        continue;
                                   // 遍历体素内的每个点
                                   for (int i(0); i < voxel_block.NumPoints(); ++i)
                                   {
                                        auto &neighbor = voxel_block.points[i];
                                        // 检查单点法向量与查询点法向量的夹角
                                        double angle = calculateAngle(neighbor.getNormal(), normal);
                                        if (angle > 100.0)  // 夹角大于100°则跳过（法向量方向差异太大）
                                             continue;

                                        // 计算点到查询点的距离
                                        Eigen::Vector3d neighbor_point = neighbor.getPosition();
                                        double distance = (neighbor_point - point).norm();
                                        
                                        // 维护大小为max_num_neighbors的优先队列
                                        if (priority_queue.size() == max_num_neighbors)
                                        {
                                             // 队列已满，只有当新点距离更近时才替换队首（最远的点）
                                             if (distance < std::get<0>(priority_queue.top()))
                                             {
                                                  priority_queue.pop();
                                                  priority_queue.emplace(distance, neighbor_point, voxel_temp);
                                             }
                                        }
                                        else
                                             // 队列未满，直接加入
                                             priority_queue.emplace(distance, neighbor_point, voxel_temp);
                                   }
                              }

                              // ========== 处理次方向点集（other_points） ==========
                              // 体素内可能存储两个方向的平面点（双面体素）
                              angle = calculateAngle(voxel_block.other_normal, normal);
                              if (angle < 90.0)
                              {
                                   if (voxel_block.NumOtherPoints() < threshold_voxel_capacity)
                                        continue;
                                   for (int i(0); i < voxel_block.NumOtherPoints(); ++i)
                                   {
                                        auto &neighbor = voxel_block.other_points[i];
                                        double angle = calculateAngle(neighbor.getNormal(), normal);
                                        if (angle > 100.0)
                                             continue;

                                        Eigen::Vector3d neighbor_point = neighbor.getPosition();
                                        double distance = (neighbor_point - point).norm();
                                        if (priority_queue.size() == max_num_neighbors)
                                        {
                                             if (distance < std::get<0>(priority_queue.top()))
                                             {
                                                  priority_queue.pop();
                                                  priority_queue.emplace(distance, neighbor_point, voxel_temp);
                                             }
                                        }
                                        else
                                             priority_queue.emplace(distance, neighbor_point, voxel_temp);
                                   }
                              }
                         }
                    }
               }
          }

          // ==================== 步骤3: 从优先队列中提取结果 ====================
          auto size = priority_queue.size();
          std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> closest_neighbors(size);
          if (voxels != nullptr)
          {
               voxels->resize(size);
          }
          // 优先队列是大顶堆，队首是最远的点
          // 倒序填充结果数组，使得最终结果按距离从近到远排序
          for (auto i = 0; i < size; ++i)
          {
               closest_neighbors[size - 1 - i] = std::get<1>(priority_queue.top());  // 点坐标
               if (voxels != nullptr)
                    (*voxels)[size - 1 - i] = std::get<2>(priority_queue.top());     // 体素索引
               priority_queue.pop();
          }

          return closest_neighbors;
     }

     /**
      * [功能描述]: 将点添加到体素地图中
      *            包含法向量一致性检查、距离过滤、容量控制等策略
      * @param map: 体素哈希地图（引用，会被修改）
      * @param pt: 待添加的点（包含位置、法向量、强度等信息）
      * @param voxel_size: 体素大小（米）
      * @param max_num_points_in_voxel: 每个体素内最大点数
      * @param min_distance_points: 体素内点之间的最小距离（避免过于密集）
      * @param min_num_points: 体素内最小点数阈值（用于延迟添加策略）
      */
     void lidarodom::addPointToMap(voxelHashMap2 &map, const point3D &pt, double voxel_size,
                                   int max_num_points_in_voxel, double min_distance_points,
                                   int min_num_points)
     {
          // ==================== 步骤1: 计算点所属的体素索引 ====================
          short kx = static_cast<short>(pt.point[0] / voxel_size);
          short ky = static_cast<short>(pt.point[1] / voxel_size);
          short kz = static_cast<short>(pt.point[2] / voxel_size);

          // 将点添加到用于可视化的PCL点云中
          addPointToPcl(points_world, pt.point, pt.intensity);

          // 在哈希表中查找该体素
          voxelHashMap2::iterator search = map.find(voxel(kx, ky, kz));

          /**
           * Lambda函数：计算两个向量之间的夹角（单位：度）
           */
          auto calculateAngle = [&](const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
          {
               double dot_product = v1.dot(v2);
               return std::acos(dot_product) * 180.0 / M_PI; // 转换为角度
          };

          // ==================== 步骤2: 根据体素是否存在采取不同策略 ====================
          if (search != map.end())
          {
               // ========== 体素已存在 ==========
               auto &voxel_block = (search.value());
               
               // 计算新点法向量与体素主法向量的夹角
               double angle = calculateAngle(pt.normal, voxel_block.normal);
               
               if (angle > 100.0)
               {
                    // 法向量夹角过大（>100°），说明新点属于不同方向的平面
                    // 重置体素块，用新点替换（处理场景变化或动态物体）
                    voxelBlock3 block(max_num_points_in_voxel);
                    block.AddPoint(normalPoint(pt));
                    voxel_block = block;
               }
               else if (!voxel_block.IsFull())
               {
                    // 体素未满且法向量一致，尝试添加点
                    
                    // 计算新点与体素内现有点的最小距离平方
                    double sq_dist_min_to_points = 10 * voxel_size * voxel_size;  // 初始化为较大值
                    for (int i(0); i < voxel_block.NumPoints(); ++i)
                    {
                         auto &_point = voxel_block.points[i];
                         double sq_dist = (_point.getPosition() - pt.point).squaredNorm();
                         if (sq_dist < sq_dist_min_to_points)
                         {
                              sq_dist_min_to_points = sq_dist;
                         }
                    }
                    
                    // 只有当新点与所有现有点的距离都大于阈值时才添加
                    // 避免点过于密集，保持空间分布均匀
                    if (sq_dist_min_to_points > (min_distance_points * min_distance_points))
                    {
                         // min_num_points <= 0 表示无延迟添加限制
                         // 或者体素内点数已达到最小阈值才允许继续添加
                         if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points)
                         {
                              voxel_block.AddPoint(normalPoint(pt));
                         }
                    }
               }
               // 如果体素已满（IsFull()），则不添加新点
          }
          else
          {
               // ========== 体素不存在，需要创建新体素 ==========
               // 只有当 min_num_points <= 0 时才立即创建
               // 否则需要等待累积足够的点（延迟创建策略）
               if (min_num_points <= 0)
               {
                    voxelBlock3 block(max_num_points_in_voxel);
                    block.AddPoint(normalPoint(pt));
                    // 使用move语义避免拷贝，提高效率
                    map[voxel(kx, ky, kz)] = std::move(block);
               }
          }
     }

     void lidarodom::addPointToPcl(pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_points, const Eigen::Vector3d &point, const double &intensity)
     {
          pcl::PointXYZI cloudTemp;

          cloudTemp.x = point.x();
          cloudTemp.y = point.y();
          cloudTemp.z = point.z();
          cloudTemp.intensity = intensity;
          // cloudTemp.intensity = 50 * (point.z() - p_frame->p_state->translation.z());
          pcl_points->points.push_back(cloudTemp);
     }

     /**
      * [功能描述]: 增量式地图更新函数，将当前帧点云添加到体素地图中
      * @param p_frame: 当前点云帧指针，包含已变换到世界坐标系的点云
      * @param min_num_points: 体素内最小点数阈值，默认值为0
      */
     void lidarodom::map_incremental(cloudFrame *p_frame, int min_num_points)
     {
          // 遍历当前帧的所有面点，逐个添加到体素地图中
          for (const auto &point : p_frame->point_surf)
               // 添加点到体素地图，参数依次为：
               // voxel_map: 全局体素地图
               // point: 待添加的点
               // size_voxel_map: 体素大小
               // max_num_points_in_voxel: 每个体素内最大点数
               // min_distance_points: 体素内点之间的最小距离（避免点过于密集）
               // min_num_points: 体素内最小点数阈值
               addPointToMap(voxel_map, point,
                             options_.size_voxel_map, options_.max_num_points_in_voxel,
                             options_.min_distance_points, min_num_points);

          // 发布点云到ROS用于可视化
          {
               std::string laser_topic = "laser";
               pub_cloud_to_ros(laser_topic, points_world, p_frame->time_frame_end);
          }
          // 清空世界坐标系点云缓存，为下一帧做准备
          points_world->clear();
     }

     void lidarodom::lasermap_fov_segment()
     {
          //   use predict pose here
          Eigen::Vector3d location = current_state->translation;

          for (auto &pair : voxel_map)
          {
               Eigen::Vector3d pt = pair.second.points[0].getPosition();
               if ((pt - location).squaredNorm() > (options_.max_distance * options_.max_distance))
                    voxel_map.erase(pair.first);
          }
     }

     cloudFrame *lidarodom::buildFrame(std::vector<point3D> &const_surf, state *cur_state,
                                       double timestamp_begin, double timestamp_end)
     {
          std::vector<point3D> frame_surf(const_surf);
          if (index_frame >= 2)
               zjloc::common::Timer::Evaluate([&]()
                                              { Undistort(frame_surf); },
                                              "cloudFrame");

          cloudFrame *p_frame = new cloudFrame(frame_surf, const_surf, cur_state);

          p_frame->time_frame_begin = timestamp_begin;
          p_frame->time_frame_end = timestamp_end;
          p_frame->frame_id = index_frame;

          return p_frame;
     }

     void lidarodom::stateInitialization()
     {
          if (index_frame < 2) //   only first frame
          {
               current_state->rotation = Eigen::Quaterniond(imu_states_.back().R_.matrix());
               current_state->translation = imu_states_.back().p_;
          }
          else
          {
               //   use imu predict
               current_state->rotation = Eigen::Quaterniond(imu_states_.back().R_.matrix());
               current_state->translation = imu_states_.back().p_;
          }
     }

     std::vector<MeasureGroup> lidarodom::getMeasureMents()
     {
          std::vector<MeasureGroup> measurements;
          while (true)
          {
               if (imu_buffer_.empty())
                    return measurements;

               if (lidar_buffer_.empty())
                    return measurements;

               if (lidar_buffer_.size() < 2)
                    return measurements;

               MeasureGroup meas;

               meas.lidar_ = lidar_buffer_.front();
               meas.lidar_begin_time_ = time_buffer_.front().first;
               meas.lidar_end_time_ = meas.lidar_begin_time_ + time_buffer_.front().second;
               lidar_buffer_.pop_front();
               time_buffer_.pop_front();

               time_curr = meas.lidar_end_time_;

               double imu_time = imu_buffer_.front()->timestamp_;
               meas.imu_.clear();
               while ((!imu_buffer_.empty()) && (imu_time < meas.lidar_end_time_))
               {
                    imu_time = imu_buffer_.front()->timestamp_;
                    if (imu_time > meas.lidar_end_time_)
                    {
                         break;
                    }
                    meas.imu_.push_back(imu_buffer_.front());
                    imu_buffer_.pop_front();
               }

               if (meas.imu_.empty())
               {
                    std::cout << "no imu before current lidar frame." << std::endl;
                    return measurements;
               }

               if (!imu_buffer_.empty())
                    meas.imu_.push_back(imu_buffer_.front()); //   added for Interp

               measurements.push_back(meas);
          }
     }

     void lidarodom::Predict()
     {
          imu_states_.emplace_back(eskf_.GetNominalState());

          /// 对IMU状态进行预测
          double time_current = measures_.lidar_end_time_;
          Vec3d last_gyr, last_acc;
          for (auto &imu : measures_.imu_)
          {
               double time_imu = imu->timestamp_;
               if (imu->timestamp_ <= time_current)
               {
                    if (last_imu_ == nullptr)
                         last_imu_ = imu;
                    eskf_.Predict(*imu);
                    imu_states_.emplace_back(eskf_.GetNominalState());
                    last_imu_ = imu;
               }
               else
               {
                    double dt_1 = time_imu - time_current;
                    double dt_2 = time_current - last_imu_->timestamp_;
                    double w1 = dt_1 / (dt_1 + dt_2);
                    double w2 = dt_2 / (dt_1 + dt_2);
                    Eigen::Vector3d acc_temp = w1 * last_imu_->acce_ + w2 * imu->acce_;
                    Eigen::Vector3d gyr_temp = w1 * last_imu_->gyro_ + w2 * imu->gyro_;
                    IMUPtr imu_temp = std::make_shared<zjloc::IMU>(time_current, gyr_temp, acc_temp);
                    eskf_.Predict(*imu_temp);
                    imu_states_.emplace_back(eskf_.GetNominalState());
                    last_imu_ = imu_temp;
               }
          }
     }

     void lidarodom::Undistort(std::vector<point3D> &points)
     {
          auto imu_state = eskf_.GetNominalState();
          SE3 T_end = SE3(imu_state.R_, imu_state.p_);

#ifdef P_USE_TBB_
          tbb::parallel_for(size_t(0), points.size(), [&](size_t id)
                            {
                                   SE3 Ti = T_end;
                                   NavStated match;
                                   point3D& pt = points[id];
                                   
                                   math::PoseInterp<NavStated>(
                                   pt.timestamp, imu_states_, [](const NavStated &s)
                                   { return s.timestamp_; },
                                   [](const NavStated &s)
                                   { return s.GetSE3(); },
                                   Ti, match);

                                   pt.raw_point = TIL_.inverse() * T_end.inverse() * Ti * TIL_ * pt.raw_point; });
#else
          for (auto &pt : points)
          {
               SE3 Ti = T_end;
               NavStated match;

               math::PoseInterp<NavStated>(
                   pt.timestamp, imu_states_, [](const NavStated &s)
                   { return s.timestamp_; },
                   [](const NavStated &s)
                   { return s.GetSE3(); },
                   Ti, match);

               pt.raw_point = TIL_.inverse() * T_end.inverse() * Ti * TIL_ * pt.raw_point;
          }
#endif
     }

     void lidarodom::TryInitIMU()
     {
          for (auto imu : measures_.imu_)
          {
               imu_init_.AddIMU(*imu);
          }

          if (imu_init_.InitSuccess())
          {
               zjloc::ESKFD::Options options;
               eskf_.SetInitialConditions(options, imu_init_.GetInitBg(), imu_init_.GetInitBa(), imu_init_.GetGravity());
               imu_need_init_ = false;
               RIG_ = SO3(g2R(imu_init_.GetMeanAcc()));

               std::cout << ANSI_COLOR_GREEN_BOLD << "IMU init successful..." << ANSI_COLOR_RESET << std::endl;
          }
     }

     int lidarodom::getIndex() { return index_frame; }
     void lidarodom::setFunc(std::function<bool(std::string &topic_name,
                                                CloudPtr &cloud, double time)> &fun)
     {
          pub_cloud_to_ros = fun;
     }
     void lidarodom::setFunc(std::function<bool(std::string &topic_name,
                                                SE3 &pose, double time)> &fun)
     {
          pub_pose_to_ros = fun;
     }

}
