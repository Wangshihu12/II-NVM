#include "cloud_convert.h"

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include "common/nano_flann/nanoflann_pcl.hpp"
#include "common/timer/timer.h"
#include "tbb/tbb.h"
#include <mutex>

#define USE_TBB_PARAL 1

namespace zjloc
{

    void CloudConvert::Process(const livox_ros_driver2::CustomMsg::ConstPtr &msg, std::vector<point3D> &pcl_out)
    {
        AviaHandler(msg);
        pcl_out = cloud_out_;
    }

    void CloudConvert::Process(const sensor_msgs::PointCloud2::ConstPtr &msg,
                               std::vector<point3D> &pcl_out)
    {
        switch (param_.lidar_type)
        {
        case LidarType::OUST64:
            Oust64Handler(msg);
            break;

        case LidarType::VELO32:
            VelodyneHandler(msg);
            break;

        case LidarType::ROBOSENSE16:
            RobosenseHandler(msg);
            break;

        case LidarType::AVIA_PC:
            AVIAPCHandler(msg);
            break;

        default:
            LOG(ERROR) << "Error LiDAR Type: " << int(lidar_type_);
            break;
        }
        pcl_out = cloud_out_;
    }

    /**
     * [功能描述]: 处理 Livox Avia 激光雷达的自定义点云消息
     *            包括：点云滤波、KD树构建、法向量估计、时间戳计算
     * @param msg: Livox ROS驱动发布的自定义点云消息（CustomMsg类型）
     */
    void CloudConvert::AviaHandler(const livox_ros_driver2::CustomMsg::ConstPtr &msg)
    {
        // 清空输出点云容器
        cloud_out_.clear();
        cloud_full_.clear();
        int plsize = msg->point_num;  // 点云中的点数
        cloud_out_.reserve(plsize);   // 预分配内存

        // 时间单位转换：Livox的offset_time单位是纳秒，需转换为秒
        static double tm_scale = 1e9;

        // 获取消息头的时间戳（帧起始时间）
        double headertime = msg->header.stamp.toSec();
        // 计算帧时间跨度：最后一个点的相对时间即为整帧扫描耗时
        timespan_ = msg->points.back().offset_time / tm_scale;

        // std::cout << "span:" << timespan_ << ",0: " << msg->points[0].offset_time / tm_scale
        //           << " , 100: " << msg->points[100].offset_time / tm_scale << std::endl;

        // ==================== 步骤1: 构建有效点云用于KD树 ====================
        zjloc::common::TicToc tc;
        tc.tic();
        CloudPtr cloud(new PointCloudType);
        for (int i = 0; i < plsize; i++)
        {
            // 过滤无效点（NaN或Inf）
            if (!(std::isfinite(msg->points[i].x) &&
                  std::isfinite(msg->points[i].y) &&
                  std::isfinite(msg->points[i].z)))
                continue;
            PointType pt;
            pt.x = msg->points[i].x;
            pt.y = msg->points[i].y;
            pt.z = msg->points[i].z;
            pt.intensity = msg->points[i].reflectivity;  // 反射强度
            cloud->push_back(pt);
        }
        double t1 = tc.toc();  // 记录点云构建耗时
        
        // ==================== 步骤2: 构建KD树用于邻域搜索 ====================
        tc.tic();
        nanoflann::KdTreeFLANN<PointType> nano_kdtree;
        nano_kdtree.setInputCloud(cloud);
        double t2 = tc.toc();  // 记录KD树构建耗时
        tc.tic();

        // ==================== 步骤3: 遍历点云，计算法向量 ====================
#ifdef USE_TBB_PARAL
        // TBB并行版本：使用多线程加速处理
        std::mutex valid_points_mutex;  // 用于保护共享数据的互斥锁
        tbb::parallel_for(tbb::blocked_range<size_t>(0, plsize),
                          [&](const tbb::blocked_range<size_t> &r)
                          {
                              std::vector<point3D> local_points;  // 每个线程的局部点云
                              for (size_t i = r.begin(); i != r.end(); ++i)
                              {
                                  // 过滤无效点
                                  if (!(std::isfinite(msg->points[i].x) &&
                                        std::isfinite(msg->points[i].y) &&
                                        std::isfinite(msg->points[i].z)))
                                      continue;
                                  // 降采样：每隔point_filter_num个点取一个
                                  if (i % param_.point_filter_num != 0)
                                      continue;
                                  // 距离滤波：过滤太远(>250m)或太近(<blind)的点
                                  double range = msg->points[i].x * msg->points[i].x + msg->points[i].y * msg->points[i].y +
                                                 msg->points[i].z * msg->points[i].z;
                                  if (range > 250 * 250 || range < param_.blind * param_.blind)
                                      continue;

                                  /**
                                   * Lambda函数：根据距离自适应计算邻域搜索半径
                                   * 近距离点用小半径（精细），远距离点用大半径（粗略）
                                   */
                                  auto adaptive_r = [&](pcl::PointXYZI &pt)
                                  {
                                      double max_dist = 30;   // 最大距离阈值
                                      double min_dist = 5;    // 最小距离阈值
                                      double max_r = 4;       // 最大搜索半径
                                      double min_r = 0.2;     // 最小搜索半径
                                      double dist = pt.getVector3fMap().norm();  // 计算点到原点的距离
                                      if (dist > max_dist)
                                          return max_r;
                                      if (dist < min_dist)
                                          return min_r;
                                      // 线性插值计算半径
                                      return (dist - min_dist) / (max_dist - min_dist) * (max_r - min_r);
                                  };

                                  // 检查点的有效性标签（Livox特有的tag字段）
                                  // tag & 0x30 == 0x10: 正常回波
                                  // tag & 0x30 == 0x00: 第一回波
                                  if (/*(msg->points[i].line < N_SCANS) &&*/ ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
                                  {
                                      PointType pt;
                                      pt.x = msg->points[i].x, pt.y = msg->points[i].y, pt.z = msg->points[i].z;
                                      pt.intensity = msg->points[i].reflectivity;
                                      double radius = adaptive_r(pt);  // 计算自适应搜索半径

                                      // 在KD树中进行半径搜索，寻找邻近点
                                      std::vector<int> pointIdxRSearch;
                                      std::vector<float> pointRSquaredDistance;
                                      // 只有找到超过5个邻近点时才计算法向量（确保PCA稳定）
                                      if (nano_kdtree.radiusSearch(pt, radius, pointIdxRSearch, pointRSquaredDistance) > 5)
                                      {
                                          // 收集邻近点坐标
                                          std::vector<Eigen::Vector3f> neighbors;
                                          for (size_t j = 0; j < pointIdxRSearch.size(); ++j)
                                          {
                                              neighbors.push_back(cloud->points[pointIdxRSearch[j]].getVector3fMap());
                                          }

                                          // 通过PCA计算法向量
                                          Eigen::Vector3f normal = computeNormal(neighbors);

                                          // 构建point3D结构体，填充所有必要信息
                                          point3D point_temp;
                                          point_temp.raw_point = Eigen::Vector3d(msg->points[i].x, msg->points[i].y, msg->points[i].z);
                                          point_temp.point = point_temp.raw_point;
                                          point_temp.raw_normal = normal.cast<double>();  // 法向量（LiDAR坐标系）
                                          point_temp.normal = point_temp.raw_normal;
                                          point_temp.relative_time = msg->points[i].offset_time / tm_scale;  // 相对时间（秒）
                                          point_temp.intensity = msg->points[i].reflectivity;

                                          point_temp.timestamp = headertime + point_temp.relative_time;  // 绝对时间戳
                                          point_temp.alpha_time = point_temp.relative_time / timespan_;  // 归一化时间[0,1]
                                          point_temp.timespan = timespan_;  // 帧时间跨度
                                          point_temp.ring = msg->points[i].line;  // 扫描线ID
                                          point_temp.lid = 1;  // LiDAR ID

                                          local_points.push_back(point_temp);
                                      }
                                  }
                              }
                              // 使用互斥锁将局部结果合并到全局输出
                              std::lock_guard<std::mutex> lock(valid_points_mutex);
                              cloud_out_.insert(cloud_out_.end(), local_points.begin(), local_points.end());
                          });

#else
        // 串行版本：单线程处理
        for (int i = 0; i < plsize; i++)
        {
            // 过滤无效点
            if (!(std::isfinite(msg->points[i].x) &&
                  std::isfinite(msg->points[i].y) &&
                  std::isfinite(msg->points[i].z)))
                continue;

            // 时间戳合法性检查
            if (msg->points[i].offset_time / tm_scale > timespan_)
                std::cout << "------" << __FUNCTION__ << ", " << __LINE__ << ", error timespan:" << timespan_ << " < " << msg->points[i].offset_time / tm_scale << std::endl;

            // 降采样：每隔point_filter_num个点取一个
            if (i % param_.point_filter_num != 0)
                continue;

            // if (msg->points[i].reflectivity < 5)
            //     continue;

            // 距离滤波：过滤太远(>250m)或太近(<blind)的点
            double range = msg->points[i].x * msg->points[i].x + msg->points[i].y * msg->points[i].y +
                           msg->points[i].z * msg->points[i].z;
            if (range > 250 * 250 || range < param_.blind * param_.blind)
                continue;

            /**
             * Lambda函数：根据距离自适应计算邻域搜索半径
             * 近距离点用小半径（精细），远距离点用大半径（粗略）
             */
            auto adaptive_r = [&](pcl::PointXYZI &pt)
            {
                double max_dist = 30;   // 最大距离阈值
                double min_dist = 5;    // 最小距离阈值
                double max_r = 4;       // 最大搜索半径
                double min_r = 0.2;     // 最小搜索半径
                double dist = pt.getVector3fMap().norm();  // 计算点到原点的距离
                if (dist > max_dist)
                    return max_r;
                if (dist < min_dist)
                    return min_r;
                // 线性插值计算半径
                return (dist - min_dist) / (max_dist - min_dist) * (max_r - min_r);
            };

            // 检查点的有效性标签
            if (/*(msg->points[i].line < N_SCANS) &&*/ ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
            {
                PointType pt;
                pt.x = msg->points[i].x, pt.y = msg->points[i].y, pt.z = msg->points[i].z;
                pt.intensity = msg->points[i].reflectivity;
                double radius = adaptive_r(pt);  // 计算自适应搜索半径

                // 在KD树中进行半径搜索
                std::vector<int> pointIdxRSearch;
                std::vector<float> pointRSquaredDistance;
                // 只有找到超过5个邻近点时才计算法向量
                if (nano_kdtree.radiusSearch(pt, radius, pointIdxRSearch, pointRSquaredDistance) > 5)
                {
                    // 收集邻近点坐标
                    std::vector<Eigen::Vector3f> neighbors;
                    for (size_t j = 0; j < pointIdxRSearch.size(); ++j)
                    {
                        neighbors.push_back(cloud->points[pointIdxRSearch[j]].getVector3fMap());
                    }

                    // 通过PCA计算法向量
                    Eigen::Vector3f normal = computeNormal(neighbors);

                    // 构建point3D结构体
                    point3D point_temp;
                    point_temp.raw_point = Eigen::Vector3d(msg->points[i].x, msg->points[i].y, msg->points[i].z);
                    point_temp.point = point_temp.raw_point;
                    point_temp.raw_normal = normal.cast<double>();  // 法向量
                    point_temp.normal = point_temp.raw_normal;
                    point_temp.relative_time = msg->points[i].offset_time / tm_scale;  // 相对时间（秒）
                    point_temp.intensity = msg->points[i].reflectivity;

                    point_temp.timestamp = headertime + point_temp.relative_time;  // 绝对时间戳
                    point_temp.alpha_time = point_temp.relative_time / timespan_;  // 归一化时间[0,1]
                    point_temp.timespan = timespan_;  // 帧时间跨度
                    point_temp.ring = msg->points[i].line;  // 扫描线ID
                    point_temp.lid = 1;  // LiDAR ID

                    cloud_out_.push_back(point_temp);
                }
            }
        }
#endif
        double t3 = tc.toc();  // 记录点云处理耗时
        // 输出各步骤耗时：t1=点云构建, t2=KD树构建, t3=法向量计算
        std::cout << "takes: " << t1 << ", " << t2 << ", " << t3 << std::endl;
    }

    void CloudConvert::Oust64Handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();
        pcl::PointCloud<ouster_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.size();
        cloud_out_.reserve(plsize);

        static double tm_scale = 1e9;

        double headertime = msg->header.stamp.toSec();
        // timespan_ = pl_orig.points.back().t / tm_scale;
        // std::cout << "span:" << timespan_ << ",0: " << pl_orig.points[0].t / tm_scale
        //           << " , 100: " << pl_orig.points[100].t / tm_scale << " , 1000: " << pl_orig.points[1000].t / tm_scale
        //           << " ," << pl_orig.points.back().t / tm_scale << std::endl;

        zjloc::common::TicToc tc;
        tc.tic();
        CloudPtr cloud(new PointCloudType);
        for (int i = 0; i < plsize; i++)
        {
            if (!(std::isfinite(pl_orig.points[i].x) &&
                  std::isfinite(pl_orig.points[i].y) &&
                  std::isfinite(pl_orig.points[i].z)))
                continue;
            PointType pt;
            pt.x = pl_orig.points[i].x;
            pt.y = pl_orig.points[i].y;
            pt.z = pl_orig.points[i].z;
            pt.intensity = pl_orig.points[i].intensity;
            cloud->push_back(pt);
        }
        double t1 = tc.toc();
        tc.tic();
        nanoflann::KdTreeFLANN<PointType> nano_kdtree;
        nano_kdtree.setInputCloud(cloud);
        double t2 = tc.toc();
        tc.tic();

#ifdef USE_TBB_PARAL
        std::mutex valid_points_mutex;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, plsize),
                          [&](const tbb::blocked_range<size_t> &r)
                          {
                              std::vector<point3D> local_points;
                              for (size_t i = r.begin(); i != r.end(); ++i)
                              {
                                  if (!(std::isfinite(pl_orig.points[i].x) &&
                                        std::isfinite(pl_orig.points[i].y) &&
                                        std::isfinite(pl_orig.points[i].z)))
                                      continue;
                                  if (i % param_.point_filter_num != 0)
                                      continue;
                                  double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                                                 pl_orig.points[i].z * pl_orig.points[i].z;
                                  if (range > 150 * 150 || range < param_.blind * param_.blind)
                                      continue;

                                  auto adaptive_r = [&](pcl::PointXYZI &pt)
                                  {
                                      double max_dist = 30;
                                      double min_dist = 5;
                                      double max_r = 4;
                                      double min_r = 0.2;
                                      double dist = pt.getVector3fMap().norm();
                                      if (dist > max_dist)
                                          return max_r;
                                      if (dist < min_dist)
                                          return min_r;
                                      return (dist - min_dist) / (max_dist - min_dist) * (max_r - min_r);
                                  };

                                  {
                                      PointType pt;
                                      pt.x = pl_orig.points[i].x, pt.y = pl_orig.points[i].y, pt.z = pl_orig.points[i].z;
                                      pt.intensity = pl_orig.points[i].intensity;
                                      double radius = adaptive_r(pt);

                                      std::vector<int> pointIdxRSearch;
                                      std::vector<float> pointRSquaredDistance;
                                      if (nano_kdtree.radiusSearch(pt, radius, pointIdxRSearch, pointRSquaredDistance) > 5)
                                      {
                                          std::vector<Eigen::Vector3f> neighbors;
                                          for (size_t j = 0; j < pointIdxRSearch.size(); ++j)
                                          {
                                              neighbors.push_back(cloud->points[pointIdxRSearch[j]].getVector3fMap());
                                          }

                                          Eigen::Vector3f normal = computeNormal(neighbors);

                                          point3D point_temp;
                                          point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
                                          point_temp.point = point_temp.raw_point;
                                          point_temp.raw_normal = normal.cast<double>();
                                          point_temp.normal = point_temp.raw_normal;
                                          point_temp.relative_time = pl_orig.points[i].t / tm_scale; // curvature unit: ms
                                          point_temp.intensity = pl_orig.points[i].intensity;

                                          point_temp.timestamp = headertime + point_temp.relative_time;
                                          point_temp.alpha_time = point_temp.relative_time / timespan_;
                                          point_temp.timespan = timespan_;
                                          point_temp.ring = pl_orig.points[i].ring;

                                          local_points.push_back(point_temp);
                                      }
                                  }
                              }
                              std::lock_guard<std::mutex> lock(valid_points_mutex);
                              cloud_out_.insert(cloud_out_.end(), local_points.begin(), local_points.end());
                          });
#else
        for (int i = 0; i < pl_orig.points.size(); i++)
        {
            if (!(std::isfinite(pl_orig.points[i].x) &&
                  std::isfinite(pl_orig.points[i].y) &&
                  std::isfinite(pl_orig.points[i].z)))
                continue;

            if (i % param_.point_filter_num != 0)
                continue;

            double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;
            if (range > 150 * 150 || range < param_.blind * param_.blind)
                continue;

            point3D point_temp;
            point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
            point_temp.point = point_temp.raw_point;
            point_temp.relative_time = pl_orig.points[i].t / tm_scale; // curvature unit: ms
            point_temp.intensity = pl_orig.points[i].intensity;

            point_temp.timestamp = headertime + point_temp.relative_time;
            point_temp.alpha_time = point_temp.relative_time / timespan_;
            point_temp.timespan = timespan_;
            point_temp.ring = pl_orig.points[i].ring;

            cloud_out_.push_back(point_temp);
        }
#endif
        double t3 = tc.toc();
        std::cout << "takes: " << t1 << ", " << t2 << ", " << t3 << std::endl;
    }

    void CloudConvert::RobosenseHandler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();
        pcl::PointCloud<robosense_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.size();
        cloud_out_.reserve(plsize);

        double headertime = msg->header.stamp.toSec();
        //  FIXME:  时间戳大于0.1
        auto time_list_robosense = [&](robosense_ros::Point &point_1, robosense_ros::Point &point_2)
        {
            return (point_1.timestamp < point_2.timestamp);
        };
        sort(pl_orig.points.begin(), pl_orig.points.end(), time_list_robosense);
        while (pl_orig.points[plsize - 1].timestamp - pl_orig.points[0].timestamp >= 0.1)
        {
            plsize--;
            pl_orig.points.pop_back();
        }

        timespan_ = pl_orig.points.back().timestamp - pl_orig.points[0].timestamp;

        // std::cout << timespan_ << std::endl;

        // std::cout << pl_orig.points[1].timestamp - pl_orig.points[0].timestamp << ", "
        //           << msg->header.stamp.toSec() - pl_orig.points[0].timestamp << ", "
        //           << msg->header.stamp.toSec() - pl_orig.points.back().timestamp << std::endl;

        for (int i = 0; i < pl_orig.points.size(); i++)
        {
            // if (i % param_.point_filter_num != 0)
            //     continue;
            if (!(std::isfinite(pl_orig.points[i].x) &&
                  std::isfinite(pl_orig.points[i].y) &&
                  std::isfinite(pl_orig.points[i].z)))
                continue;

            double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;
            if (range > 150 * 150 || range < param_.blind * param_.blind)
                continue;

            point3D point_temp;
            point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
            point_temp.point = point_temp.raw_point;
            point_temp.relative_time = pl_orig.points[i].timestamp - pl_orig.points[0].timestamp; // curvature unit: s
            point_temp.intensity = pl_orig.points[i].intensity;

            // point_temp.timestamp = headertime + point_temp.relative_time;
            point_temp.timestamp = pl_orig.points[i].timestamp;
            point_temp.alpha_time = point_temp.relative_time / timespan_;
            point_temp.timespan = timespan_;
            point_temp.ring = pl_orig.points[i].ring;
            point_temp.lid = 4;
            if (point_temp.alpha_time > 1 || point_temp.alpha_time < 0)
                std::cout << point_temp.alpha_time << ", this may error." << std::endl;

            cloud_out_.push_back(point_temp);
        }
    }

    void CloudConvert::VelodyneHandler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();

        pcl::PointCloud<velodyne_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.points.size();
        cloud_out_.reserve(plsize);

        double headertime = msg->header.stamp.toSec();

        static double tm_scale = 1; //   1e6 - nclt kaist or 1

        //  FIXME:  nclt 及kaist时间戳大于0.1
        auto time_list_velodyne = [&](velodyne_ros::Point &point_1, velodyne_ros::Point &point_2)
        {
            return (point_1.time < point_2.time);
        };
        sort(pl_orig.points.begin(), pl_orig.points.end(), time_list_velodyne);
        // std::cout << "cloud sizeZ:" << plsize << ",last t:" << pl_orig.points[plsize - 1].time << std::endl;
        while (pl_orig.points[plsize - 1].time / tm_scale >= 0.1)
        {
            plsize--;
            pl_orig.points.pop_back();
        }
        // timespan_ = -pl_orig.points[0].time / tm_scale;
        timespan_ = pl_orig.points.back().time / tm_scale;
        // std::cout << "span:" << timespan_ << ",0: " << pl_orig.points[0].time / tm_scale << " , 800: " << pl_orig.points[100].time / tm_scale << std::endl;

        zjloc::common::TicToc tc;
        tc.tic();
        CloudPtr cloud(new PointCloudType);
        for (int i = 0; i < plsize; i++)
        {
            if (!(std::isfinite(pl_orig.points[i].x) &&
                  std::isfinite(pl_orig.points[i].y) &&
                  std::isfinite(pl_orig.points[i].z)))
                continue;
            PointType pt;
            pt.x = pl_orig.points[i].x;
            pt.y = pl_orig.points[i].y;
            pt.z = pl_orig.points[i].z;
            pt.intensity = pl_orig.points[i].intensity;
            cloud->push_back(pt);
        }
        double t1 = tc.toc();
        tc.tic();
        nanoflann::KdTreeFLANN<PointType> nano_kdtree;
        nano_kdtree.setInputCloud(cloud);
        double t2 = tc.toc();
        tc.tic();
#ifdef USE_TBB_PARAL
        std::mutex valid_points_mutex;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, plsize),
                          [&](const tbb::blocked_range<size_t> &r)
                          {
                              std::vector<point3D> local_points;
                              for (size_t i = r.begin(); i != r.end(); ++i)
                              {
                                  if (!(std::isfinite(pl_orig.points[i].x) &&
                                        std::isfinite(pl_orig.points[i].y) &&
                                        std::isfinite(pl_orig.points[i].z)))
                                      continue;
                                  if (i % param_.point_filter_num != 0)
                                      continue;
                                  double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                                                 pl_orig.points[i].z * pl_orig.points[i].z;
                                  if (range > 150 * 150 || range < param_.blind * param_.blind)
                                      continue;

                                  auto adaptive_r = [&](pcl::PointXYZI &pt)
                                  {
                                      double max_dist = 30;
                                      double min_dist = 5;
                                      double max_r = 4;
                                      double min_r = 0.2;
                                      double dist = pt.getVector3fMap().norm();
                                      if (dist > max_dist)
                                          return max_r;
                                      if (dist < min_dist)
                                          return min_r;
                                      return (dist - min_dist) / (max_dist - min_dist) * (max_r - min_r);
                                  };

                                  {
                                      PointType pt;
                                      pt.x = pl_orig.points[i].x, pt.y = pl_orig.points[i].y, pt.z = pl_orig.points[i].z;
                                      pt.intensity = pl_orig.points[i].intensity;
                                      double radius = adaptive_r(pt);

                                      std::vector<int> pointIdxRSearch;
                                      std::vector<float> pointRSquaredDistance;
                                      if (nano_kdtree.radiusSearch(pt, radius, pointIdxRSearch, pointRSquaredDistance) > 5)
                                      {
                                          std::vector<Eigen::Vector3f> neighbors;
                                          for (size_t j = 0; j < pointIdxRSearch.size(); ++j)
                                          {
                                              neighbors.push_back(cloud->points[pointIdxRSearch[j]].getVector3fMap());
                                          }

                                          Eigen::Vector3f normal = computeNormal(neighbors);

                                          point3D point_temp;
                                          point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
                                          point_temp.point = point_temp.raw_point;
                                          point_temp.raw_normal = normal.cast<double>();
                                          point_temp.normal = point_temp.raw_normal;
                                          point_temp.relative_time = pl_orig.points[i].time / tm_scale; // curvature unit: ms
                                          point_temp.intensity = pl_orig.points[i].intensity;

                                          point_temp.timestamp = headertime + point_temp.relative_time;
                                          point_temp.alpha_time = point_temp.relative_time / timespan_;
                                          point_temp.timespan = timespan_;
                                          point_temp.ring = pl_orig.points[i].ring;
                                          point_temp.lid = 2;

                                          local_points.push_back(point_temp);
                                      }
                                  }
                              }
                              std::lock_guard<std::mutex> lock(valid_points_mutex);
                              cloud_out_.insert(cloud_out_.end(), local_points.begin(), local_points.end());
                          });
#else
        //  TODO:  need modified
        for (int i = 0; i < plsize; i++)
        {
            if (!(std::isfinite(pl_orig.points[i].x) &&
                  std::isfinite(pl_orig.points[i].y) &&
                  std::isfinite(pl_orig.points[i].z)))
                continue;

            if (i % param_.point_filter_num != 0)
                continue;

            double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;
            if (range > 150 * 150 || range < param_.blind * param_.blind)
                continue;

            point3D point_temp;
            point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
            point_temp.point = point_temp.raw_point;
            point_temp.relative_time = pl_orig.points[i].time / tm_scale; // curvature unit: s
            point_temp.intensity = pl_orig.points[i].intensity;

            point_temp.timestamp = headertime + point_temp.relative_time;
            point_temp.alpha_time = point_temp.relative_time / timespan_;
            point_temp.timespan = timespan_;
            point_temp.ring = pl_orig.points[i].ring;
            point_temp.lid = 2;

            cloud_out_.push_back(point_temp);
        }
#endif
        double t3 = tc.toc();
        std::cout << "takes: " << t1 << ", " << t2 << ", " << t3 << std::endl;
    }

    void CloudConvert::AVIAPCHandler(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        cloud_out_.clear();
        cloud_full_.clear();

        pcl::PointCloud<livox_ros::Point> pl_orig;
        pcl::fromROSMsg(*msg, pl_orig);
        int plsize = pl_orig.points.size();
        cloud_out_.reserve(plsize);

        double headertime = msg->header.stamp.toSec();

        static double tm_scale = 1e9;
        auto time_list_livox = [&](livox_ros::Point &point_1, livox_ros::Point &point_2)
        {
            return (point_1.timestamp < point_2.timestamp);
        };
        sort(pl_orig.points.begin(), pl_orig.points.end(), time_list_livox);
        // while (pl_orig.points[plsize - 1].timestamp - pl_orig.points[0].timestamp >= 0.1)
        // {
        //     plsize--;
        //     pl_orig.points.pop_back();
        // }
        timespan_ = (pl_orig.points.back().timestamp - pl_orig.points[0].timestamp) / tm_scale;

        zjloc::common::TicToc tc;
        tc.tic();
        CloudPtr cloud(new PointCloudType);
        for (int i = 0; i < plsize; i++)
        {
            if (!(std::isfinite(pl_orig.points[i].x) &&
                  std::isfinite(pl_orig.points[i].y) &&
                  std::isfinite(pl_orig.points[i].z)))
                continue;
            PointType pt;
            pt.x = pl_orig.points[i].x;
            pt.y = pl_orig.points[i].y;
            pt.z = pl_orig.points[i].z;
            pt.intensity = pl_orig.points[i].intensity;
            cloud->push_back(pt);
        }
        double t1 = tc.toc();
        tc.tic();
        nanoflann::KdTreeFLANN<PointType> nano_kdtree;
        nano_kdtree.setInputCloud(cloud);
        double t2 = tc.toc();
        tc.tic();
#ifdef USE_TBB_PARAL
        std::mutex valid_points_mutex;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, plsize),
                          [&](const tbb::blocked_range<size_t> &r)
                          {
                              std::vector<point3D> local_points;
                              for (size_t i = r.begin(); i != r.end(); ++i)
                              {
                                  if (!(std::isfinite(pl_orig.points[i].x) &&
                                        std::isfinite(pl_orig.points[i].y) &&
                                        std::isfinite(pl_orig.points[i].z)))
                                      continue;
                                  if (i % param_.point_filter_num != 0)
                                      continue;
                                  double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                                                 pl_orig.points[i].z * pl_orig.points[i].z;
                                  if (range > 150 * 150 || range < param_.blind * param_.blind)
                                      continue;

                                  auto adaptive_r = [&](pcl::PointXYZI &pt)
                                  {
                                      double max_dist = 30;
                                      double min_dist = 5;
                                      double max_r = 4;
                                      double min_r = 0.2;
                                      double dist = pt.getVector3fMap().norm();
                                      if (dist > max_dist)
                                          return max_r;
                                      if (dist < min_dist)
                                          return min_r;
                                      return (dist - min_dist) / (max_dist - min_dist) * (max_r - min_r);
                                  };

                                  {
                                      PointType pt;
                                      pt.x = pl_orig.points[i].x, pt.y = pl_orig.points[i].y, pt.z = pl_orig.points[i].z;
                                      pt.intensity = pl_orig.points[i].intensity;
                                      double radius = adaptive_r(pt);

                                      std::vector<int> pointIdxRSearch;
                                      std::vector<float> pointRSquaredDistance;
                                      if (nano_kdtree.radiusSearch(pt, radius, pointIdxRSearch, pointRSquaredDistance) > 5)
                                      {
                                          std::vector<Eigen::Vector3f> neighbors;
                                          for (size_t j = 0; j < pointIdxRSearch.size(); ++j)
                                          {
                                              neighbors.push_back(cloud->points[pointIdxRSearch[j]].getVector3fMap());
                                          }

                                          Eigen::Vector3f normal = computeNormal(neighbors);

                                          point3D point_temp;
                                          point_temp.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
                                          point_temp.point = point_temp.raw_point;
                                          point_temp.raw_normal = normal.cast<double>();
                                          point_temp.normal = point_temp.raw_normal;
                                          point_temp.relative_time = (pl_orig.points[i].timestamp - pl_orig.points[0].timestamp) / tm_scale;
                                          point_temp.intensity = pl_orig.points[i].intensity;

                                          point_temp.timestamp = headertime + point_temp.relative_time;
                                          point_temp.alpha_time = point_temp.relative_time / timespan_;
                                          point_temp.timespan = timespan_;
                                          point_temp.ring = pl_orig.points[i].line;
                                          point_temp.lid = 1;

                                          local_points.push_back(point_temp);
                                      }
                                  }
                              }
                              std::lock_guard<std::mutex> lock(valid_points_mutex);
                              cloud_out_.insert(cloud_out_.end(), local_points.begin(), local_points.end());
                          });
#else
        for (int i = 0; i < plsize; i++)
        {
            if (!(std::isfinite(pl_orig.points[i].x) && std::isfinite(pl_orig.points[i].y) && std::isfinite(pl_orig.points[i].z)))
                continue;

            if (i % param_.point_filter_num != 0)
                continue;

            double range = pl_orig.points[i].x * pl_orig.points[i].x +
                           pl_orig.points[i].y * pl_orig.points[i].y +
                           pl_orig.points[i].z * pl_orig.points[i].z;

            if (range > 120 * 120 || range < blind * blind)
                continue;

            point3D tPoint;
            tPoint.raw_point = Eigen::Vector3d(pl_orig.points[i].x, pl_orig.points[i].y, pl_orig.points[i].z);
            tPoint.point = tPoint.raw_point;
            tPoint.relative_time = (pl_orig.points[i].timestamp - pl_orig.points[0].timestamp) / tm_scale;
            tPoint.intensity = pl_orig.points[i].intensity;
            tPoint.timestamp = headertime + tPoint.relative_time;
            tPoint.alpha_time = tPoint.relative_time / timespan_;
            tPoint.timespan = timespan_;
            tPoint.ring = pl_orig.points[i].line;

            cloud_out_.push_back(tPoint);
        }
#endif
        // std::cout << "cloud size out: " << cloud_out_.size() << std::endl;
    }

    /**
     * [功能描述]: 通过PCA主成分分析计算邻域点的法向量
     *            法向量为协方差矩阵最小特征值对应的特征向量
     * @param neighbors: 邻域点集合（LiDAR坐标系下的3D点）
     * @return 法向量（归一化，指向LiDAR传感器方向）
     */
    Eigen::Vector3f CloudConvert::computeNormal(const std::vector<Eigen::Vector3f> &neighbors)
    {
        // ==================== 步骤1: 计算质心和协方差矩阵 ====================
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();   // 质心（均值）
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero(); // 协方差矩阵

        // 累加所有点的坐标和外积
        for (const auto &point : neighbors)
        {
            centroid += point;                       // 累加坐标，用于计算均值
            covariance += point * point.transpose(); // 累加外积 p * p^T
        }
        // 计算均值（质心）
        centroid /= (float)neighbors.size();
        // 计算协方差矩阵：Cov = E[p*p^T] - E[p]*E[p]^T
        // 这是协方差矩阵的简化计算公式，避免了两次遍历
        covariance /= (float)neighbors.size();
        covariance -= centroid * centroid.transpose();

        // ==================== 步骤2: 特征值分解（PCA） ====================
        // 使用自伴随特征值求解器（适用于对称矩阵）
        // 特征值按升序排列：eigenvalues[0] < eigenvalues[1] < eigenvalues[2]
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        // 最小特征值对应的特征向量即为法向量方向
        // 因为平面上点的分布在法向量方向上方差最小
        Eigen::Vector3f normal(solver.eigenvectors().col(0).normalized());
        
        // ==================== 步骤3: 法向量方向校正 ====================
        // 确保法向量指向LiDAR传感器（即原点方向）
        // -centroid 是从质心指向原点的向量
        // 如果法向量与该向量夹角 > 90°（点积 < 0），则翻转法向量
        if (normal.dot(-centroid) < 0)
            normal *= -1.0;
            
        return normal;
    }

    void CloudConvert::initFromConfig(const CVTParam &param)
    {
        param_.point_filter_num = param.point_filter_num;
        param_.blind = param.blind;
        param_.lidar_type = param.lidar_type;
        lidar_type_ = param.lidar_type;
        if (param_.lidar_type == LidarType::AVIA)
            LOG(INFO) << "Using AVIA Lidar";
        else if (param_.lidar_type == LidarType::VELO32)
            LOG(INFO) << "Using Velodyne 32 Lidar";
        else if (param_.lidar_type == LidarType::OUST64)
            LOG(INFO) << "Using OUST 64 Lidar";
        else if (param_.lidar_type == LidarType::ROBOSENSE16)
            LOG(INFO) << "Using Robosense 16 LIdar";
        else if (param_.lidar_type == LidarType::AVIA_PC)
            LOG(INFO) << "Using AVIA PC LIdar";
        else
            LOG(WARNING) << "unknown lidar_type";
    }

    void CloudConvert::LoadFromYAML(const std::string &yaml_file)
    {
        auto yaml = YAML::LoadFile(yaml_file);
        int lidar_type = yaml["preprocess"]["lidar_type"].as<int>();

        param_.point_filter_num = yaml["preprocess"]["point_filter_num"].as<int>();
        param_.blind = yaml["preprocess"]["blind"].as<double>();

        if (lidar_type == 1)
        {
            lidar_type_ = LidarType::AVIA;
            LOG(INFO) << "Using AVIA Lidar";
        }
        else if (lidar_type == 2)
        {
            lidar_type_ = LidarType::VELO32;
            LOG(INFO) << "Using Velodyne 32 Lidar";
        }
        else if (lidar_type == 3)
        {
            lidar_type_ = LidarType::OUST64;
            LOG(INFO) << "Using OUST 64 Lidar";
        }
        else if (lidar_type == 4)
        {
            lidar_type_ = LidarType::ROBOSENSE16;
            LOG(INFO) << "Using Robosense 16 LIdar";
        }
        else if (lidar_type == 5)
        {
            lidar_type_ = LidarType::AVIA_PC;
            LOG(INFO) << "Using AVIA PC LIdar";
        }
        else
        {
            LOG(WARNING) << "unknown lidar_type";
        }
        param_.lidar_type = lidar_type_;
    }

} // namespace zjloc
