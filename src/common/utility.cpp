#include "utility.h"

double AngularDistance(const Eigen::Quaterniond &q_a, const Eigen::Quaterniond &q_b)
{
     Eigen::Matrix3d rota = q_a.toRotationMatrix();
     Eigen::Matrix3d rotb = q_b.toRotationMatrix();

     double norm = ((rota * rotb.transpose()).trace() - 1) / 2;
     norm = std::acos(norm) * 180 / M_PI;
     return norm;
}

/**
 * [功能描述]: 体素降采样函数，将点云按体素网格划分，每个体素只保留一个点
 *            用于减少点云数量，同时保持空间分布均匀
 * @param frame: 输入/输出点云，函数执行后会被降采样结果覆盖
 * @param size_voxel: 体素边长大小，单位与点云坐标一致（通常为米）
 */
void subSampleFrame(std::vector<point3D> &frame, double size_voxel)
{
     // 创建体素网格哈希表：key为体素索引，value为该体素内的所有点
     std::tr1::unordered_map<voxel, std::vector<point3D>, std::hash<voxel>> grid;
     
     // 遍历所有点，将每个点分配到对应的体素中
     for (int i = 0; i < (int)frame.size(); i++)
     {
          // 计算点所属体素的三维索引 (kx, ky, kz)
          // 通过坐标除以体素大小并取整得到体素索引
          auto kx = static_cast<short>(frame[i].point[0] / size_voxel);  // x方向体素索引
          auto ky = static_cast<short>(frame[i].point[1] / size_voxel);  // y方向体素索引
          auto kz = static_cast<short>(frame[i].point[2] / size_voxel);  // z方向体素索引
          // 将点添加到对应体素的点集合中
          grid[voxel(kx, ky, kz)].push_back(frame[i]);
     }
     
     // 清空原始点云，准备存储降采样结果
     frame.resize(0);
     int step = 0;
     
     // 遍历所有非空体素，每个体素只取第一个点作为代表点
     for (const auto &n : grid)
     {
          if (n.second.size() > 0)
          {
               // 取该体素内的第一个点作为代表点
               frame.push_back(n.second[0]);
               step++;
          }
     }
}

/**
 * [功能描述]: 网格降采样函数，对输入点云进行体素化降采样，提取关键点
 * @param frame: 输入点云，原始的3D点集合（只读）
 * @param keypoints: 输出关键点，降采样后的点集合（输出参数）
 * @param size_voxel_subsampling: 体素大小，控制降采样的分辨率，值越大点越稀疏
 */
void gridSampling(const std::vector<point3D> &frame, std::vector<point3D> &keypoints, double size_voxel_subsampling)
{
     // 清空输出关键点容器
     keypoints.resize(0);
     
     // 创建临时副本，因为subSampleFrame会修改输入数据
     std::vector<point3D> frame_sub;
     frame_sub.resize(frame.size());
     
     // 将原始点云复制到临时副本中
     for (int i = 0; i < (int)frame_sub.size(); i++)
     {
          frame_sub[i] = frame[i];
     }
     
     // 执行体素降采样，frame_sub中会保留降采样后的点
     subSampleFrame(frame_sub, size_voxel_subsampling);
     
     // 预分配内存，提高push_back效率
     keypoints.reserve(frame_sub.size());
     
     // 将降采样后的点复制到输出关键点容器中
     for (int i = 0; i < (int)frame_sub.size(); i++)
     {
          keypoints.push_back(frame_sub[i]);
     }
}

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
     Eigen::Vector3d n = R.col(0);
     Eigen::Vector3d o = R.col(1);
     Eigen::Vector3d a = R.col(2);

     Eigen::Vector3d ypr(3);
     double y = atan2(n(1), n(0));
     double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
     double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
     ypr(0) = y;
     ypr(1) = p;
     ypr(2) = r;

     return ypr / M_PI * 180.0;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
{
     typedef typename Derived::Scalar Scalar_t;

     Scalar_t y = ypr(0) / 180.0 * M_PI;
     Scalar_t p = ypr(1) / 180.0 * M_PI;
     Scalar_t r = ypr(2) / 180.0 * M_PI;

     Eigen::Matrix<Scalar_t, 3, 3> Rz;
     Rz << cos(y), -sin(y), 0,
         sin(y), cos(y), 0,
         0, 0, 1;

     Eigen::Matrix<Scalar_t, 3, 3> Ry;
     Ry << cos(p), 0., sin(p),
         0., 1., 0.,
         -sin(p), 0., cos(p);

     Eigen::Matrix<Scalar_t, 3, 3> Rx;
     Rx << 1., 0., 0.,
         0., cos(r), -sin(r),
         0., sin(r), cos(r);

     return Rz * Ry * Rx;
}

Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
{
     Eigen::Matrix3d R0;
     Eigen::Vector3d ng1 = g.normalized();
     Eigen::Vector3d ng2{0, 0, 1.0};
     R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
     double yaw = R2ypr(R0).x();
     R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
     // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
     return R0;
}