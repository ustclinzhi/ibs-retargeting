import torch
import torch.nn as nn
from pytorch3d.ops import knn_points


class DifferentiableIBS(nn.Module):
    """
    可微分的插值边界表面(IBS)计算模块
    输入：
    - obj_points: 物体点云 [B, N, 3]
    - hand_points: 手部点云 [B, M, 3]
    输出：
    - ibs_points: IBS表面点 [B, K, 3]
    """
    def __init__(self, resolution=0.4, max_iter=40, n_samples=512):
        super().__init__()
        self.resolution = resolution      # 网格分辨率
        self.max_iter = max_iter          # 最大迭代次数
        self.n_samples = n_samples        # 初始采样点数
        
    def forward(self, obj_points, hand_points):
        B = obj_points.shape[0]
        
        # 1. 生成初始采样点（球体内均匀分布）
        init_points = self._generate_initial_points(hand_points, obj_points)  # 传入 hand_points 和 obj_points
        
        # 2. 迭代优化生成IBS点
        ibs_points = self._iterative_optimization(init_points, obj_points, hand_points)
        
        return ibs_points


    def _generate_initial_points(self, hand_points, obj_points):
        """在球体内生成均匀分布的初始采样点，球的半径为两个点云中心的距离"""
        B, M, _ = hand_points.shape
        device = hand_points.device

        # 计算手部和物体点云中心
        hand_center = torch.mean(hand_points, dim=1, keepdim=True)  # [B, 1, 3]
        obj_center = torch.mean(obj_points, dim=1, keepdim=True)      # [B, 1, 3]
        # 球心设为两者平均值
        center = 0.5 * (hand_center + obj_center)                     # [B, 1, 3]
        # 球半径为两个中心间的距离
        radius_val =0.8*torch.norm(hand_center - obj_center, dim=-1, keepdim=True)+0.05  # [B, 1, 1]

        # 生成均匀分布在球体内的采样点
        n = self.n_samples
        
        # 生成均匀分布的随机数
        u = torch.rand(B, n, 1, device=device)  # 用于半径
        v = torch.rand(B, n, 1, device=device)  # 用于角度1
        w = torch.rand(B, n, 1, device=device)  # 用于角度2
        
        # 体积均匀分布中半径应使用逆变换 u^(1/3)
        radius = radius_val * (u.pow(1/3))      # [B, n, 1]
        theta = torch.acos(2 * v - 1)            # 极角 [0, pi]
        phi = 2 * torch.pi * w                   # 方位角 [0, 2pi]
        
        # 转换为笛卡尔坐标
        x = radius * torch.sin(theta) * torch.cos(phi)
        y = radius * torch.sin(theta) * torch.sin(phi)
        z = radius * torch.cos(theta)
        
        points = torch.cat([x, y, z], dim=-1)    # [B, n, 3]
        
        return center + points
    def _iterative_optimization(self, points, obj_cloud, hand_cloud):
        """可导迭代优化过程"""
        for _ in range(self.max_iter):
            # 计算到手和物体的最近距离及法线
            hand_dists, hand_normals = self._compute_signed_distance(points, hand_cloud)
            obj_dists, obj_normals = self._compute_signed_distance(points, obj_cloud)
            
            # 计算符号距离差
            signed_dist = hand_dists - obj_dists  # [B, K]
            
            # 如果手与物体的最近距离差小于阈值，则对应点不更新
            tol = 1e-4
            move_mask = (torch.abs(signed_dist) >= tol).float()  # [B, K]
            
            # 计算移动方向
            direction = torch.where(
                (signed_dist.unsqueeze(-1) >= 0), 
                hand_normals, 
                obj_normals
            )  # [B, K, 3]
            
            # 计算权重
            dot_product = torch.sum(hand_normals * obj_normals, dim=-1)  # [B, K]
            denominator = torch.where(
                signed_dist >= 0,
                hand_dists - obj_dists * dot_product,
                obj_dists - hand_dists * dot_product
            )
            weight = 0.5 * (hand_dists + obj_dists) / (denominator + 1e-10)  # [B, K]
            
            # 计算移动量
            movement = weight.unsqueeze(-1) * direction * torch.abs(signed_dist).unsqueeze(-1)  # [B, K, 3]
            movement = movement * move_mask.unsqueeze(-1)  # 针对已接近的点不移动
            
            # 更新点位置
            points = points + movement
            
            # 提前终止条件
            if torch.max(torch.norm(movement, dim=-1)) < 1e-4:
                break
                
        return points

    def _compute_signed_distance(self, query_points, target_cloud):
        """ 使用pytorch3d的knn_points进行KNN搜索，返回最近邻点 """
        knn_result = knn_points(query_points, target_cloud, K=1, return_nn=True)
        # knn_result.dists: [B, K, 1]，平方距离，开根号得到欧氏距离
        dists = torch.sqrt(knn_result.dists.squeeze(-1))
        # knn_result.knn: [B, K, 1, 3]，移除多余维度，得到最近点坐标
        nearest_points = knn_result.knn.squeeze(2)  # [B, K, 3]
        # 计算法线：从 query_points 指向最近点
        normals = nearest_points - query_points
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-10)
        return dists, normals