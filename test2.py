import torch
import numpy as np
from utils.ibs import DifferentiableIBS
import json
from scipy.spatial.transform import Rotation as R
import os
# 检查 CUDA 可用性，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ibs_model = DifferentiableIBS(n_samples=4096, max_iter=100, resolution=0.3).to(device)

# 生成测试用的随机物体点云和手部点云
# 物体点云 shape: [1, N, 3], 手部点云 shape: [1, M, 3]
for j in range(1,15):
    if j in [ ]:
        continue
    dataobjpoint = np.load(f'/mnt/e/IBS/related-work/dex-retargeting/guiji/data/data_{j}/objpoint.npz')
    point_cloud = dataobjpoint['points']  # 假设点云数据存储在 'points' 键下
    with open(f"/mnt/e/IBS/related-work/dex-retargeting/guiji/data/data_{j}/objmesh_pose.json", 'r') as file:
        # 读取 JSON 数据并解析为 Python 对象
        data = json.load(file)
    # 给定的位姿 [x, y, z, qw, qx, qy, qz]
    num=data["total_frame"]
    for i in range(1,num + 1):
        pose = np.array(data[f"objectpose{i}"])


        # 提取旋转四元数
        translation = pose[:3]
        quaternion = pose[[4,5,6,3]]

        # 创建旋转对象
        rotation = R.from_quat(quaternion)

        # 转换点云
        rotated_point_cloud = rotation.apply(point_cloud)
        obj_point_cloud = torch.from_numpy(rotated_point_cloud + translation).unsqueeze(0).to(device).float()
        datahand = np.load(f'/mnt/e/IBS/related-work/dex-retargeting/guiji/data/data_{j}/handpoint/handpoint_{i}.npz')
        handpoint_cloud = torch.from_numpy(datahand[datahand.files[0]]).unsqueeze(0).to(device).float()


        # 初始化 DifferentiableIBS 模块
       

        # 前向计算 IBS 点云
        ibs_points = ibs_model(obj_point_cloud, handpoint_cloud)  # 输出形状: [1, K, 3]

        # 将输出的 IBS 点云转换为 numpy 格式，并 squeeze 去掉批次维度
        ibs_points_np = ibs_points.squeeze(0).cpu().detach().numpy()

        # 保存为 OBJ 文件（每行一个顶点）
        ibsfoldpath = f"/mnt/e/IBS/related-work/dex-retargeting/guiji/data/data_{j}/ibs"
        # 创建文件夹（如果不存在）
        os.makedirs(ibsfoldpath, exist_ok=True)
        output_obj_path = f"/mnt/e/IBS/related-work/dex-retargeting/guiji/data/data_{j}/ibs/ibs{i}.obj"
        with open(output_obj_path, "w") as f:
            for pt in ibs_points_np:
                f.write("v {} {} {}\n".format(pt[0], pt[1], pt[2]))
        print(f"ibs {j}-{i}\n")