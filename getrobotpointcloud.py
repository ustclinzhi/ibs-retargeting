import torch
import numpy as np
from utils.handmodel import get_handmodel
#python getrobotpointcloud.py
qpos=[
4.214643537998199463e-01,8.424473404884338379e-01,3.674198314547538757e-02,1.124095082283020020e+00,3.951686024665832520e-01,-9.135456383228302002e-02,-4.786605238914489746e-01,-4.227935373783111572e-01,-5.250931531190872192e-02,-6.504364311695098877e-03,1.668661385774612427e-01,2.186184376478195190e-02,1.122261956334114075e-01,5.432741641998291016e-01,5.615013837814331055e-01,7.027676105499267578e-01,1.558304578065872192e-01,8.611050248146057129e-01,5.787225961685180664e-01,6.200705766677856445e-01,7.217127680778503418e-01,6.008514165878295898e-01,2.022181600332260132e-01,6.744181513786315918e-01,6.786985993385314941e-01,7.842558026313781738e-01,8.582049608230590820e-01,6.115192547440528870e-02,9.264037013053894043e-01,6.723133921623229980e-01
        ]

def save_point_cloud_as_obj(point_cloud: np.ndarray, filename: str):
    """
    将点云保存为 OBJ 文件，每一行代表一个顶点，格式为 "v x y z"
    
    Args:
        point_cloud: numpy 数组，形状为 (N, 3)
        filename: 输出文件路径，例如 "hand_pcd.obj"
    """
    with open(filename, 'w') as f:
        for pt in point_cloud:
            f.write("v {:.6f} {:.6f} {:.6f}\n".format(pt[0], pt[1], pt[2]))
    print(f"Hand point cloud saved to {filename}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取手部模型；根据需要调整 batch_size、hand_scale 和 robot 类型
    hand_model = get_handmodel(batch_size=1, device=device, hand_scale=1.0, robot='shadowhand')
    
    # 模拟一个 qpos（pred_x0），要求尺寸与 update_kinematics(q) 时一致
    newq=qpos[:6]+qpos[6:9]+[qpos[13],qpos[18],qpos[23],qpos[9],qpos[14],qpos[19],qpos[24],qpos[10],qpos[15],qpos[20],qpos[25],qpos[11],qpos[16],qpos[21],qpos[26],qpos[28],qpos[12],qpos[17],qpos[22],qpos[27],qpos[29]]

    dummy_qpos = torch.tensor(newq, device=device, dtype=torch.float32).unsqueeze(0)
    
    # 更新手部运动学状态
    hand_model.update_kinematics(q=dummy_qpos)
    
    # 获取手部表面点云，注意该函数内部调用了更新的运动学状态
    hand_pcd = hand_model.get_surface_points(q=dummy_qpos).to(dtype=torch.float32)
    # 提取第一组（batch）点云，shape 为 [N, 3]
    hand_pcd_np = hand_pcd.detach().cpu().numpy()[0]
    
    # 保存点云为 OBJ 文件
    output_filename = "/mnt/e/IBS/related-work/dex-retargeting/guiji/datatest/对比点云/datatrain5.2.13.34-modeloutput.obj"
    save_point_cloud_as_obj(hand_pcd_np, output_filename)

if __name__ == '__main__':
    main()