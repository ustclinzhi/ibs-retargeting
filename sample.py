import os
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from termcolor import cprint
from utils.misc import timestamp_str, compute_model_dim
from utils.io import mkdir_if_not_exists
# from datasets.base import create_dataset
from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
import json
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
import csv
#bash scripts/grasp_gen_ur/sample.sh /home/lz/DexGrasp-Anything/outputs/ibs0 [OPT]
#bash scripts/grasp_gen_ur/sample.sh /home/lz/DexGrasp-Anything/outputs/ycb90ibs0 [OPT]

modelpath = "model_734loss0.0632.pth"

def load_ckpt(model: torch.nn.Module, path: str) -> None:
    """ load ckpt for current model

    Args:
        model: current model
        path: save path
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path)['model']
    model_state_dict = model.state_dict()

    for key in model_state_dict:
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
        ## model is trained with ddm
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
    
    model.load_state_dict(model_state_dict)

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True # update slurm config
    
    ## set output dir
    eval_dir = os.path.join(cfg.exp_dir, 'eval')
    mkdir_if_not_exists(eval_dir)
    vis_dir = os.path.join(eval_dir, 
        'series' if cfg.task.visualizer.vis_denoising else 'final', timestamp_str())

    logger.add(vis_dir + '/sample.log') # set logger file
    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg)) # record configuration

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    ## prepare dataset for visual evaluation
    ## only load scene
    # datasets = {
    #     'test': create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True),
    # }
    # for subset, dataset in datasets.items():
    #     logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    if cfg.model.scene_model.name == 'PointTransformer':
        collate_fn = collate_fn_squeeze_pcd_batch
    else:
        collate_fn = collate_fn_general
    
    # dataloaders = {
    #     'test': datasets['test'].get_dataloader(
    #         batch_size=cfg.task.test.batch_size,
    #         collate_fn=collate_fn,
    #         num_workers=cfg.task.test.num_workers,
    #         pin_memory=True,
    #         shuffle=True,
    #     )
    # }
    
    ## create model and load ckpt
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    ## if your models are seperately saved in each epoch, you need to change the model path manually
    ckpt_path = os.path.join(cfg.ckpt_dir, 'model.pth')
    if not os.path.exists(ckpt_path):
        
        
        ckpt_path = os.path.join(cfg.ckpt_dir, modelpath)
        logger.info(f"Using the latest checkpoint: {ckpt_path}")
    load_ckpt(model, path=ckpt_path)
    if cfg.diffuser.sample.use_dpmsolver:
        cprint("\033[1;35m[INFO] Using DPMSolver++ for sampling.\033[0m", "magenta")  
    else:
        cprint("\033[1;35m[INFO] Using DDPM for sampling.\033[0m", "magenta")  
    ## create visualizer and visualize
    dataset = GraspDataset("/mnt/e/IBS/related-work/dex-retargeting/guiji/datatest/totaldata.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    totalloss=0
    inum=0
    totaljointloss=0
    if 1:
        for batch in dataloader:
            inum+=1

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()} 

            outputs = model.sample(batch, k=1)
            
            result = outputs[:, :, -1, :].squeeze(1)
            if 0:
                for i in range(result.shape[0]):
                
                    json_filepath = os.path.join(batch['datapath'][i], 'sampledatafinal_ycb90_ibs50_epoch_750.json')
                    if not os.path.exists(json_filepath):
                        # 如果文件不存在，写入默认数据（此处为空字典，也可以根据需要调整）
                        default_data = {}
                        # 创建文件夹（如果父目录不存在）
                        os.makedirs(os.path.dirname(json_filepath), exist_ok=True)
                        with open(json_filepath, "w") as f:
                            json.dump(default_data, f, indent=4)
                        
                    else:
                        pass
                    with open(json_filepath, "r+") as f:
                        dataqpos = json.load(f)         # 读取现有数据（应为字典）
                        dataqpos['robotpose'+str(batch['sampledataidj'][i].item())] = result[i].detach().cpu().numpy().tolist()  # 添加新的键值对
                        f.seek(0)                    # 回到文件开头，准备写入
                        json.dump(dataqpos, f, indent=4)   # 写入更新后的数据
                        f.truncate() 

                
                
                                    
          
            
                
            loss=torch.mean(torch.abs( result[:,:6] - batch["realx"][:,:6]))
            jointloss=torch.mean(torch.abs( result[:,6:] - batch["realx"][:,6:]))
            print("6dloss: ", loss.item())
            print("joint loss: ", jointloss.item())
            totaljointloss+=jointloss
            totalloss+=loss
    
        print("total average 6dpos qpos dist: ", totalloss/inum)
        print("total average joint  qpos dist: ", totaljointloss/inum)
    if 0:
        evaluate(model, dataloader, device)
    torch.cuda.empty_cache()
def evaluate(model, dataloader, device):
    """
    在给定的 dataloader 上评估模型的损失。
    """
    model.eval() # !! 将模型设置为评估模式 !!

    total_eval_loss = 0.0
    num_eval_batches = 0
    total_eval_ibsloss=0.0
    # !! 关闭梯度计算 !! 在评估阶段不需要计算梯度
    with torch.no_grad():
        # 循环遍历测试/验证数据集的 batch
        for batch in dataloader:
            # 将数据转移到指定设备
            batch = {k: v.to(device) for k, v in batch.items()}

            # 前向传播，计算模型的输出
            # 注意：在这里，model(batch) 应该仍然返回包含 'loss' 的字典
            outputs = model(batch)

            # 获取损失值
            loss = outputs['loss']
            
            # 累加损失
            total_eval_loss += loss.item()
            


            # 记录批次数量
            num_eval_batches += 1
            if num_eval_batches>20:
                break

    # 计算平均损失
    avg_eval_loss = total_eval_loss / num_eval_batches if num_eval_batches > 0 else 0.0
    avg_eval_ibsloss= total_eval_ibsloss / num_eval_batches if num_eval_batches > 0 else 0.0

    # 打印或记录评估结果
    print(f'[EVALUATION] Finished evaluation. Average Loss: {avg_eval_loss:.3f}')
    print(f'[EVALUATION] Finished evaluation. Average ibs Loss: {avg_eval_ibsloss:.3f}')
    # logger.info(f'[EVALUATION] Average Loss: {avg_eval_loss:.3f}') # 如果使用 logger

    # 你可以选择返回平均损失或其他评估指标
    return avg_eval_loss
class GraspDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.lzdata = list(json.load(f).values())
        
    def __len__(self):
        return len(self.lzdata)
    
    def __getitem__(self, idx):
        item = self.lzdata[idx]
        npz_data = np.load(item["pos"])
        points = npz_data[npz_data.files[0]][:2048, :3]
        objpose = torch.tensor(item["objpose"], dtype=torch.float32)
        objnpzdata = np.load(item["objnpz"])
        point_cloud = objnpzdata['points']  # 假设点云数据存储在 'points' 键下

        pose =  objpose.to(torch.float32).cpu().numpy()

        # 提取平移与四元数
        translation = pose[:3]
        quaternion = pose[[4,5,6,3]]

        rotation = R.from_quat(quaternion)
        rotated_point_cloud = rotation.apply(point_cloud)
        transformed_point_cloud = rotated_point_cloud + translation

        
        vertices = []
        with open(item["handibs"], 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                        vertices.append(vertex)
        if vertices:
           
            sampled_vertices = vertices
            
        return {
            'x':torch.rand(30,dtype=torch.float32),
           

            'realx': torch.tensor(item["x"], dtype=torch.float32),
            'pos': torch.tensor(points, dtype=torch.float32),
            'pospath': item["pos"],
            'ibspointcloud': torch.tensor(sampled_vertices, dtype=torch.float32),
            'objpointcloud': torch.tensor(transformed_point_cloud, dtype=torch.float32),
            # 'datapath': item['datapath'],
            # 'sampledataidj': item[ 'sampledataidj'],
            # 'sampledataidid': item[ 'sampledataidid'],
            
        }
if __name__ == '__main__':
    ## set random seed
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()