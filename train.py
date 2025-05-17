import os
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from utils.misc import compute_model_dim
from utils.io import mkdir_if_not_exists
from utils.plot import Ploter
# from datasets.base import create_dataset
# from datasets.misc import collate_fn_general, collate_fn_squeeze_pcd_batch
from models.base import create_model
from models.visualizer import create_visualizer
# from tqdm import tqdm  
# from functools import partial
import json
import numpy as np
#cd /home/lz/DexGrasp-Anything
#bash scripts/grasp_gen_ur/train.sh ${EXP_NAME}
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
import time
class GraspDatasettest(Dataset):
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
            
            'x': torch.tensor(item["x"], dtype=torch.float32),
            'pos': torch.tensor(points, dtype=torch.float32),
            'ibspointcloud': torch.tensor(sampled_vertices, dtype=torch.float32),
            'objpointcloud': torch.tensor(transformed_point_cloud, dtype=torch.float32),
            
        }
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
            'x': torch.tensor(item["x"], dtype=torch.float32),
            'pos': torch.tensor(points, dtype=torch.float32),
            'ibspointcloud': torch.tensor(sampled_vertices, dtype=torch.float32),
            'objpointcloud': torch.tensor(transformed_point_cloud, dtype=torch.float32),
            
        }


def save_ckpt(model: torch.nn.Module, epoch: int, step: int, path: str, save_scene_model: bool) -> None:
    """ Save current model and corresponding data

    Args:
        model: best model
        epoch: best epoch
        step: current step
        path: save path
        save_scene_model: if save scene_model
    """
    saved_state_dict = {}
    model_state_dict = model.state_dict()
    for key in model_state_dict:
        ## if use frozen pretrained scene model, we can avoid saving scene model to save space
        if 'scene_model' in key and not save_scene_model:
            continue

        saved_state_dict[key] = model_state_dict[key]
    
    logger.info('Saving model!!!' + ('[ALL]' if save_scene_model else '[Except SceneModel]'))
    torch.save({
        'model': saved_state_dict,
        'epoch': epoch, 'step': step,
    }, path)

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
def train(cfg: DictConfig) -> None:
    """ training portal

    Args:
        cfg: configuration dict
    """
    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    print('device:', device)
    ## prepare dataset for train and test
    # datasets = {
    #     'train': create_dataset(cfg.task.dataset, 'train', cfg.slurm),
    # }
    # if cfg.task.visualizer.visualize:
    #     datasets['test_for_vis'] = create_dataset(cfg.task.dataset, 'test', cfg.slurm, case_only=True)
    # for subset, dataset in datasets.items():
    #     logger.info(f'Load {subset} dataset size: {len(dataset)}')
    
    # if cfg.model.scene_model.name == 'PointTransformer':
    #     collate_fn = partial(collate_fn_squeeze_pcd_batch, use_llm=cfg.model.use_llm)
    # else:
    #     collate_fn = partial(collate_fn_general, use_llm=cfg.model.use_llm)
    
    # dataloaders = {
    #     'train': datasets['train'].get_dataloader(
    #         batch_size=cfg.task.train.batch_size,
    #         collate_fn=collate_fn,
    #         num_workers=cfg.task.train.num_workers,
    #         pin_memory=True,
    #         shuffle=True,
    #     ),
    # }
    writer = SummaryWriter(log_dir=cfg.tb_dir)
    Ploter.setWriter(writer)
    print("start loading dataset//////////////////////")
    dataset = GraspDataset("/mnt/e/IBS/related-work/dex-retargeting/guiji/data/totaldata.json")
    datasettest= GraspDatasettest("/mnt/e/IBS/related-work/dex-retargeting/guiji/datatest/totaldata.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False,drop_last=True)
    dataloadertest = DataLoader(datasettest, batch_size=32, shuffle=True, num_workers=4, pin_memory=False,drop_last=True)
    print("end loading dataset//////////////////////")
    ## create model and optimizer
    model = create_model(cfg, slurm=cfg.slurm, device=device)
    model.to(device=device)
    #load_ckpt(model, path="/home/lz/DexGrasp-Anything/outputs/ycb90ibs50/ckpts/model_359loss0.10310035376724871.pth")
    
    
    params = []
    nparams = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            nparams.append(p.nelement())
    
    params_group = [
        {'params': params, 'lr': cfg.task.lr},
    ]
    optimizer = torch.optim.Adam(params_group) # use adam optimizer in default
    logger.info(f'{len(params)} parameters for optimization.')
    logger.info(f'total model size is {sum(nparams)}.')
    ## create visualizer if visualize in training process
    if cfg.task.visualizer.visualize:
        visualizer = create_visualizer(cfg.task.visualizer)
   
    ## start training
    step = 0
   
    
    for epoch in range(0, cfg.task.train.num_epochs):
        model.train()
        epoch_loss = 0.0    # 记录当前 epoch 累计 loss
        num_batches = 0     # 当前 epoch 的 batch 数量
        it = 0
        # 对每个 batch 循环
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            batch['epoch'] = epoch
            outputs = model(batch)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_loss = loss.item()   
            
            epoch_loss += total_loss
            num_batches += 1

            ## plot loss
            if (step + 1) % cfg.task.train.log_step == 0:
                writer.add_scalar('train/loss', total_loss, step)
                log_str = f'[TRAIN] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f}'
                logger.info(log_str)
                for key in outputs:
                    val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
                    Ploter.write({
                        f'train/{key}': {'plot': True, 'value': val, 'step': step},
                        'train/epoch': {'plot': True, 'value': epoch, 'step': step},
                    })

            step += 1
            it += 1
            print(f'[TRAIN] ==> Epoch: {epoch+1:3d} | Iter: {it+1:5d} | Step: {step+1:7d} | Loss: {total_loss:.3f},IBSloss:{outputs["ibs_loss"]}')
        
        # 计算并打印当前 epoch 的平均 loss
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f'[EPOCH SUMMARY] Epoch: {epoch+1:3d} | Average Loss: {avg_loss:.3f}')
        if (epoch+1) % 30 == 0:

            time.sleep(180)
        
        ## save ckpt in epoch
        if (epoch + 1) % 5 == 0 :
           
            save_path = os.path.join(
                cfg.ckpt_dir, 
                f'model_{epoch}loss{avg_loss:.4f}.pth' if cfg.save_model_seperately else 'model.pth'
            )
            save_ckpt(
                model=model, epoch=epoch, step=step, path=save_path,
                save_scene_model=cfg.save_scene_model,
            )
    writer.close()

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
            ibsloss=outputs["ibs_loss"]

            # 累加损失
            total_eval_loss += loss.item()
            total_eval_ibsloss += ibsloss.item()


            # 记录批次数量
            num_eval_batches += 1
            if num_eval_batches>30:
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



@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    ## compute modeling dimension according to task
    cfg.model.d_x = compute_model_dim(cfg.task)
    if os.environ.get('SLURM') is not None:
        cfg.slurm = True # update slurm config
        logger.remove(handler_id=0) # remove default handler
    print("ok--2  ")
    ## set output logger and tensorboard
    logger.add(cfg.exp_dir + '/runtime.log')

    mkdir_if_not_exists(cfg.tb_dir)
    mkdir_if_not_exists(cfg.vis_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    print("ok--3  ")
    writer = SummaryWriter(log_dir=cfg.tb_dir)
    Ploter.setWriter(writer)
   
    ## Begin training progress
    logger.info('Configuration: \n' + OmegaConf.to_yaml(cfg))
    logger.info('Begin training..')

    train(cfg) # training portal

    ## Training is over!
    writer.close() # close summarywriter and flush all data to disk
    logger.info('End training..')

if __name__ == '__main__':
    print("ok--1  ")
    main()
