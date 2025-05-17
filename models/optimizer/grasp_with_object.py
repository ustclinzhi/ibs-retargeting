from typing import Dict
import os
import numpy as np
import torch
from omegaconf import DictConfig
from utils.handmodel import get_handmodel
from models.optimizer.optimizer import Optimizer
from models.base import OPTIMIZER
import pickle
import torch.functional as F
from utils.handmodel import get_handmodel, ERF_loss, SPF_loss, SRF_loss
from einops import rearrange
from termcolor import cprint
from utils.ibs import *
from chamfer_distance.chamfer_distance import ChamferDistance as chamfer_dist
@OPTIMIZER.register()
class GraspWithObject(Optimizer):

    _N_OBJ = 4096
    _joint_angle_lower = torch.tensor([-0.5235988, -0.7853982, -0.43633232, 0., 0., 0., -0.43633232, 0., 0., 0.,
                                       -0.43633232, 0., 0., 0., 0., -0.43633232, 0., 0., 0., -1.047, 0., -0.2618,
                                       -0.5237, 0.], device='cuda')
    _joint_angle_upper = torch.tensor([0.17453292, 0.61086524, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 0.43633232,
                                       1.5707964, 1.5707964, 1.5707964, 0.43633232, 1.5707964, 1.5707964, 1.5707964,
                                       0.6981317, 0.43633232, 1.5707964, 1.5707964, 1.5707964, 1.047, 1.309, 0.2618,
                                       0.5237, 1.], device='cuda')

    _global_trans_lower = torch.tensor([-0.13128923, -0.10665303, -0.45753425], device='cuda')
    _global_trans_upper = torch.tensor([0.12772022, 0.22954416, -0.21764427], device='cuda')

    _NORMALIZE_LOWER = -1.
    _NORMALIZE_UPPER = 1.
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self._BATCH_SIZE = cfg.batch_size
        self.slurm = slurm
        self.scale = cfg.grad_scale
        self.clip_grad_by_value = cfg.clip_grad_by_value
        self.modeling_keys = cfg.modeling_keys
        self.normalize_x = cfg.normalize_x
        self.normalize_x_trans = cfg.normalize_x_trans
        self.asset_dir = cfg.asset_dir_slrum if self.slurm else cfg.asset_dir
        self.hand_model = get_handmodel(batch_size=self._BATCH_SIZE, device=self.device)
        self.guidance_scale = cfg.guidance_scale
        self.weight_ERF_loss = cfg.loss_weights.ERF_loss
        self.weight_SPF_loss = cfg.loss_weights.SPF_loss
        self.weight_SRF_loss = cfg.loss_weights.SRF_loss
        self.weight_ibs_loss = cfg.loss_weights.ibs_loss
        self.ibs_model = DifferentiableIBS(n_samples=4096)
        cprint(f"\033[1;33m[INFO] ERF weight: {self.weight_ERF_loss}\033[0m", "yellow")
        cprint(f"\033[1;33m[INFO] SPF weight: {self.weight_SPF_loss}\033[0m", "yellow")
        cprint(f"\033[1;33m[INFO] SRF weight: {self.weight_SRF_loss}\033[0m", "yellow")
        cprint("\033[1;33m[INFO] Modifying ERF and SPF weights can effectively control the generated results you desire.\033[0m", "yellow")

    def optimize(self, x: torch.Tensor, data: Dict, t: int) -> torch.Tensor:
        """ Compute gradient for optimizer constraint

        Args:
            x: the denosied signal at current step, which is detached and is required grad
            data: data dict that provides original data
            t: sample time

        Return:
            The optimizer objective value of current step
        """
        # obj_pcd = data['pos'].to(self.device)
        # self.hand_model.update_kinematics(q = x)
        # hand_pcd = self.hand_model.get_surface_points(q= x).to(dtype=torch.float32)
        # normal = data['normal'].clone().detach().requires_grad_(True).to(self.device)
        # obj_pcd = rearrange(obj_pcd, '(b n) c -> b n c', b=self._BATCH_SIZE, n=normal.shape[1])

        # obj_pcd_nor = torch.cat((obj_pcd, normal), dim=-1).to(dtype=torch.float32)
        # ERF_loss_value = ERF_loss(obj_pcd_nor, hand_pcd)
        # dis_keypoint = self.hand_model.get_dis_keypoints(q= x)
        # SPF_loss_value = SPF_loss(dis_keypoint, obj_pcd)
        # hand_keypoint = self.hand_model.get_keypoints(q= x)
        # SRF_loss_value = SRF_loss(hand_keypoint)
        #return  self.weight_SRF_loss * SRF_loss_value  + self.weight_SPF_loss * SPF_loss_value + self.weight_ERF_loss * ERF_loss_value
        qpos=x
        newq = torch.cat([
            qpos[:, :6],
            qpos[:, 6:9],
            qpos[:, [13, 18, 23, 9, 14, 19, 24, 10, 15, 20, 25, 11, 16, 21, 26, 28, 12, 17, 22, 27, 29]]
        ], dim=1)
        self.hand_model.update_kinematics(q =newq)

        hand_pcd = self.hand_model.get_surface_points(q =newq).to(dtype=torch.float32)  #[B,N,3]
        #物体tensor
        
        
        
        # 加载点云数据
        objpoint = data['objpointcloud'].to(self.device)  #[B,N,3]
        
        #计算ibs
        #todo
            # 初始化模型
        self.ibs_model = self.ibs_model.to(self.device)
        
        # 前向计算
        ibs_modelpoint = self.ibs_model(objpoint, hand_pcd)  # 输出形状 [B, N, 3]
        # 读取原始ibs
        

        ibs_originpoint = data['ibspointcloud'].to(self.device)  # 形状: [B, N, 3]
          
        #计算与真实ibs差
        #todo
        chd = chamfer_dist()
        
       
        dist1, dist2, _, _ = chd(ibs_modelpoint, ibs_originpoint )
        ibs_loss = torch.mean(dist1) + torch.mean(dist2)
        return self.weight_ibs_loss*ibs_loss

    def gradient(self, x: torch.Tensor = None, x_0: torch.Tensor= None, data: Dict= None, x_mean: torch.Tensor= None, x_sample: torch.Tensor= None, std = None) -> torch.Tensor:
        """ 
        Compute gradient for optimizer constraint and update the state based on the current gradient.

        Args:
            x: The denoised signal at the current step, used to compute the gradient (default: None).
            x_0: The predicted ground truth signal (default: None), used as a reference for computing the objective.
            data: A data dictionary that provides the original input data and other relevant information :normal (default: None).
            x_mean: The mean of the predicted signal, used for correction in the mixing step (optional, default: None).
            x_sample: The sampled noisy signal at the current step 
            std: The standard deviation (noise scale) used to guide the gradient update (default: None).

        Returns:
            torch.Tensor: The updated signal after applying the computed gradient and correction step.

        Raises:
            AssertionError: If neither `x`, `x_0`, `data`, `x_sample` are provided.

        Process Overview:
            1. The function asserts that at least one of the required tensors (`x`, `x_0`, `data`, or `x_sample`) is provided.
            2. It initializes the gradients for each batch and computes an optimization objective based on `x_0`.
            3. The objective is normalized, and the gradient of the objective is computed with respect to `x`.
            4. The gradient is scaled and adjusted based on a guidance rate and noise scale, and a direction mix is computed.
            5. The function computes the final step to update the state (`x_t`), either based on `x_mean` or `x_sample`.
        
        Detailed Steps:
            - A batch-wise gradient computation is performed by splitting `x_0` into chunks of size `self._BATCH_SIZE`.
            - The signal is optionally denormalized (for translations and angles) before computing the objective.
            - The gradient is scaled using `self.scale` and optionally clipped.
            - A mixing step is applied between the computed gradient direction (`d_star`) and the difference from the mean or sample (`d_sample`).
            - The updated signal is returned, which will be used for further iterations of the optimization.
        """
        assert (x_sample !=None and x != None and x_0 != None and data != None and x.shape[0] == self._BATCH_SIZE), 'x, x_0, data, x_sample must be provided and x.shape[0] == self._BATCH_SIZE'
        _, *x_shape= x.shape
        eps = 1e-8
        with torch.enable_grad():
            # concatenate the id rot to x_in
            grad_list = []
            for i in range(x.shape[0] // self._BATCH_SIZE):
                
                i_x_0_in =x_0[i*self._BATCH_SIZE:(i+1)*self._BATCH_SIZE, :].requires_grad_(True)
                # if self.normalize_x_trans:
                #     i_x_0_in_denorm_trans = self.trans_denormalize(i_x_0_in[:, :3])
                # else:
                #     i_x_0_in_denorm_trans = i_x_0_in[:, :3]
                # if self.normalize_x:
                #     i_x_0_in_denorm_angle = self.angle_denormalize(i_x_0_in[:, 6:])
                # else:
                #     i_x_0_in_denorm_angle = i_x_0_in[:, 3:]
                # i_x_0_in_denorm = torch.cat([i_x_0_in_denorm_trans, i_x_0_in_denorm_angle], dim=-1)
                obj = self.optimize(i_x_0_in, data, t=i)
                ###TODO
                obj = torch.linalg.norm(obj)
                i_grad = torch.autograd.grad(obj, x, retain_graph=True, allow_unused=True)[0]
                grad_list.append(i_grad)
            grad = torch.cat(grad_list, dim=0)
            grad = grad * self.scale
            gradient = torch.cat([grad[:, :3], grad[:, 3:]], dim=-1) 
            grad_norm = torch.linalg.norm(gradient, dim=1).view(-1, 1)
            if std.dim() == 1:
                std = std.unsqueeze(1)
            r = torch.sqrt(torch.tensor(*x_shape)) * std[0,0]
            guidance_rate = self.guidance_scale
            d_star = -r * gradient/ (grad_norm + eps)
            if x_mean !=None:
                d_sample = x_sample -x_mean
            else:
                d_sample = std*torch.randn_like(data['x'], device=self.device)###dpm_solver++
            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction, dim= 1).view(-1, 1)
            mix_step = mix_direction / (mix_direction_norm + eps) * r
            if x_mean !=None:
                x_t =  x_mean + mix_step
            else:
                x_t =  x_sample + mix_step
            return x_t
    def angle_denormalize(self, joint_angle: torch.Tensor):
        joint_angle_denorm = joint_angle + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        joint_angle_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        joint_angle_denorm = joint_angle_denorm * (self._joint_angle_upper - self._joint_angle_lower) + self._joint_angle_lower
        return joint_angle_denorm

    def trans_denormalize(self, global_trans: torch.Tensor):
        global_trans_denorm = global_trans + (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER) / 2
        global_trans_denorm /= (self._NORMALIZE_UPPER - self._NORMALIZE_LOWER)
        global_trans_denorm = global_trans_denorm * (self._global_trans_upper - self._global_trans_lower) + self._global_trans_lower
        return global_trans_denorm