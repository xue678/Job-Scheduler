"""
ai_scheduler.py - RL调度器 
"""
import torch
import torch.nn as nn
import numpy as np
from config import *

class PPONetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=128):
        super(PPONetwork, self).__init__()
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid() 
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 【修改】给 log_std 一个初始负值 (-0.5)，让它一开始不要太随机
        # 这样 Ent 会从一个较低的值（比如 4.0）开始，而不是 7.0
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)
    
    def forward(self, state):
        features = self.shared(state)
        action_mean = self.actor(features)
        state_value = self.critic(features)
        return action_mean, state_value
    
    def get_action_inference(self, state, deterministic=True):
        with torch.no_grad():
            action_mean, _ = self.forward(state)
            if deterministic:
                return action_mean
            else:
                noise = torch.randn_like(action_mean) * 0.1
                return torch.clamp(action_mean + noise, 0, 1)

class FeatureExtractor:
    @staticmethod
    def get_observation(cluster, current_time):
        # 1. 集群特征
        # 使用 log1p 平滑处理等待队列长度
        num_waiting = len(cluster.job_list)
        if num_waiting > 0:
            wait_times = [(current_time - j['submit_time']) for j in cluster.job_list]
            avg_wait = np.mean(wait_times)
        else:
            avg_wait = 0
            
        obs = [
            min(cluster.job_gpus / max(cluster.num_gpus, 1), 1.0),
            min(cluster.job_cpus / max(cluster.num_cpus, 1), 1.0),
            np.log1p(num_waiting) / 10.0,  # 【修改】使用 Log 压缩
            np.log1p(avg_wait) / 10.0      # 【修改】使用 Log 压缩
        ]
        
        # 2. 作业特征
        for i in range(ACTION_DIM):
            if i < len(cluster.job_list):
                job = cluster.job_list[i]
                wait_time = current_time - job['submit_time']
                
                # 【修改】全部使用 log1p 进行压缩，防止数值过大
                # 假设 max duration 约为 20000s, log(20000) ≈ 9.9
                # 除以 10 可以归一化到 0-1 之间
                job_feat = [
                    min(job['num_gpu'] / 8.0, 1.0),       
                    min(job['num_cpu'] / 96.0, 1.0),      
                    np.log1p(job['duration']) / 10.0,   # Log处理
                    min(job.get('num_inst', 1) / 20.0, 1.0),
                    np.log1p(wait_time) / 10.0,         # Log处理
                    1.0 
                ]
            else:
                job_feat = [0.0] * JOB_FEAT_DIM
            
            obs.extend(job_feat)
            
        return np.array(obs, dtype=np.float32)

class AIScheduler:
    # ... 只负责加载模型 ...
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = PPONetwork(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM
        ).to(self.device)
        if model_path is not None: self.load_model(model_path)
        self.model.eval()
        
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
    def select_action(self, state, deterministic=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.model.get_action_inference(state_tensor, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def schedule_jobs(self, cluster, current_time=None):
        if len(cluster.job_list) == 0: return
        if current_time is None: current_time = getattr(cluster, 'cur_time', 0)
        state = FeatureExtractor.get_observation(cluster, current_time)
        action = self.select_action(state, deterministic=True)
        num_to_sort = min(len(cluster.job_list), ACTION_DIM)
        for i in range(num_to_sort):
            cluster.job_list[i]['rl_priority'] = float(action[i])
        partial_list = cluster.job_list[:num_to_sort]
        partial_list.sort(key=lambda j: j.get('rl_priority', 0), reverse=True)
        cluster.job_list[:num_to_sort] = partial_list
