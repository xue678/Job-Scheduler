
import gym
from gym import spaces
import numpy as np
from pathlib import Path

class ClusterSchedulingEnvRealistic(gym.Env):
    
    def __init__(self, csv_file, num_gpus=6500, num_jobs=1000, 
                 max_steps=10000, hetero=True, use_real_nodes=True):
        super(ClusterSchedulingEnvRealistic, self).__init__()
        
        self.csv_file = csv_file
        self.num_gpus = num_gpus
        self.num_cpus = int(23.22 * num_gpus)
        self.num_jobs = num_jobs
        self.max_steps = max_steps
        self.hetero = hetero
        self.use_real_nodes = use_real_nodes
        self.current_step = 0
        
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(15,), dtype=np.float32
        )
        
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32
        )
        
        self._last_obs = None
        self._obs_cache_valid = False
        
        self.simulator = None
        self.reset()
    
    def reset(self):
        
        self.current_step = 0
        self._obs_cache_valid = False
        
        from simulator import Simulator
        
        
        self.simulator = Simulator(
            csv_file=self.csv_file,
            alloc_policy=99,
            preempt_policy=2,
            num_gpus=self.num_gpus,
            num_cpus=self.num_cpus,
            num_nodes=1,  
            num_jobs_limit=self.num_jobs,
            max_time=int(1e9),
            export_job_stats=False,
            export_cluster_util=False,
            verbose=0,
            sort_node_policy=0,
            hetero=self.hetero  
        )
        
        self.simulator.init_go(num_jobs=self.num_jobs)
        
        return self._get_observation()
    
    def _get_observation(self):
        
        if self._obs_cache_valid and self._last_obs is not None:
            return self._last_obs
        
        cluster = self.simulator.cluster
        
        
        gpu_util = cluster.job_gpus / max(cluster.num_gpus, 1)
        cpu_util = cluster.job_cpus / max(cluster.num_cpus, 1)
        
        num_waiting = len(cluster.job_list)
        avg_wait_time = 0
        if num_waiting > 0:
            total_wait = sum(j.get('on_time', 0) for j in cluster.job_list)
            avg_wait_time = total_wait / num_waiting
        
        
        if num_waiting > 0:
            job = cluster.job_list[0]
            job_features = [
                min(job['num_gpu'] / max(cluster.num_gpus, 1), 1.0),
                min(job['num_cpu'] / max(cluster.num_cpus, 1), 1.0),
                min(job['duration'] / 10000.0, 1.0),
                min(job.get('num_inst', 1) / 20.0, 1.0),
                1.0  
            ]
        else:
            job_features = [0, 0, 0, 0, 0]
        
        
        node_list = cluster.node_list
        
        if len(node_list) > 0:
            
            node_gpu_utils = [
                (n.num_gpus - n.idl_gpus) / max(n.num_gpus, 1) 
                for n in node_list if n.num_gpus > 0
            ]
            
            if node_gpu_utils:
                avg_node_util = np.mean(node_gpu_utils)
                std_node_util = np.std(node_gpu_utils)
                max_node_util = np.max(node_gpu_utils)
            else:
                avg_node_util = std_node_util = max_node_util = 0
            
            
            idle_nodes = sum(1 for n in node_list if n.idl_gpus == n.num_gpus)
            idle_node_ratio = idle_nodes / max(len(node_list), 1)
            
            
            fragmented_nodes = sum(
                1 for n in node_list 
                if 0 < n.idl_gpus < n.num_gpus * 0.5
            )
            fragmentation = fragmented_nodes / max(len(node_list), 1)
            
            
            running_jobs = len(cluster.job_runn_list)
            normalized_running = min(running_jobs / 1000.0, 1.0)
            
            load_features = [
                min(avg_node_util, 1.0),
                min(std_node_util, 1.0),
                min(max_node_util, 1.0),
                min(idle_node_ratio, 1.0),
                min(fragmentation, 1.0),
                min(normalized_running, 1.0)
            ]
        else:
            load_features = [0, 0, 0, 0, 0, 0]
        
        
        obs = np.array([
            min(gpu_util, 1.0),
            min(cpu_util, 1.0),
            min(num_waiting / 100.0, 1.0),
            min(avg_wait_time / 1000.0, 1.0),
        ] + job_features + load_features, dtype=np.float32)
        
        self._last_obs = obs
        self._obs_cache_valid = True
        
        return obs
    
    def step(self, action):
        
        cluster = self.simulator.cluster
        self._obs_cache_valid = False
        
        
        if len(cluster.job_list) > 0:
            num_to_sort = min(len(cluster.job_list), 10)  # 从5增加到10
            priorities = action[:num_to_sort]
            
            for i in range(num_to_sort):
                cluster.job_list[i]['rl_priority'] = float(priorities[i])
            
            cluster.job_list.sort(key=lambda j: j.get('rl_priority', 0), reverse=True)
        
        
        prev_jct = cluster.job_history.jct_summary
        prev_jobs_done = cluster.job_history.num_jobs_done
        prev_waiting = len(cluster.job_list)
        
        
        try:
            self.simulator.tic(delta=10)
            self.current_step += 1
        except Exception as e:
            return self._get_observation(), 0, True, {}
        
        
        reward = self._calculate_reward(prev_jct, prev_jobs_done, prev_waiting)
        
        done = (self.simulator.exit_flag == 1) or (self.current_step >= self.max_steps)
        
        return self._get_observation(), reward, done, {}
    
    def _calculate_reward(self, prev_jct, prev_jobs_done, prev_waiting):
        
        cluster = self.simulator.cluster
        curr_jct = cluster.job_history.jct_summary
        curr_jobs_done = cluster.job_history.num_jobs_done
        curr_waiting = len(cluster.job_list)
        
        reward = 0.0
        
        
        jobs_finished = curr_jobs_done - prev_jobs_done
        reward += jobs_finished * 10.0
        
        
        if jobs_finished > 0:
            avg_jct_increase = (curr_jct - prev_jct) / max(jobs_finished, 1)
            reward -= avg_jct_increase / 1000.0
        
        
        gpu_util = cluster.job_gpus / max(cluster.num_gpus, 1)
        cpu_util = cluster.job_cpus / max(cluster.num_cpus, 1)
        reward += (gpu_util + cpu_util) * 0.5
        
        
        waiting_change = prev_waiting - curr_waiting
        reward += waiting_change * 0.1
        
        
        node_list = cluster.node_list
        if len(node_list) > 0:
            node_utils = [
                (n.num_gpus - n.idl_gpus) / max(n.num_gpus, 1)
                for n in node_list if n.num_gpus > 0
            ]
            if node_utils:
                
                util_std = np.std(node_utils)
                reward -= util_std * 2.0  
        
        return reward
    
    def render(self, mode='human'):
        
        if self.current_step % 200 == 0:
            cluster = self.simulator.cluster
            node_count = len(cluster.node_list)
            print(f"[Step {self.current_step}] "
                  f"Nodes: {node_count}, "
                  f"Done: {cluster.job_history.num_jobs_done}/{self.num_jobs}, "
                  f"Wait: {len(cluster.job_list)}, "
                  f"GPU: {cluster.job_gpus/cluster.num_gpus:.1%}")