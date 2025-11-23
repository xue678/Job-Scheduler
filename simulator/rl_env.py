import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traceback 
from simulator import Simulator
from config import *
from ai_scheduler import FeatureExtractor

class ClusterSchedulingEnv(gym.Env):
    def __init__(self):
        super(ClusterSchedulingEnv, self).__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)
        
        print(f"🔧 Init Simulator: Jobs={CURRENT_CONFIG['num_jobs']}, Delta={DELTA}")
        self.simulator = Simulator(
            csv_file=CSV_FILE,
            alloc_policy=16, 
            num_jobs_limit=CURRENT_CONFIG['num_jobs'],
            hetero=HETERO,
            num_nodes=NUM_NODES,
            num_gpus=NUM_GPUS,
            num_cpus=NUM_CPUS,
            arrival_rate=-1, 
            delta=DELTA,     
            export_job_stats=False,
            export_cluster_util=False,
            verbose=0
        )
        self.simulator.init_go()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.init_go(num_jobs=CURRENT_CONFIG['num_jobs'])
        self.last_jobs_done = 0
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        cluster = self.simulator.cluster
        
        # 1. 排序 
        if len(cluster.job_list) > 0:
            n = min(len(cluster.job_list), ACTION_DIM)
            for i in range(n):
                cluster.job_list[i]['rl_score'] = float(action[i])
            head = cluster.job_list[:n]
            tail = cluster.job_list[n:]
            head.sort(key=lambda x: x.get('rl_score', 0), reverse=True)
            cluster.job_list = head + tail

        # 2. 推进 
        try:
            self.simulator.tic(delta=DELTA)
            self.current_step += 1
        except Exception as e:
            print(f"❌ Crash: {e}")
            return self._get_obs(), 0, True, False, {}

        #  3. 奖励 
        reward = 0.0
        
        # 排队长度 (除以 100，防止负数太大)
        queue_len = len(cluster.job_list)
        reward -= (queue_len / 100.0) 
        
        # GPU利用率 (0-1之间，稍微放大一点权重)
        util_gpu = cluster.job_gpus / max(cluster.num_gpus, 1)
        reward += (util_gpu * 0.5) 
        
        # 完成作业 (给小奖励，鼓励流动)
        current_jobs_done = cluster.job_history.num_jobs_done
        jobs_finished = current_jobs_done - self.last_jobs_done
        if jobs_finished > 0:
            reward += (jobs_finished * 0.2)
            self.last_jobs_done = current_jobs_done

        done = (self.simulator.exit_flag == 1)
        truncated = (self.current_step >= CURRENT_CONFIG['max_steps'])
        
        return self._get_obs(), reward, done, truncated, {}
        
    def _get_obs(self):
        return FeatureExtractor.get_observation(self.simulator.cluster, self.simulator.cur_time)
