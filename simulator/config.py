from pathlib import Path
import os

#  1. 基础配置 
#BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = Path('/kaggle/working/simulator_rl_fixed')
DATA_DIR = BASE_DIR / 'traces/pai/'
CSV_FILE = DATA_DIR / 'pai_job_duration_estimate_100K.csv' 

#CSV_FILE = /kaggle/input/simulatorrl/traces/pai/pai_job_duration_estimate_100K.csv'

# 集群 
HETERO = True     
NUM_NODES = 380   
NUM_GPUS = 2500   
NUM_CPUS = int(23.22 * NUM_GPUS) 

# 网络维度
ACTION_DIM = 5        
JOB_FEAT_DIM = 6      
CLUSTER_FEAT_DIM = 4  
STATE_DIM = CLUSTER_FEAT_DIM + (ACTION_DIM * JOB_FEAT_DIM) # 34
HIDDEN_DIM = 128      

# 训练参数
LEARNING_RATE = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
BATCH_SIZE = 2048     

#  2. 学习配置 (关键!) 

# (快速跑通，学会基本规则)
STAGE_1_CONFIG = {
    'num_jobs': 1000,      # 作业少，容易处理
    'num_episodes': 200,   # 跑200轮
    'max_steps': 3000,     
    'save_freq': 20,
    'model_name': 'stage1_model.pth'
}

# (上强度，解决拥堵)
STAGE_2_CONFIG = {
    'num_jobs': 5000,      # 作业多，压力大
    'num_episodes': 1000,  # 多跑几轮
    'max_steps': 10000,    
    'save_freq': 50,
    'model_name': 'final_model.pth'
}


# 先用 STAGE_1 跑，跑完后改这里为 STAGE_2 再跑
CURRENT_CONFIG = STAGE_1_CONFIG 
# CURRENT_CONFIG = STAGE_2_CONFIG

# 3. 路径与优化 
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / 'best_model.pth'
FINAL_MODEL_PATH = MODEL_DIR / CURRENT_CONFIG['model_name']

VERBOSE = False     
DELTA = 20          # 稍微调大Delta到20
USE_CACHE = True
