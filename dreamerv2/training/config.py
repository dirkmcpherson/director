import numpy as np
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Tuple, Dict

# Following HPs are not a result of detailed tuning.   

@dataclass
class MinAtarConfig():
    '''default HPs that are known to work for MinAtar envs '''
    #env desc
    env : str                                           
    obs_shape: Tuple                                            
    action_size: int
    pixel: bool = True
    action_repeat: int = 1
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(5e6)
    train_every: int = 50                                  #reduce this to potentially improve sample requirements
    collect_intervals: int = 5 
    batch_size: int = 40
    seq_len: int = 40
    eval_episode: int = 4
    eval_render: bool = True
    save_every: int = int(1e5)
    seed_steps: int = 4000
    model_dir: int = 'results'
    gif_dir: int = 'results'
    buffer_dir: int = 'buffer'
    
    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 200
    rssm_node_size: int = 200
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':1024, 'stoch_size':128, 'class_size':20, 'category_size':20, 'min_std':0.1})
    
    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 10
    lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4, 'goal':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':0.1, 'reward':1.0, 'discount':5.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 100
    slow_target_fraction: float = 1.00

    #actor critic
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': 'normal', 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':7000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':3, 'depth':16})
    # obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':3, 'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'kernel':3, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})

    #HRL
    goal_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':200, 'dist':"onehotst", 'activation':nn.ELU, 'depth':16, 'class_size':16, 'category_size':16}) # dist is onehot in original code
    goal_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':200, 'dist':None, 'activation':nn.ELU, 'depth':16, 'class_size':16, 'category_size':16}) # Dist is MSE in original code
    worker_grad: str = 'reinforce'
    manager_grad: str = 'reinforce'

    ## HRL from tensorflow director
    # skill_shape: Tuple = (8, 8)
    # env_skill_duration: 8
    # train_skill_duration: 8
    # skill_shape: [8, 8]
    # manager_rews: {extr: 1.0, expl: 0.1, goal: 0.0}
    # worker_rews: {extr: 0.0, expl: 0.0, goal: 1.0}
    # worker_inputs: [deter, stoch, goal]
    # worker_report_horizon: 64
    # skill_proposal: manager
    # goal_proposal: replay
    # goal_reward: cosine_max
    # goal_encoder: {layers: 4, units: 512, act: elu, norm: layer, dist: onehot, outscale: 0.1, unimix: 0.0, inputs: [goal]}
    # goal_decoder: {layers: 4, units: 512, act: elu, norm: layer, dist: mse, outscale: 0.1, inputs: [skill]}
    # encdec_kl: {impl: mult, scale: 0.0, target: 10.0, min: 1e-5, max: 1.0}
    # encdec_opt: {opt: adam, lr: 1e-4, eps: 1e-6, clip: 100.0, wd: 1e-2, wd_pattern: 'kernel'}
    # worker_goals: [manager]
    # jointly: new
    # vae_imag: False
    # vae_replay: True
    # vae_span: False
    # explorer: False
    # explorer_repeat: False
    # expl_rew: adver  # disag
    # manager_dist: onehot
    # manager_grad: reinforce
    # manager_actent: 0.5
    # adver_impl: squared  # elbo_scaled
    # manager_delta: False
    # goal_kl: True


@dataclass
class MiniGridConfig():
    '''default HPs that are known to work for MiniGrid envs'''
    #env desc
    env : str                                           
    obs_shape: Tuple                                            
    action_size: int
    pixel: bool = True
    action_repeat: int = 1
    time_limit: int = 1000
    
    #buffer desc
    capacity: int = int(1e6)
    obs_dtype: np.dtype = np.uint8
    action_dtype: np.dtype = np.float32

    #training desc
    train_steps: int = int(1e6)
    train_every: int = 5
    collect_intervals: int = 5
    batch_size: int = 40
    seq_len: int = 8
    eval_episode: int = 5
    eval_render: bool = False
    save_every: int = int(5e4)
    seed_episodes: int = 5
    model_dir: int = 'results'
    gif_dir: int = 'results'
    buffer_dir: int = 'buffer'

    #latent space desc
    rssm_type: str = 'discrete'
    embedding_size: int = 100
    rssm_node_size: int = 100
    rssm_info: Dict = field(default_factory=lambda:{'deter_size':100, 'stoch_size':256, 'class_size':16, 'category_size':16, 'min_std':0.1})

    #objective desc
    grad_clip: float = 100.0
    discount_: float = 0.99
    lambda_: float = 0.95
    horizon: int = 8
    lr: Dict = field(default_factory=lambda:{'model':2e-4, 'actor':4e-5, 'critic':1e-4})
    loss_scale: Dict = field(default_factory=lambda:{'kl':1, 'reward':1.0, 'discount':10.0})
    kl: Dict = field(default_factory=lambda:{'use_kl_balance':True, 'kl_balance_scale':0.8, 'use_free_nats':False, 'free_nats':0.0})
    use_slow_target: float = True
    slow_target_update: int = 50
    slow_target_fraction: float = 1.0

    #actor critic
    actor: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'one_hot', 'min_std':1e-4, 'init_std':5, 'mean_scale':5, 'activation':nn.ELU})
    critic: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': 'normal', 'activation':nn.ELU})
    expl: Dict = field(default_factory=lambda:{'train_noise':0.4, 'eval_noise':0.0, 'expl_min':0.05, 'expl_decay':10000.0, 'expl_type':'epsilon_greedy'})
    actor_grad: str ='reinforce'
    actor_grad_mix: int = 0.0
    actor_entropy_scale: float = 1e-3

    #learnt world-models desc
    obs_encoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist': None, 'activation':nn.ELU, 'kernel':2, 'depth':16})
    obs_decoder: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU, 'kernel':2, 'depth':16})
    reward: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'normal', 'activation':nn.ELU})
    discount: Dict = field(default_factory=lambda:{'layers':3, 'node_size':100, 'dist':'binary', 'activation':nn.ELU, 'use':True})

    
