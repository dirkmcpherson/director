import numpy as np
import torch 
from dreamerv2.models.actor import DiscreteActionModel
from dreamerv2.models.rssm import RSSM
from dreamerv2.models.dense import DenseModel
from dreamerv2.models.pixel import ObsDecoder, ObsEncoder
from dreamerv2.models.goal import GoalEncoder

from IPython import embed as ipshell
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

class Evaluator(object):
    '''
    used this only for minigrid envs
    '''
    def __init__(
        self, 
        config,
        device,
        env = None,
    ):
        self.device = device
        self.config = config
        self.action_size = config.action_size
        self.env = env

    def load_model(self, config, model_path):
        saved_dict = torch.load(model_path)
        obs_shape = config.obs_shape
        action_size = config.action_size
        deter_size = config.rssm_info['deter_size']
        if config.rssm_type == 'continuous':
            stoch_size = config.rssm_info['stoch_size']
        elif config.rssm_type == 'discrete':
            category_size = config.rssm_info['category_size']
            class_size = config.rssm_info['class_size']
            stoch_size = category_size*class_size

        embedding_size = config.embedding_size
        rssm_node_size = config.rssm_node_size
        modelstate_size = stoch_size + deter_size 

        if config.pixel:
                self.ObsEncoder = ObsEncoder(obs_shape, embedding_size, config.obs_encoder).to(self.device).eval()
                self.ObsDecoder = ObsDecoder(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()
        else:
            self.ObsEncoder = DenseModel((embedding_size,), int(np.prod(obs_shape)), config.obs_encoder).to(self.device).eval()
            self.ObsDecoder = DenseModel(obs_shape, modelstate_size, config.obs_decoder).to(self.device).eval()

        self.ActionModel = DiscreteActionModel(action_size, deter_size, stoch_size, embedding_size, config.actor, config.expl).to(self.device).eval()
        self.RSSM = RSSM(action_size, rssm_node_size, embedding_size, self.device, config.rssm_type, config.rssm_info).to(self.device).eval()

        self.RSSM.load_state_dict(saved_dict["RSSM"])
        self.ObsEncoder.load_state_dict(saved_dict["ObsEncoder"])
        self.ObsDecoder.load_state_dict(saved_dict["ObsDecoder"])
        self.ActionModel.load_state_dict(saved_dict["ActionModel"])

        
        if "GoalEncoder" in saved_dict:
            s_size = config.rssm_info['deter_size']
            z_size = config.goal_encoder['category_size'] * config.goal_encoder['class_size']
            self.GoalEncoder = GoalEncoder(output_shape=(z_size,), input_size=s_size, info=config.goal_encoder).to(self.device).eval()
            self.GoalDecoder = DenseModel(output_shape=(s_size,), input_size=z_size, info=config.goal_decoder).to(self.device).eval()
            self.GoalEncoder.load_state_dict(saved_dict["GoalEncoder"])
            self.GoalDecoder.load_state_dict(saved_dict["GoalDecoder"])

            self.n_channels = 4 # should be self.env.state_shape()[2]

            self.cmap = sns.color_palette("cubehelix", self.n_channels)
            self.cmap.insert(0,(0,0,0))
            self.cmap=colors.ListedColormap(self.cmap)
            bounds = [i for i in range(self.n_channels+2)]
            self.norm = colors.BoundaryNorm(bounds, self.n_channels+1)

            self.fig = plt.figure(num=9, figsize=(5,5))
            self.ax = self.fig.add_subplot(111)
        else:
            print(f"no goal encoder found in {model_path}")


    def render_embedding(self, decoded_img):
        self.ax.cla()
        decoded_img = (decoded_img == 1)
        numerical_state = np.amax(decoded_img*np.reshape(np.arange(self.n_channels)+1,(1,1,-1)),2)+0.5
        # numerical_state = np.amax(decoded_img, 2)+0.5
        self.ax.imshow(numerical_state, cmap=self.cmap, norm=self.norm, interpolation='none')

    def eval_saved_agent(self, env, model_path):
        self.env = env
        self.load_model(self.config, model_path)
        eval_episode = self.config.eval_episode
        eval_scores = []    
        for e in range(eval_episode):
            obs, score = env.reset(), 0
            done = False
            prev_rssmstate = self.RSSM._init_rssm_state(1)
            prev_action = torch.zeros(1, self.action_size).to(self.device)
            while not done:
                with torch.no_grad():
                    embed = self.ObsEncoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))    
                    _, posterior_rssm_state = self.RSSM.rssm_observe(embed, prev_action, not done, prev_rssmstate)
                    model_state = self.RSSM.get_model_state(posterior_rssm_state)
                    action, _ = self.ActionModel(model_state)
                    prev_rssmstate = posterior_rssm_state
                    prev_action = action



                    # Goal Encoder
                    # s = posterior_rssm_state.deter
                    # goal = self.GoalEncoder(s).sample() # dont need the gradient on the onehotcategorical so no rsample
                    # goal = goal.view(1, -1)
                    # decoded_s = self.GoalDecoder(goal)
                    # decoded_img = self.ObsDecoder(torch.cat((decoded_s, posterior_rssm_state.stoch), dim=-1)).sample()
                    decoded_img = self.ObsDecoder(model_state).sample()
                    decoded_img = decoded_img.squeeze(0).cpu().numpy()
                    decoded_img = np.transpose(decoded_img, (1,2,0))
                    # ipshell()
                    # sys.exit()
                    
                next_obs, rew, done, _ = env.step(action.squeeze(0).cpu().numpy())
                # ipshell()
                # sys.exit()
                if self.config.eval_render:
                    self.render_embedding(decoded_img)
                    env.render()

                score += rew
                obs = next_obs
            eval_scores.append(score)
        print('average evaluation score for model at ' + model_path + ' = ' +str(np.mean(eval_scores)))
        env.close()
        return np.mean(eval_scores)
