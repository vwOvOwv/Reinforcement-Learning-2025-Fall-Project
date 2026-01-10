from multiprocessing import Process
import numpy as np
import torch
from torch.nn import functional as F
from rl_ppo.utils.model_pool import ModelPoolClient
from rl_ppo.envs.env import MahjongGBEnv
from rl_ppo.agents.ppo_agent import PPOAgent
from rl_ppo.models.network import CNNModel
import random

class Actor(Process):
    
    def __init__(self, config, id, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = f"Actor-{id}"
        self.history_sample_prob = config['history_sample_prob']
        self.have_shanten_reward = (config['shenten_reward'] is not None)
        if self.have_shanten_reward:
            self.init_shanten_reward = config['shenten_reward']['init']
            self.shanten_decay_episodes = config['shenten_reward']['decay_episode']
        self.tenpai_reward = config['tenpai_reward']
        
    def run(self):
        torch.set_num_threads(1)
    
        # connect to model pool
        model_pool = ModelPoolClient(f"model-pool-{self.config['run_id']}")
        
        # create network model
        model = CNNModel()
        opponent_model = CNNModel()
        
        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        if state_dict is not None:
            model.load_state_dict(state_dict)
            opponent_model.load_state_dict(state_dict)
        
        # collect data
        env = MahjongGBEnv(config = {'agent_clz': PPOAgent, 
                                     'reward_scaling': self.config['reward_scaling']})
        
        total_episodes = self.config['episodes_per_actor']
        for episode in range(total_episodes):
            # update model
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:    # type: ignore
                state_dict = model_pool.load_model(latest)
                if state_dict is not None:
                    model.load_state_dict(state_dict)
                    version = latest

            is_league_game = False
            if np.random.rand() < self.history_sample_prob:
                hist_version = model_pool.get_random_model()
                hist_state_dict = model_pool.load_model(hist_version)
                if hist_state_dict is not None:
                    opponent_model.load_state_dict(hist_state_dict)
                    is_league_game = True
                else:
                    opponent_model.load_state_dict(model.state_dict())
            else:
                opponent_model.load_state_dict(model.state_dict())
            
            learner_names = []
            if is_league_game:
                learner_agent = env.agent_names[random.randint(0, len(env.agent_names) - 1)]
                learner_names = [learner_agent]
            else:
                learner_names = env.agent_names

            # run one episode and collect data
            shanten_weight = self.init_shanten_reward * \
                (1 - episode / self.shanten_decay_episodes) if self.have_shanten_reward else 0.0
            obs = env.reset(shanten_weight=shanten_weight, tenpai_weight=self.tenpai_reward)
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'log_prob': [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            
            done = False
            step_count = 0
            while not done:
                # player 0 acts
                # player 1,2,3 response
                # player 1 acts
                # player 2,3,4 response
                # ...
                step_count += 1
                for agent_name in obs:
                    if agent_name in learner_names:
                        current_net = model
                    else:
                        current_net = opponent_model
                    
                    agent_state = obs[agent_name].copy()
                    agent_data = episode_data[agent_name]
                    agent_data['state']['observation'].append(agent_state['observation'])
                    agent_data['state']['action_mask'].append(agent_state['action_mask'])
                    
                    agent_state['observation'] = torch.tensor(agent_state['observation'], 
                                                              dtype = torch.float).unsqueeze(0)
                    agent_state['action_mask'] = torch.tensor(agent_state['action_mask'], 
                                                              dtype = torch.float).unsqueeze(0)
                    current_net.eval()
                    with torch.no_grad():
                        logits, value = current_net(agent_state)
                        action_dist = torch.distributions.Categorical(logits = logits)
                        action = action_dist.sample()
                        log_prob = action_dist.log_prob(action).item()
                        action = action.item()
                        value = value.item()
                    
                    agent_data['action'].append(action)
                    agent_data['value'].append(value)
                    agent_data['log_prob'].append(log_prob)
                    
                # interact with env
                agent_actions = {agent_name : episode_data[agent_name]['action'][-1] for agent_name in obs}
                next_obs, rewards, done = env.step(agent_actions)
                for agent_name in obs:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                obs = next_obs

            for agent_name, agent_data in episode_data.items():
                obs = np.stack(agent_data['state']['observation'])  # (num_steps, obs_dim)
                mask = np.stack(agent_data['state']['action_mask']) # (num_steps, action_dim)
                actions = np.array(agent_data['action'], dtype = np.int64)
                log_probs = np.array(agent_data['log_prob'], dtype = np.float32)
                rewards = np.array(agent_data['reward'], dtype = np.float32)
                values = np.array(agent_data['value'], dtype = np.float32)
                next_values = np.array(agent_data['value'][1:] + [0], dtype = np.float32)   # rotate left
                
                td_target = rewards + next_values * self.config['gamma']
                td_delta = td_target - values
                advs = []
                adv = 0
                for delta in td_delta[::-1]:
                    adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                    advs.append(adv) # GAE
                advs.reverse()
                advantages = np.array(advs, dtype = np.float32)
                returns = advantages + values
                
                # send samples to replay_buffer (per agent)
                self.replay_buffer.push({
                    'state': {
                        'observation': obs,
                        'action_mask': mask
                    },
                    'action': actions,
                    'log_prob': log_probs,
                    'raw_adv': advantages,
                    'target': returns
                })