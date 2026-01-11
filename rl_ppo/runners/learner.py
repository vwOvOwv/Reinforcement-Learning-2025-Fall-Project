from multiprocessing import Process
import time
import numpy as np
import torch
from torch.nn import functional as F
from rl_ppo.utils.model_pool import ModelPoolServer
from rl_ppo.models.network import CNNModel
import wandb

def toggle_grad(model, mode='warmup'):
    if mode == 'warmup':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.value_conv.parameters():
            param.requires_grad = True
        for param in model.value_fc.parameters():
            param.requires_grad = True
            
    elif mode == 'train':
        for param in model.parameters():
            param.requires_grad = True

class Learner(Process):
    
    def __init__(self, config, replay_buffer, log_flag):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.log_flag = log_flag
    
    def run(self):
        # initialize wandb for logging
        output_dir = self.config['output_dir'] + self.config['run_id']
        if self.log_flag:
            wandb.init(project="RL-2025-Fall-Project",
                    name=self.config['run_id'],
                    job_type="learner",
                    config=self.config,
                    dir=output_dir)

        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], 
                                     f"model-pool-{self.config['run_id']}")
        
        # initialize model params
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device=device)
        print(f"Using device: {device}")

        # load pretrained model
        model = CNNModel()
        state_dict = torch.load(self.config['pretrained_model_path'], 
                                map_location='cpu',
                                weights_only=True)
        model.load_state_dict(state_dict)
        print("Learner loading pretrained model from", self.config['pretrained_model_path'])
        
        # send to model pool
        # print("Pushing initial model to model pool...")
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        model = model.to(device)
        toggle_grad(model, mode='warmup')
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['warmup_learning_rate'])
        
        iterations = 0
        full_batch_size = self.config['batch_size']
        mini_batch_size = self.config['mini_batch_size']
        while True:
            # wait for samples
            while self.replay_buffer.size() < full_batch_size:
                time.sleep(1e-3)
            # sample batch
            batch = self.replay_buffer.sample(full_batch_size)
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device).detach()
            raw_advs = torch.tensor(batch['raw_adv']).unsqueeze(-1).to(device).detach()
            targets = torch.tensor(batch['target']).unsqueeze(-1).to(device).detach()
            old_log_probs = torch.tensor(batch['log_prob']).unsqueeze(-1).to(device).detach()

            advs = (raw_advs - raw_advs.mean()) / (raw_advs.std() + 1e-8)
            advs = advs.detach()

            # wandb log list
            policy_loss_log = []
            value_loss_log = []
            entropy_loss_log = []

            # calculate PPO loss
            model.train()    
            for epoch in range(self.config['epochs']):
                indices = torch.randperm(full_batch_size).to(device)
                for start in range(0, full_batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    mini_batch_indices = indices[start:end]
                    mini_batch_obs = obs[mini_batch_indices]
                    mini_batch_mask = mask[mini_batch_indices]
                    mini_batch_states = {
                        'observation': mini_batch_obs,
                        'action_mask': mini_batch_mask
                    }
                    mini_batch_actions = actions[mini_batch_indices]
                    mini_batch_advs = advs[mini_batch_indices]
                    mini_batch_targets = targets[mini_batch_indices]
                    mini_batch_old_log_probs = old_log_probs[mini_batch_indices]

                    logits, values = model(mini_batch_states)
                    action_dist = torch.distributions.Categorical(logits = logits)
                    log_probs = action_dist.log_prob(mini_batch_actions.squeeze(-1)).unsqueeze(-1)
                    ratio = torch.exp(log_probs - mini_batch_old_log_probs)
                    surr1 = ratio * mini_batch_advs
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 
                                        1 + self.config['clip']) * mini_batch_advs
                    raw_policy_loss = torch.min(surr1, surr2)
                    clip_c = self.config['dppo_clip']
                    policy_loss = -torch.mean(torch.where(
                        mini_batch_advs < 0, 
                        torch.max(raw_policy_loss, clip_c * mini_batch_advs), 
                        raw_policy_loss
                    ))
                    value_loss = torch.mean(F.mse_loss(values, mini_batch_targets))
                    entropy_loss = -torch.mean(action_dist.entropy())
                    if iterations < self.config['warmup_iters']:
                        loss = value_loss
                    else:
                        loss = policy_loss + self.config['value_coeff'] * value_loss + \
                           self.config['entropy_coeff'] * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
                    optimizer.step()

                    policy_loss_log.append(policy_loss.item())
                    value_loss_log.append(value_loss.item())
                    entropy_loss_log.append(-entropy_loss.item())

                    # send to wandb
                    if self.log_flag and iterations % self.config['log_interval'] == 0:
                        wandb.log({
                            "Loss/Policy": np.mean(policy_loss_log),
                            "Loss/Value": np.mean(value_loss_log),
                            "Loss/Entropy": np.mean(entropy_loss_log),
                            "Train/LearningRate": optimizer.param_groups[0]['lr'],
                        }, step=iterations)
                    
                    # save checkpoints
                    if self.log_flag and iterations % self.config['ckpt_save_interval'] == 0:
                        path = output_dir + f"/iteration-{iterations}.pt"
                        torch.save(model.state_dict(), path)

                    iterations += 1
                    if iterations == self.config['warmup_iters']:
                        toggle_grad(model, mode='train')
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = self.config['learning_rate']
                        print("Learner leaving warmup")

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)