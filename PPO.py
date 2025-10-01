# 1.1
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torch.distributions import Normal
import os
import hydra 
from omegaconf import DictConfig
from pathlib import Path

os.environ['MUJOCO_GL'] = 'osmesa'
# Seed and environment setup
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#1.2
# Custom weight initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, n_features, n_neuron):
        super(Actor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_neuron, bias=True),
            nn.ReLU()
        )
        """        self.linear1 = nn.Sequential(
            nn.Linear(in_features=n_neuron, out_features=n_neuron, bias=True),
            nn.ReLU()
        )"""

        self.mu = nn.Sequential(
            nn.Linear(in_features=n_neuron, out_features=2, bias=True),
            nn.Tanh()
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features=n_neuron, out_features=2, bias=True),
            nn.Softplus()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.linear(x)
       # y = self.linear1(z)
        mu = 2 * self.mu(y)
        sigma = self.sigma(y) + 1e-5  # Ensure sigma is never zero
        dist = Normal(mu, sigma)
        return dist

class Critic(nn.Module):
    def __init__(self, n_features, n_neuron):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_neuron, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=n_neuron, out_features=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class PPO(object):
    def __init__(self, cfg,num_in):
        self.actor_lr = cfg.a_lr
        self.critic_lr = cfg.c_lr
        self.actor_old = Actor(num_in, cfg.n_neuron)
        self.actor = Actor(num_in, cfg.n_neuron)
        self.critic = Critic(num_in, cfg.n_neuron)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(params=self.critic.parameters(), lr=self.critic_lr)
        self.max_grad_norm = cfg.max_grad_norm
        self.kl=cfg.kl
        self.epsilon = cfg.epsilon
        self.temperature_max = cfg.temperature_max
        self.temperature_min = cfg.temperature_min
        self.decay_rate = cfg.decay_rate
        self.temperature = self.temperature_max

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.actor_losses = []
        self.critic_losses = []

        self.max_ep=cfg.ep_max

    def save_actor(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def save_critic(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.critic.state_dict(), path)

    def update(self, s, a, r, log_old, next_s,cfg):
        if len(s) == 0:
            return  # Skip if the buffer is empty

        self.actor_old.load_state_dict(self.actor.state_dict())
        state = torch.FloatTensor(s)
        action = torch.FloatTensor(a)
        discounted_r = torch.FloatTensor(r)
        next_state = torch.FloatTensor(next_s)

        #old_action_log_prob = torch.FloatTensor(log_old)
        dist_old = self.actor_old(state)

        old_action_log_prob = dist_old.log_prob(action).sum(-1, keepdim=True).detach()
    
        target_v = discounted_r.unsqueeze(-1)
        advantage = (target_v - self.critic(state)).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) # Normalize advantage
        if cfg.method == 'kl_pen':
            for _ in range(cfg.update_steps):
                dist = self.actor(state)
                new_action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                new_action_prob = torch.exp(new_action_log_prob)
                old_action_prob = torch.exp(old_action_log_prob)

                self.kl = torch.distributions.kl_divergence( dist_old, dist).mean()
                ratio = new_action_prob / old_action_prob
                actor_loss = -torch.mean(ratio * advantage - cfg.lem * self.kl)
                self.actor_losses.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                #if self.kl> 4 * METHOD['kl_target']:
                 #   break
                print(f"kl after update steps: {self.kl}")
                if self.kl < cfg.kl_target / 1.5:
                    cfg.lam /= 2
                elif self.kl > cfg.kl_target * 1.5:
                    cfg.lam *= 2
                cfg.lam = np.clip(cfg.lam, 1e-4, 100)
                critic_loss = nn.MSELoss()(self.critic(state), target_v)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.critic_losses.append(critic_loss.item())
        else:
            for _ in range(cfg.update_steps):
                dist = self.actor(state)
                new_action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                ratio = torch.exp(new_action_log_prob - old_action_log_prob)
                L1 = ratio * advantage
                L2 = torch.clamp(ratio, 1 - cfg.mepsilon, 1 + cfg.mepsilon) * advantage
                actor_loss = -torch.min(L1, L2).mean()
                self.actor_losses.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

        
                critic_loss = nn.MSELoss()(self.critic(state), target_v)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.critic_losses.append(critic_loss.item())

    def select_action(self, s):
        s = torch.FloatTensor(s).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(s)
        action = dist.sample().squeeze(0)
        action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, action_log_prob
    
    
    def get_v(self, s):
        s = torch.FloatTensor(s).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(s)
        return value


    # Plotting function
    def plot_training(self,reward_history,episode):
        sma = np.convolve(reward_history, np.ones(100) / 100, mode='valid')
        plt.figure()
        plt.title("Rewards")
        plt.plot(reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 100', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        if episode == self.max_ep:
            plt.savefig('reward_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        #plt.show()
        #plt.clf()
        plt.close()

        plt.figure()
        plt.title("Actor Loss")
        plt.plot(self.actor_losses, label='Loss', color='r', alpha=1)
        plt.xlabel("Update Steps")
        plt.ylabel("Actor Loss")
        if episode == self.max_ep:
            plt.savefig('actor_loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        #plt.show()
        #plt.clf()
        plt.close()

        plt.figure()
        plt.title("Critic Loss")
        plt.plot(self.critic_losses, label='Critic Loss', color='b', alpha=1)
        plt.xlabel("Update Steps")
        plt.ylabel("Critic Loss")
        if episode == self.max_ep:
            plt.savefig('critic_loss_plot.png', format='png', dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        #plt.show()
        #plt.clf()
        plt.close()



@hydra.main(version_base="1.1", config_path="./conf", config_name="config")
def main(cfg: DictConfig):

    env = gym.make(cfg.env_name, render_mode="human" if not cfg.train else None).unwrapped
    num_in = env.observation_space.shape[0]
    ppo = PPO(cfg,num_in)
    reward_history=[]
    total_steps = 1
    if cfg.train:
        for ep in range(1,cfg.ep_max+1):
            s, info = env.reset(seed=seed)
            buffer_s, buffer_a, buffer_r = [], [], []
            buffer_log_old = []
            buffer_next_state = []
            ep_r = 0
            ep_step = 0
            for t in range(cfg.ep_len):
                a, a_log_prob = ppo.select_action(s)
                s_, r, terminated, truncated, info = env.step(a.numpy())
                done = terminated or truncated
                buffer_s.append(s)
                buffer_a.append(a.numpy())
                buffer_r.append(r)
                buffer_log_old.append(a_log_prob.detach().numpy())
                buffer_next_state.append(s_)
                s = s_
                ep_r += r
                ep_step += 1
                if (t + 1) % cfg.batch == 0 or t == cfg.ep_len - 1 or done:
                    v_s_ = ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + cfg.gamma * v_s_
                        discounted_r.append(v_s_.item())
                    discounted_r.reverse()

                    bs = np.array(buffer_s)
                    ba = np.array(buffer_a)
                    br = np.array(discounted_r)
                    blog = np.array(buffer_log_old)
                    br_next_state = np.array(buffer_next_state)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    buffer_log_old = []
                    buffer_next_state = []

                    ppo.update(bs, ba, br, blog, br_next_state,cfg)
                total_steps += 1
                if done:
                    break
            reward_history.append(ep_r)
                #-- based on interval
            if ep % cfg.Interval_Save == 0:
                #ppo.save(save_path + '_' + f'{ep}' + '.pth')
                ppo.save_actor(cfg.actor_save_path+ '_' + f'{ep}' + '.pth')
                ppo.save_critic(cfg.critic_save_path+ '_' + f'{ep}' + '.pth')
                ppo.plot_training(reward_history,ep)
                print('\n~~~~~~Interval Save: Model saved.\n')
            #result = (f"Episode: {ep}/1000 | "f"Episode Reward:{ep_r:.2f} | "f"lam:{METHOD['lam']} ") 
            result = (f"Episode: {ep}/1000 | "f"Episode Reward:{ep_r:.2f}")       
            print(result)
        env.close()
    if cfg.test:
        state_dict_actor = torch.load('./actor_weights-and-plot/final_weights_1000.pth')
        state_dict_critic = torch.load('./critic_weights-and-plot/final_weights_1000.pth')
        ppo.actor.load_state_dict(state_dict_actor)
        ppo.critic.load_state_dict(state_dict_critic)
        print("Loaded weights")
    
        for ep in range(5):
            s, info = env.reset(seed=seed)
            ep_r = 0
            for t in range(1,cfg.ep_len+1):
                a, _ = ppo.select_action(s)
                s_, r, terminated, truncated, info = env.step(a.numpy())
                s = s_
                done = terminated or truncated
                if done:
                    s, info = env.reset()
                ep_r+=r
            print(f"Reward of test Episode{ep}:",ep_r)

        env.close()

if __name__=="__main__":
    main()
