"""
Alpha Robot 训练器
支持PPO, SAC, TD3等算法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
from datetime import datetime


class BaseNetwork(nn.Module):
    """基础神经网络"""
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        return self.network(x)

class ActorNetwork(BaseNetwork):
    """Actor网络（策略网络）"""
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__(state_dim, action_dim * 2, hidden_dims)
        self.action_dim = action_dim
        
    def forward(self, state):
        # print(f"当前state shape:", state.shape)  # 调试输出 结果是13
        output = self.network(state)
        mean, log_std = output.split(self.action_dim, dim=-1)
        
        # 限制log_std的范围
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        return mean, std
        
    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Tanh squashing
        action = torch.tanh(action)
        
        return action, log_prob


class CriticNetwork(BaseNetwork):
    """Critic网络（价值网络）"""
    def __init__(self, state_dim, action_dim=0, hidden_dims=[512, 512]):
        input_dim = state_dim + action_dim
        super().__init__(input_dim, 1, hidden_dims)
        
    def forward(self, state, action=None):
        if action is not None:
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        return self.network(x)


class BaseTrainer(ABC):
    """基础训练器"""
    def __init__(self, env, config=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # 默认配置
        self.config = {
            'lr': 3e-4,
            'batch_size': 256,
            'gamma': 0.99,
            'tau': 0.005,
            'buffer_size': 1000000,
            'exploration_noise': 0.1,
            'policy_noise': 0.2,
            'noise_clip': 0.5,
            'policy_freq': 2,
            'save_freq': 10,
            'eval_freq': 10,
            'eval_episodes': 5,
        }
        
        if config:
            self.config.update(config)
            
        # 创建日志目录
        self.log_dir = f"logs/{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 性能追踪
        self.training_rewards = []
        self.eval_rewards = []
        self.success_rates = []
        
    @abstractmethod
    def train(self, num_episodes):
        """训练方法（子类实现）"""
        pass
        
    def evaluate(self, num_episodes=5):
        """评估当前策略"""
        rewards = []
        successes = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    action = self.select_action(state, evaluate=True)
                    
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
                state = next_state
                
            rewards.append(episode_reward)
            successes.append(info.get('success', False))
            
        avg_reward = np.mean(rewards)
        success_rate = np.mean(successes)
        
        return avg_reward, success_rate
        
    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = os.path.join(self.log_dir, "model.pth")
        torch.save(self.get_state_dict(), path)
        
    def load_model(self, path):
        """加载模型"""
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        
    def plot_results(self):
        """绘制训练结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 奖励曲线
        ax1.plot(self.training_rewards, label='Training', alpha=0.7)
        if self.eval_rewards:
            eval_x = np.arange(0, len(self.training_rewards), 
                              len(self.training_rewards) // len(self.eval_rewards))
            ax1.plot(eval_x, self.eval_rewards, label='Evaluation', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)
        
        # 成功率曲线
        if self.success_rates:
            ax2.plot(self.success_rates)
            ax2.set_xlabel('Evaluation')
            ax2.set_ylabel('Success Rate')
            ax2.set_title('Success Rate Progress')
            ax2.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()
        
    @abstractmethod
    def select_action(self, state, evaluate=False):
        """选择动作（子类实现）"""
        pass
        
    @abstractmethod
    def get_state_dict(self):
        """获取模型状态（子类实现）"""
        pass
        
    @abstractmethod  
    def load_state_dict(self, state_dict):
        """加载模型状态（子类实现）"""
        pass


class PPOTrainer(BaseTrainer):
    """PPO训练器"""
    def __init__(self, env, config=None):
        super().__init__(env, config)
        
        # PPO特定配置
        self.config.update({
            'ppo_epochs': 10,
            'clip_epsilon': 0.2,
            'value_coeff': 0.5,
            'entropy_coeff': 0.01,
            'gae_lambda': 0.95,
            'max_grad_norm': 0.5,
            'rollout_length': 2048,
        })
        
        if config:
            self.config.update(config)
            
        # 获取维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # 创建网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['lr'])
        
        # 经验缓冲
        self.rollout_buffer = RolloutBuffer()
        
    def train(self, num_episodes):
        """PPO训练循环"""
        total_steps = 0
        
        for episode in range(num_episodes):
            # 收集经验
            self.rollout_buffer.clear()
            rollout_reward = self.collect_rollout()
            self.training_rewards.append(rollout_reward)
            
            # 更新策略
            self.update_policy()
            
            # 评估
            if episode % self.config['eval_freq'] == 0:
                avg_reward, success_rate = self.evaluate()
                self.eval_rewards.append(avg_reward)
                self.success_rates.append(success_rate)
                
                print(f"Episode {episode}, "
                      f"Train Reward: {rollout_reward:.2f}, "
                      f"Eval Reward: {avg_reward:.2f}, "
                      f"Success Rate: {success_rate:.2%}")
                
            # 保存模型
            if episode % self.config['save_freq'] == 0:
                self.save_model()
                self.plot_results()
                
    def collect_rollout(self):
        """收集一个rollout的经验"""
        print("开始采样 rollout ...")
        state, _ = self.env.reset()
        total_reward = 0
        
        for _ in range(self.config['rollout_length']):
            # 选择动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                # print(f"当前state维数在collect_rollout方法中:", state.shape)  # 调试输出
                # print(f"当前state_tensor shape:", state_tensor.shape)  # 调试输出
                action, log_prob = self.actor.sample(state_tensor)
                value = self.critic(state_tensor)
                
            action_np = action.cpu().numpy()[0]
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated
            total_reward += reward
            
            # 存储经验
            self.rollout_buffer.add(
                state, action_np, reward, done,
                log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
            )
            
            state = next_state
            
            if done:
                state, _ = self.env.reset()
                
        return total_reward
        
    def update_policy(self):
        """更新策略网络"""
        # 准备数据
        states, actions, rewards, dones, old_log_probs, old_values = self.rollout_buffer.get()
        
        # 计算优势和回报
        advantages, returns = self.compute_gae(rewards, old_values, dones)
        
        # 转换为tensor
        # 更高效地转换为 tensor
        states         = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions        = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        old_log_probs  = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(self.device)
        advantages     = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
        returns        = torch.tensor(np.array(returns), dtype=torch.float32).to(self.device)
        returns        = returns.squeeze(-1)  # 注意保持正确 shape


        # PPO更新
        for _ in range(self.config['ppo_epochs']):
            # 计算新的log概率和价值
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            
            values = self.critic(states).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 
                               1 - self.config['clip_epsilon'], 
                               1 + self.config['clip_epsilon']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values, returns)
            
            # 总损失
            actor_loss = policy_loss - self.config['entropy_coeff'] * entropy
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['max_grad_norm'])
            self.actor_optimizer.step()
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm'])
            self.critic_optimizer.step()
            
    def compute_gae(self, rewards, values, dones):
        """计算广义优势估计(GAE)"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
                
            delta = rewards[i] + self.config['gamma'] * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.config['gamma'] * self.config['gae_lambda'] * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
        
    def select_action(self, state, evaluate=False):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if evaluate:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
                
        return action.cpu().numpy()[0]
        
    def get_state_dict(self):
        """获取模型状态"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        
    def load_state_dict(self, state_dict):
        """加载模型状态"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])


class SACTrainer(BaseTrainer):
    """SAC (Soft Actor-Critic) 训练器"""
    def __init__(self, env, config=None):
        super().__init__(env, config)
        
        # SAC特定配置
        self.config.update({
            'alpha': 0.2,  # 温度参数
            'automatic_entropy_tuning': True,
            'target_update_interval': 1,
        })
        
        if config:
            self.config.update(config)
            
        # 获取维度
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        # 创建网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
        
        self.critic1 = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        
        self.critic1_target = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target = CriticNetwork(self.state_dim, self.action_dim).to(self.device)
        
        # 复制目标网络参数
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['lr'])
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config['lr'])
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config['lr'])
        
        # 自动温度调整
        if self.config['automatic_entropy_tuning']:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config['lr'])
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = self.config['alpha']
            
        # 经验回放
        self.replay_buffer = ReplayBuffer(self.config['buffer_size'])
        
    def train(self, num_episodes):
        """SAC训练循环"""
        total_steps = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # 存储经验
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # 更新网络
                if len(self.replay_buffer) > self.config['batch_size']:
                    self.update_networks()
                    
                state = next_state
                total_steps += 1
                
            self.training_rewards.append(episode_reward)
            
            # 评估
            if episode % self.config['eval_freq'] == 0:
                avg_reward, success_rate = self.evaluate()
                self.eval_rewards.append(avg_reward)
                self.success_rates.append(success_rate)
                
                print(f"Episode {episode}, "
                      f"Train Reward: {episode_reward:.2f}, "
                      f"Eval Reward: {avg_reward:.2f}, "
                      f"Success Rate: {success_rate:.2%}, "
                      f"Alpha: {self.alpha:.4f}")
                      
            # 保存模型
            if episode % self.config['save_freq'] == 0:
                self.save_model()
                self.plot_results()
                
    def update_networks(self):
        """更新所有网络"""
        # 采样批次
        batch = self.replay_buffer.sample(self.config['batch_size'])
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            # 采样下一个动作
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 计算目标Q值
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            
            q_target = rewards + (1 - dones) * self.config['gamma'] * q_next
            
        # 更新Critic
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(q1, q_target)
        critic2_loss = F.mse_loss(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新温度参数
        if self.config['automatic_entropy_tuning']:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            
        # 软更新目标网络
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)
        
    def soft_update(self, source, target):
        """软更新目标网络"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config['tau'] * param.data + (1 - self.config['tau']) * target_param.data
            )
            
    def select_action(self, state, evaluate=False):
        """选择动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if evaluate:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
                
        return action.cpu().numpy()[0]
        
    def get_state_dict(self):
        """获取模型状态"""
        state_dict = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
        }
        
        if self.config['automatic_entropy_tuning']:
            state_dict['log_alpha'] = self.log_alpha
            state_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
            
        return state_dict
        
    def load_state_dict(self, state_dict):
        """加载模型状态"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.critic1_target.load_state_dict(state_dict['critic1_target'])
        self.critic2_target.load_state_dict(state_dict['critic2_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(state_dict['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(state_dict['critic2_optimizer'])
        
        if self.config['automatic_entropy_tuning'] and 'log_alpha' in state_dict:
            self.log_alpha = state_dict['log_alpha']
            self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])


class RolloutBuffer:
    """PPO的Rollout缓冲区"""
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.dones),
            np.array(self.log_probs),
            np.array(self.values)
        )


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = np.array([self.buffer[i]['state'] for i in batch])
        actions = np.array([self.buffer[i]['action'] for i in batch])
        rewards = np.array([self.buffer[i]['reward'] for i in batch])
        next_states = np.array([self.buffer[i]['next_state'] for i in batch])
        dones = np.array([self.buffer[i]['done'] for i in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        
    def __len__(self):
        return len(self.buffer)


# 算法比较工具
class AlgorithmComparison:
    """比较不同RL算法的性能"""
    def __init__(self, env_creator):
        self.env_creator = env_creator
        self.results = {}
        
    def add_algorithm(self, name, trainer_class, config=None):
        """添加要比较的算法"""
        self.results[name] = {
            'trainer_class': trainer_class,
            'config': config,
            'rewards': [],
            'success_rates': [],
            'training_time': 0
        }
        
    def run_comparison(self, num_episodes=10000, num_seeds=3):
        """运行比较实验"""
        import time
        
        for seed in range(num_seeds):
            print(f"\n=== Seed {seed} ===")
            
            for name, info in self.results.items():
                print(f"\nTraining {name}...")
                
                # 创建环境和训练器
                env = self.env_creator()
                trainer = info['trainer_class'](env, info['config'])
                
                # 训练
                start_time = time.time()
                trainer.train(num_episodes)
                training_time = time.time() - start_time
                
                # 记录结果
                info['rewards'].append(trainer.eval_rewards)
                info['success_rates'].append(trainer.success_rates)
                info['training_time'] += training_time
                
                env.close()
                
    def plot_comparison(self):
        """绘制比较结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 奖励对比
        for name, info in self.results.items():
            if info['rewards']:
                mean_rewards = np.mean(info['rewards'], axis=0)
                std_rewards = np.std(info['rewards'], axis=0)
                x = np.arange(len(mean_rewards))
                
                ax1.plot(x, mean_rewards, label=name, linewidth=2)
                ax1.fill_between(x, 
                                mean_rewards - std_rewards,
                                mean_rewards + std_rewards,
                                alpha=0.3)
                                
        ax1.set_xlabel('Evaluation')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Algorithm Comparison - Rewards')
        ax1.legend()
        ax1.grid(True)
        
        # 成功率对比
        for name, info in self.results.items():
            if info['success_rates']:
                mean_success = np.mean(info['success_rates'], axis=0)
                std_success = np.std(info['success_rates'], axis=0)
                x = np.arange(len(mean_success))
                
                ax2.plot(x, mean_success, label=name, linewidth=2)
                ax2.fill_between(x,
                                mean_success - std_success,
                                mean_success + std_success,
                                alpha=0.3)
                                
        ax2.set_xlabel('Evaluation')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Algorithm Comparison - Success Rate')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300)
        plt.show()
        
        # 打印统计信息
        print("\n=== Training Statistics ===")
        for name, info in self.results.items():
            avg_time = info['training_time'] / len(info['rewards'])
            final_reward = np.mean([r[-1] for r in info['rewards']])
            final_success = np.mean([s[-1] for s in info['success_rates']])
            
            print(f"\n{name}:")
            print(f"  Average training time: {avg_time:.1f} seconds")
            print(f"  Final reward: {final_reward:.2f} ± {np.std([r[-1] for r in info['rewards']]):.2f}")
            print(f"  Final success rate: {final_success:.2%} ± {np.std([s[-1] for s in info['success_rates']]):.2%}")