"""
共享的工具函数和类
"""

import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        """添加经验（使用最大优先级）"""
        super().add(state, action, reward, next_state, done)
        self.priorities[self.ptr - 1] = self.max_priority
        
    def sample(self, batch_size):
        """优先采样"""
        # 计算采样概率
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # 计算重要性权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'indices': indices,
            'weights': weights
        }
        
        return batch
        
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority)
            
    def increase_beta(self, increment=0.001):
        """增加beta（用于退火）"""
        self.beta = min(1.0, self.beta + increment)


class RolloutBuffer:
    """PPO的Rollout缓冲区"""
    
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.reset()
        
    def reset(self):
        """重置缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None
        self.ptr = 0
        
    def add(self, state, action, reward, done, log_prob, value):
        """添加单步经验"""
        if self.ptr >= self.buffer_size:
            return
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.ptr += 1
        
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """计算回报和GAE优势"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)
        
        # 添加最后的价值估计
        values = np.append(values, last_value)
        
        # 计算GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
        # 计算回报
        returns = advantages + values[:-1]
        
        self.advantages = advantages
        self.returns = returns
        
    def get(self):
        """获取所有数据"""
        assert self.advantages is not None, "Call compute_returns_and_advantages first!"
        
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'log_probs': np.array(self.log_probs),
            'advantages': self.advantages,
            'returns': self.returns,
            'values': np.array(self.values)
        }


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck噪声（用于连续控制）"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
        
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.size) * self.mu
        
    def sample(self):
        """生成噪声样本"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class GaussianNoise:
    """高斯噪声"""
    
    def __init__(self, size, mu=0.0, sigma=0.1):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        
    def sample(self):
        """生成噪声样本"""
        return np.random.normal(self.mu, self.sigma, self.size)
        
    def reset(self):
        """重置（高斯噪声无需重置）"""
        pass


def soft_update(target_network, source_network, tau):
    """软更新目标网络"""
    for target_param, source_param in zip(target_network.parameters(), 
                                         source_network.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target_network, source_network):
    """硬更新目标网络"""
    target_network.load_state_dict(source_network.state_dict())


def normalize_angle(angle):
    """归一化角度到[-pi, pi]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def compute_gradient_norm(network):
    """计算梯度范数"""
    total_norm = 0
    for p in network.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class RunningMeanStd:
    """运行时均值和标准差"""
    
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        
    def update(self, x):
        """更新统计量"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
        
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """从矩更新"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
        
    def normalize(self, x):
        """标准化"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)
        
    def denormalize(self, x):
        """反标准化"""
        return x * np.sqrt(self.var + 1e-8) + self.mean


def set_random_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        
def get_linear_schedule(initial_value, final_value, total_steps):
    """获取线性调度函数"""
    def schedule(step):
        if step >= total_steps:
            return final_value
        else:
            return initial_value + (final_value - initial_value) * step / total_steps
    return schedule


def get_exponential_schedule(initial_value, decay_rate):
    """获取指数调度函数"""
    def schedule(step):
        return initial_value * (decay_rate ** step)
    return schedule, reward, next_state, done):
        """添加经验"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """采样批次"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
    def __len__(self):
        return self.size
        
    def save(self, path):
        """保存缓冲区"""
        np.savez(path,
                 states=self.states[:self.size],
                 actions=self.actions[:self.size],
                 rewards=self.rewards[:self.size],
                 next_states=self.next_states[:self.size],
                 dones=self.dones[:self.size])
                 
    def load(self, path):
        """加载缓冲区"""
        data = np.load(path)
        self.size = len(data['states'])
        self.states[:self.size] = data['states']
        self.actions[:self.size] = data['actions']
        self.rewards[:self.size] = data['rewards']
        self.next_states[:self.size] = data['next_states']
        self.dones[:self.size] = data['dones']


class PrioritizedReplayBuffer(ReplayBuffer):
    """优先经验回放缓冲区"""
    
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6, beta=0.4):
        super().__init__(capacity, state_dim, action_dim)
        
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.epsilon = 1e-6  # 防止优先级为0
        
        # 优先级树
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
    def add(self, state, action