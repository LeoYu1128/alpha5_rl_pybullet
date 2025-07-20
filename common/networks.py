"""
共享的神经网络模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaseNetwork(nn.Module):
    """基础网络类"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256], 
                 activation='relu', use_layer_norm=True, dropout_rate=0.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        
        # 选择激活函数
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'leaky_relu': nn.LeakyReLU()
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
                
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            # Xavier初始化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        return self.network(x)


class ActorNetwork(BaseNetwork):
    """Actor网络（策略网络）"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256],
                 log_std_min=-20, log_std_max=2, activation='relu'):
        # Actor输出均值和标准差
        super().__init__(state_dim, action_dim * 2, hidden_dims, activation)
        
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state):
        output = self.network(state)
        mean, log_std = output.split(self.action_dim, dim=-1)
        
        # 限制log_std的范围，确保数值稳定
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mean, std
        
    def sample(self, state):
        """从策略分布中采样"""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        
        # 重参数化采样
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Tanh squashing
        action_tanh = torch.tanh(action)
        
        # 修正对数概率（考虑tanh变换）
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)
        
        return action_tanh, log_prob, mean, std
        
    def deterministic_action(self, state):
        """获取确定性动作（用于评估）"""
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class CriticNetwork(BaseNetwork):
    """Critic网络（价值网络）"""
    
    def __init__(self, state_dim, action_dim=0, hidden_dims=[256, 256],
                 activation='relu', use_layer_norm=True):
        # 如果action_dim > 0，则是Q网络；否则是V网络
        input_dim = state_dim + action_dim
        super().__init__(input_dim, 1, hidden_dims, activation, use_layer_norm)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def forward(self, state, action=None):
        if self.action_dim > 0 and action is not None:
            # Q网络：连接状态和动作
            x = torch.cat([state, action], dim=-1)
        else:
            # V网络：只使用状态
            x = state
            
        return self.network(x)


class DoubleCritic(nn.Module):
    """双Q网络（用于SAC、TD3等）"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        self.q1 = CriticNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = CriticNetwork(state_dim, action_dim, hidden_dims)
        
    def forward(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2
        
    def q1_forward(self, state, action):
        return self.q1(state, action)
        
    def q2_forward(self, state, action):
        return self.q2(state, action)


class DeterministicActor(BaseNetwork):
    """确定性Actor网络（用于DDPG、TD3）"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256],
                 activation='relu', max_action=1.0):
        super().__init__(state_dim, action_dim, hidden_dims, activation)
        
        self.max_action = max_action
        
    def forward(self, state):
        action = self.network(state)
        # 使用tanh限制动作范围
        return self.max_action * torch.tanh(action)


class EnsembleNetwork(nn.Module):
    """集成网络（用于提高鲁棒性）"""
    
    def __init__(self, network_class, num_networks=3, **network_kwargs):
        super().__init__()
        
        self.networks = nn.ModuleList([
            network_class(**network_kwargs) for _ in range(num_networks)
        ])
        self.num_networks = num_networks
        
    def forward(self, *args, **kwargs):
        outputs = []
        for network in self.networks:
            outputs.append(network(*args, **kwargs))
            
        # 返回均值和标准差
        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        
        return mean, std
        
    def sample_network(self):
        """随机选择一个网络"""
        idx = np.random.randint(self.num_networks)
        return self.networks[idx]