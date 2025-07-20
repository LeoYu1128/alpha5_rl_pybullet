# """
# PPO (Proximal Policy Optimization) 训练器
# """

# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.distributions import Normal
# import time

# from common.base_trainer import BaseTrainer
# from common.networks import ActorNetwork, CriticNetwork
# from common.utils import RolloutBuffer


# class PPOTrainer(BaseTrainer):
#     """PPO训练器"""
    
#     def __init__(self, env, config=None):
#         # PPO默认配置
#         ppo_config = {
#             'lr_actor': 3e-4,
#             'lr_critic': 3e-4,
#             'ppo_epochs': 10,
#             'clip_epsilon': 0.2,
#             'value_coeff': 0.5,
#             'entropy_coeff': 0.01,
#             'gae_lambda': 0.95,
#             'max_grad_norm': 0.5,
#             'rollout_length': 2048,
#             'mini_batch_size': 64,
#             'advantage_normalization': True,
#             'lr_decay': True,
#             'lr_decay_rate': 0.99,
#             'clip_value_loss': True,
#         }
        
#         if config:
#             ppo_config.update(config)
            
#         super().__init__(env, ppo_config)
        
#         # 获取环境信息
#         self.state_dim = env.observation_space.shape[0]
#         self.action_dim = env.action_space.shape[0]
        
#         # 创建网络
#         self.actor = ActorNetwork(
#             self.state_dim, 
#             self.action_dim,
#             hidden_dims=[256, 256]
#         ).to(self.device)
        
#         self.critic = CriticNetwork(
#             self.state_dim,
#             hidden_dims=[256, 256]
#         ).to(self.device)
        
#         # 优化器
#         self.actor_optimizer = optim.Adam(
#             self.actor.parameters(), 
#             lr=self.config['lr_actor']
#         )
#         self.critic_optimizer = optim.Adam(
#             self.critic.parameters(), 
#             lr=self.config['lr_critic']
#         )
        
#         # 学习率调度器
#         if self.config['lr_decay']:
#             self.actor_scheduler = optim.lr_scheduler.ExponentialLR(
#                 self.actor_optimizer, 
#                 gamma=self.config['lr_decay_rate']
#             )
#             self.critic_scheduler = optim.lr_scheduler.ExponentialLR(
#                 self.critic_optimizer, 
#                 gamma=self.config['lr_decay_rate']
#             )
        
#         # Rollout缓冲区
#         self.rollout_buffer = RolloutBuffer(
#             buffer_size=self.config['rollout_length'],
#             state_dim=self.state_dim,
#             action_dim=self.action_dim
#         )
        
#         # 训练统计
#         self.ppo_stats = {
#             'policy_loss': [],
#             'value_loss': [],
#             'entropy': [],
#             'kl_divergence': [],
#             'clip_fraction': []
#         }
        
#     def train(self, num_episodes):
#         """PPO训练主循环"""
#         print(f"\nStarting PPO training for {num_episodes} episodes...")
#         start_time = time.time()
        
#         state, _ = self.env.reset()
#         episode_reward = 0
#         episode_length = 0
        
#         while self.episode_count < num_episodes:
#             # 收集rollout
#             self.rollout_buffer.reset()
            
#             with torch.no_grad():
#                 for step in range(self.config['rollout_length']):
#                     # 获取动作
#                     state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#                     action, log_prob, value = self.get_action_and_value(state_tensor)
                    
#                     # 执行动作
#                     next_state, reward, terminated, truncated, info = self.env.step(action)
#                     done = terminated or truncated
                    
#                     # 存储经验
#                     self.rollout_buffer.add(
#                         state, action, reward, done, 
#                         log_prob, value.squeeze().cpu().numpy()
#                     )
                    
#                     # 更新统计
#                     episode_reward += reward
#                     episode_length += 1
#                     self.total_steps += 1
                    
#                     if done:
#                         # 记录episode信息
#                         episode_info = {
#                             'reward': episode_reward,
#                             'length': episode_length,
#                             'success': info.get('success', False),
#                             'safety_violations': info.get('safety_penalty', 0) > 0
#                         }
#                         self.log_episode(episode_info)
                        
#                         # 重置episode
#                         state, _ = self.env.reset()
#                         episode_reward = 0
#                         episode_length = 0
#                         self.episode_count += 1
                        
#                         if self.episode_count >= num_episodes:
#                             break
#                     else:
#                         state = next_state
                        
#             # 计算优势和回报
#             with torch.no_grad():
#                 last_value = self.critic(
#                     torch.FloatTensor(state).unsqueeze(0).to(self.device)
#                 ).squeeze().cpu().numpy()
                
#             self.rollout_buffer.compute_returns_and_advantages(
#                 last_value, 
#                 self.config['gamma'], 
#                 self.config['gae_lambda']
#             )
            
#             # PPO更新
#             self.update_policy()
            
#             # 评估
#             if self.should_evaluate():
#                 eval_info = self.evaluate()
#                 self.training_metrics['eval_rewards'].append(eval_info['avg_reward'])
#                 self.training_metrics['eval_success_rate'].append(eval_info['success_rate'])
                
#                 print(f"\nEvaluation at step {self.total_steps}:")
#                 print(f"  Avg Reward: {eval_info['avg_reward']:.2f} ± {eval_info['std_reward']:.2f}")
#                 print(f"  Success Rate: {eval_info['success_rate']:.2%}")
#                 print(f"  Avg Safety Violations: {eval_info['avg_safety_violations']:.2f}")
                
#                 # 早停检查
#                 if self.check_early_stopping(eval_info['avg_reward']):
#                     break
                    
#                 # 保存最佳模型
#                 if eval_info['avg_reward'] > self.best_eval_reward:
#                     self.save_checkpoint('best')
                    
#             # 保存检查点
#             if self.should_save_checkpoint():
#                 self.save_checkpoint()
                
#             # 学习率衰减
#             if self.config['lr_decay'] and self.episode_count % 1000 == 0:
#                 self.actor_scheduler.step()
#                 self.critic_scheduler.step()
                
#         # 训练结束
#         training_time = time.time() - start_time
#         self.training_metrics['training_time'].append(training_time)
        
#         # 绘制训练曲线
#         self.plot_training_curves()
#         self.plot_ppo_stats()
        
#         print(f"\nTraining completed in {training_time/3600:.2f} hours")
        
#     def get_action_and_value(self, state):
#         """获取动作、对数概率和价值"""
#         # Actor网络输出
#         mean, std = self.actor(state)
#         dist = Normal(mean, std)
        
#         # 采样动作
#         action_sample = dist.sample()
#         action = torch.tanh(action_sample)  # 限制到[-1, 1]
        
#         # 计算对数概率（考虑tanh变换）
#         log_prob = dist.log_prob(action_sample).sum(dim=-1)
#         log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=-1)
        
#         # Critic网络输出
#         value = self.critic(state)
        
#         return action.cpu().numpy().squeeze(), log_prob.cpu().numpy(), value
        
#     def update_policy(self):
#         """PPO策略更新"""
#         # 获取rollout数据
#         rollout_data = self.rollout_buffer.get()
        
#         # 转换为tensor
#         states = torch.FloatTensor(rollout_data['states']).to(self.device)
#         actions = torch.FloatTensor(rollout_data['actions']).to(self.device)
#         old_log_probs = torch.FloatTensor(rollout_data['log_probs']).to(self.device)
#         advantages = torch.FloatTensor(rollout_data['advantages']).to(self.device)
#         returns = torch.FloatTensor(rollout_data['returns']).to(self.device)
#         old_values = torch.FloatTensor(rollout_data['values']).to(self.device)
        
#         # 优势标准化
#         if self.config['advantage_normalization']:
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
#         # PPO epochs
#         dataset_size = len(states)
#         batch_size = self.config['mini_batch_size']
        
#         for epoch in range(self.config['ppo_epochs']):
#             # 随机打乱数据
#             indices = np.random.permutation(dataset_size)
            
#             # Mini-batch更新
#             for start in range(0, dataset_size, batch_size):
#                 end = start + batch_size
#                 batch_indices = indices[start:end]
                
#                 batch_states = states[batch_indices]
#                 batch_actions = actions[batch_indices]
#                 batch_old_log_probs = old_log_probs[batch_indices]
#                 batch_advantages = advantages[batch_indices]
#                 batch_returns = returns[batch_indices]
#                 batch_old_values = old_values[batch_indices]
                
#                 # 计算新的动作概率
#                 mean, std = self.actor(batch_states)
#                 dist = Normal(mean, std)
                
#                 # 需要逆变换来获取原始动作
#                 action_sample = torch.atanh(torch.clamp(batch_actions, -0.999, 0.999))
#                 new_log_probs = dist.log_prob(action_sample).sum(dim=-1)
#                 new_log_probs -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=-1)
                
#                 # 计算比率
#                 ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
#                 # PPO损失
#                 surr1 = ratio * batch_advantages
#                 surr2 = torch.clamp(ratio, 
#                                    1 - self.config['clip_epsilon'], 
#                                    1 + self.config['clip_epsilon']) * batch_advantages
#                 policy_loss = -torch.min(surr1, surr2).mean()
                
#                 # 价值损失
#                 values = self.critic(batch_states).squeeze()
                
#                 if self.config['clip_value_loss']:
#                     # 裁剪价值损失
#                     values_clipped = batch_old_values + torch.clamp(
#                         values - batch_old_values,
#                         -self.config['clip_epsilon'],
#                         self.config['clip_epsilon']
#                     )
#                     value_loss_unclipped = F.mse_loss(values, batch_returns)
#                     value_loss_clipped = F.mse_loss(values_clipped, batch_returns)
#                     value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
#                 else:
#                     value_loss = F.mse_loss(values, batch_returns)
                    
#                 # 熵损失
#                 entropy = dist.entropy().mean()
                
#                 # 总损失
#                 loss = (policy_loss + 
#                        self.config['value_coeff'] * value_loss - 
#                        self.config['entropy_coeff'] * entropy)
                
#                 # 更新Actor
#                 self.actor_optimizer.zero_grad()
#                 policy_loss.backward()
#                 nn.utils.clip_grad_norm_(self.actor.parameters(), self.config['max_grad_norm'])
#                 self.actor_optimizer.step()
                
#                 # 更新Critic
#                 self.critic_optimizer.zero_grad()
#                 value_loss.backward()
#                 nn.utils.clip_grad_norm_(self.critic.parameters(), self.config['max_grad_norm'])
#                 self.critic_optimizer.step()
                
#                 # 记录统计
#                 with torch.no_grad():
#                     kl = (batch_old_log_probs - new_log_probs).mean()
#                     clip_fraction = ((ratio - 1).abs() > self.config['clip_epsilon']).float().mean()
                    
#                 self.ppo_stats['policy_loss'].append(policy_loss.item())
#                 self.ppo_stats['value_loss'].append(value_loss.item())
#                 self.ppo_stats['entropy'].append(entropy.item())
#                 self.ppo_stats['kl_divergence'].append(kl.item())
#                 self.ppo_stats['clip_fraction'].append(clip_fraction.item())
                
#     def select_action(self, state, evaluate=False):
#         """选择动作"""
#         with torch.no_grad():
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
#             if evaluate:
#                 # 评估时使用确定性策略
#                 mean, _ = self.actor(state_tensor)
#                 action = torch.tanh(mean)
#             else:
#                 # 训练时从分布中采样
#                 action, _, _ = self.get_action_and_value(state_tensor)
#                 action = action.squeeze()
                
#         return action if isinstance(action, np.ndarray) else action.cpu().numpy()
        
#     def plot_ppo_stats(self):
#         """绘制PPO特定的统计信息"""
#         import matplotlib.pyplot as plt
        
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#         fig.suptitle('PPO Training Statistics', fontsize=16)
        
#         stats_to_plot = [
#             ('policy_loss', 'Policy Loss'),
#             ('value_loss', 'Value Loss'),
#             ('entropy', 'Entropy'),
#             ('kl_divergence', 'KL Divergence'),
#             ('clip_fraction', 'Clip Fraction')
#         ]
        
#         for idx, (stat_name, title) in enumerate(stats_to_plot):
#             ax = axes[idx // 3, idx % 3]
#             if self.ppo_stats[stat_name]:
#                 ax.plot(self.ppo_stats[stat_name])
#                 ax.set_title(title)
#                 ax.set_xlabel('Update Step')
#                 ax.grid(True)
                
#         # 隐藏最后一个子图
#         axes[1, 2].axis('off')
        
#         plt.tight_layout()
#         plot_path = os.path.join(self.plot_dir, 'ppo_stats.png')
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#     def get_state_dict(self):
#         """获取模型状态"""
#         return {
#             'actor': self.actor.state_dict(),
#             'critic': self.critic.state_dict(),
#             'actor_optimizer': self.actor_optimizer.state_dict(),
#             'critic_optimizer': self.critic_optimizer.state_dict(),
#             'ppo_stats': self.ppo_stats
#         }
        
#     def load_state_dict(self, state_dict):
#         """加载模型状态"""
#         self.actor.load_state_dict(state_dict['actor'])
#         self.critic.load_state_dict(state_dict['critic'])
#         self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
#         self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
#         if 'ppo_stats' in state_dict:
#             self.ppo_stats = state_dict['ppo_stats']

import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
