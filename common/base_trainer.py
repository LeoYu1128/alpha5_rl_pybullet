
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime
import pandas as pd
from collections import deque


class BaseTrainer(ABC):
    """基础训练器类"""
    
    def __init__(self, env, config=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 默认配置
        self.config = {
            'lr': 3e-4,
            'batch_size': 256,
            'gamma': 0.99,
            'buffer_size': 1000000,
            'save_freq': 1000,
            'eval_freq': 1000,
            'eval_episodes': 10,
            'log_freq': 100,
            'render_eval': False,
            'checkpoint_freq': 5000,
            'early_stopping_patience': 20,
            'target_success_rate': 0.95,
        }
        
        if config:
            self.config.update(config)
            
        # 获取算法名称
        self.algorithm_name = self.__class__.__name__.replace('Trainer', '')
        
        # 创建结果目录
        self.setup_directories()
        
        # 性能追踪
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'eval_rewards': [],
            'eval_success_rate': [],
            'learning_rate': [],
            'loss': [],
            'safety_violations': [],
            'training_time': []
        }
        
        # 训练状态
        self.total_steps = 0
        self.episode_count = 0
        self.best_eval_reward = -float('inf')
        self.patience_counter = 0
        
        # 保存配置
        self.save_config()
        
    def setup_directories(self):
        """设置目录结构"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f"{self.algorithm_name}_{timestamp}"
        
        # 创建目录
        self.base_dir = f"{self.algorithm_name}/results/{self.run_name}"
        self.model_dir = os.path.join(self.base_dir, "models")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.plot_dir = os.path.join(self.base_dir, "plots")
        
        for dir_path in [self.model_dir, self.log_dir, self.plot_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def save_config(self):
        """保存配置文件"""
        config_path = os.path.join(self.base_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    @abstractmethod
    def train(self, num_episodes):
        """训练方法（子类实现）"""
        pass
        
    @abstractmethod
    def select_action(self, state, evaluate=False):
        """选择动作（子类实现）"""
        pass
        
    def evaluate(self, num_episodes=None, render=False):
        """评估当前策略"""
        if num_episodes is None:
            num_episodes = self.config['eval_episodes']
            
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        eval_safety_violations = []
        
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            safety_violations = 0
            
            while not done and episode_length < self.env.max_steps:
                with torch.no_grad():
                    action = self.select_action(state, evaluate=True)
                    
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # 检查安全违规
                if 'safety_penalty' in info and info['safety_penalty'] > 0:
                    safety_violations += 1
                    
                done = terminated or truncated
                state = next_state
                
                if render:
                    self.env.render()
                    
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(info.get('success', False))
            eval_safety_violations.append(safety_violations)
            
        # 计算统计量
        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_length = np.mean(eval_lengths)
        success_rate = np.mean(eval_successes)
        avg_violations = np.mean(eval_safety_violations)
        
        eval_info = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': avg_length,
            'success_rate': success_rate,
            'avg_safety_violations': avg_violations,
            'rewards': eval_rewards
        }
        
        return eval_info
        
    def save_checkpoint(self, tag='checkpoint'):
        """保存检查点"""
        checkpoint_path = os.path.join(self.model_dir, f"{tag}_step_{self.total_steps}.pth")
        
        checkpoint = {
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'best_eval_reward': self.best_eval_reward,
            'training_metrics': self.training_metrics,
            'model_state': self.get_state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # 保存最佳模型
        if tag == 'best':
            best_path = os.path.join(self.model_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.total_steps = checkpoint['total_steps']
        self.episode_count = checkpoint['episode_count']
        self.best_eval_reward = checkpoint['best_eval_reward']
        self.training_metrics = checkpoint['training_metrics']
        self.load_state_dict(checkpoint['model_state'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resuming from step {self.total_steps}")
        
    def log_episode(self, episode_info):
        """记录单个episode的信息"""
        # 更新指标
        self.training_metrics['episode_rewards'].append(episode_info['reward'])
        self.training_metrics['episode_lengths'].append(episode_info['length'])
        self.training_metrics['success_rate'].append(episode_info.get('success', False))
        
        if 'safety_violations' in episode_info:
            self.training_metrics['safety_violations'].append(episode_info['safety_violations'])
            
        # 打印日志
        if self.episode_count % self.config['log_freq'] == 0:
            recent_rewards = self.training_metrics['episode_rewards'][-100:]
            recent_success = self.training_metrics['success_rate'][-100:]
            
            print(f"\nEpisode {self.episode_count} | "
                  f"Steps: {self.total_steps} | "
                  f"Reward: {episode_info['reward']:.2f} | "
                  f"Avg Reward: {np.mean(recent_rewards):.2f} | "
                  f"Success Rate: {np.mean(recent_success):.2%}")
                  
    def should_evaluate(self):
        """检查是否应该评估"""
        return self.total_steps % self.config['eval_freq'] == 0
        
    def should_save_checkpoint(self):
        """检查是否应该保存检查点"""
        return self.total_steps % self.config['checkpoint_freq'] == 0
        
    def check_early_stopping(self, eval_reward):
        """检查早停条件"""
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['early_stopping_patience']:
                print("Early stopping triggered!")
                return True
        return False
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        metrics = self.training_metrics
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.algorithm_name} Training Progress', fontsize=16)
        
        # 1. Episode Rewards
        ax = axes[0, 0]
        episode_rewards = metrics['episode_rewards']
        if episode_rewards:
            ax.plot(episode_rewards, alpha=0.3, label='Episode')
            if len(episode_rewards) > 100:
                smoothed = pd.Series(episode_rewards).rolling(100).mean()
                ax.plot(smoothed, label='Moving Avg (100)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Rewards')
            ax.legend()
            ax.grid(True)
            
        # 2. Success Rate
        ax = axes[0, 1]
        if metrics['success_rate']:
            success_rate = pd.Series(metrics['success_rate']).rolling(100).mean()
            ax.plot(success_rate.values)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Success Rate')
            ax.set_title('Success Rate (100-episode avg)')
            ax.grid(True)
            ax.set_ylim([0, 1])
            
        # 3. Evaluation Performance
        ax = axes[0, 2]
        if metrics['eval_rewards']:
            eval_steps = np.arange(len(metrics['eval_rewards'])) * self.config['eval_freq']
            ax.plot(eval_steps, metrics['eval_rewards'], 'o-', label='Eval Reward')
            ax2 = ax.twinx()
            ax2.plot(eval_steps, metrics['eval_success_rate'], 's-', 
                    color='orange', label='Success Rate')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Eval Reward')
            ax2.set_ylabel('Success Rate')
            ax.set_title('Evaluation Performance')
            ax.grid(True)
            
        # 4. Episode Length
        ax = axes[1, 0]
        if metrics['episode_lengths']:
            ax.plot(metrics['episode_lengths'], alpha=0.3)
            if len(metrics['episode_lengths']) > 100:
                smoothed = pd.Series(metrics['episode_lengths']).rolling(100).mean()
                ax.plot(smoothed, label='Moving Avg (100)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
            ax.set_title('Episode Length')
            ax.grid(True)
            
        # 5. Safety Violations
        ax = axes[1, 1]
        if metrics['safety_violations']:
            violations = pd.Series(metrics['safety_violations']).rolling(100).mean()
            ax.plot(violations.values)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Violations per Episode')
            ax.set_title('Safety Violations (100-episode avg)')
            ax.grid(True)
            
        # 6. Loss (if available)
        ax = axes[1, 2]
        if metrics['loss']:
            ax.plot(metrics['loss'])
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True)
            ax.set_yscale('log')
            
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.plot_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存最终的统计信息
        self.save_final_stats()
        
    def save_final_stats(self):
        """保存最终统计信息"""
        if not self.training_metrics['episode_rewards']:
            return
            
        stats = {
            'algorithm': self.algorithm_name,
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'final_success_rate': np.mean(self.training_metrics['success_rate'][-100:]),
            'final_avg_reward': np.mean(self.training_metrics['episode_rewards'][-100:]),
            'best_eval_reward': self.best_eval_reward,
            'avg_episode_length': np.mean(self.training_metrics['episode_lengths']),
            'training_time_hours': sum(self.training_metrics.get('training_time', [0])) / 3600
        }
        
        # 添加评估统计
        if self.training_metrics['eval_rewards']:
            stats['final_eval_reward'] = self.training_metrics['eval_rewards'][-1]
            stats['final_eval_success_rate'] = self.training_metrics['eval_success_rate'][-1]
            stats['max_eval_reward'] = max(self.training_metrics['eval_rewards'])
            stats['max_eval_success_rate'] = max(self.training_metrics['eval_success_rate'])
        
        # 保存统计信息
        stats_path = os.path.join(self.base_dir, 'final_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
            
        # 打印总结
        print("\n" + "="*50)
        print(f"{self.algorithm_name} Training Complete!")
        print("="*50)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*50)
        
    @abstractmethod
    def get_state_dict(self):
        """获取模型状态（子类实现）"""
        pass
        
    @abstractmethod
    def load_state_dict(self, state_dict):
        """加载模型状态（子类实现）"""
        pass