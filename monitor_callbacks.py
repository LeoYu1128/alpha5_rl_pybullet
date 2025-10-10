import os
import numpy as np
import json
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
# ✅ 在import matplotlib之前设置后端
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ComprehensiveMonitor(BaseCallback):
    """
    论文级别的综合监控系统
    记录所有训练指标，用于生成论文图表
    """
    
    def __init__(self, log_dir, verbose=1, window_size=100):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.window_size = window_size
        
        # 创建子目录
        os.makedirs(os.path.join(log_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
        
        # ===== 核心训练指标 =====
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.timesteps = []
        
        # ===== 任务特定指标 =====
        self.final_distances = []
        self.target_drifts = []
        self.velocity_errors = []
        self.approach_velocities = []
        
        # ===== 奖励分解 =====
        self.reward_components = {
            'distance': [],
            'prediction': [],
            'velocity_match': [],
            'progress': [],
            'control': [],
            'success': []
        }
        
        # ===== 控制平滑度 =====
        self.action_norms = []
        self.action_changes = []
        self.previous_action = None
        
        # ===== 滑动窗口统计 =====
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_successes = deque(maxlen=window_size)
        self.recent_distances = deque(maxlen=window_size)
        
        # ===== Episode内的详细记录 =====
        self.current_episode_data = {
            'rewards': [],
            'distances': [],
            'actions': [],
            'positions': []
        }
        
        # 当前episode计数
        self.n_episodes = 0
        self.current_step_in_episode = 0
        
    def _on_step(self) -> bool:
        # """每步调用"""
        # # 获取info
        # infos = self.locals.get('infos', [])
        # actions = self.locals.get('actions', None)
        
        # for info in infos:
        #     # ===== 记录episode内数据 =====
        #     if 'reward' in self.locals:
        #         self.current_episode_data['rewards'].append(self.locals['rewards'][0])
            
        #     if 'distance' in info:
        #         self.current_episode_data['distances'].append(info['distance'])
            
        #     if actions is not None:
        #         action = actions[0] if len(actions.shape) > 1 else actions
        #         self.current_episode_data['actions'].append(action.copy())
                
        #         # 记录动作平滑度
        #         action_norm = np.linalg.norm(action)
        #         self.action_norms.append(action_norm)
                
        #         if self.previous_action is not None:
        #             action_change = np.linalg.norm(action - self.previous_action)
        #             self.action_changes.append(action_change)
                
        #         self.previous_action = action.copy()
            
        #     # ===== 记录奖励分解 =====
        #     if 'reward_distance' in info:
        #         self.reward_components['distance'].append(info['reward_distance'])
        #     if 'reward_prediction' in info:
        #         self.reward_components['prediction'].append(info['reward_prediction'])
        #     if 'reward_velocity_match' in info:
        #         self.reward_components['velocity_match'].append(info['reward_velocity_match'])
        #     if 'reward_progress_vel' in info:
        #         self.reward_components['progress'].append(info['reward_progress_vel'])
        #     if 'reward_control' in info:
        #         self.reward_components['control'].append(info['reward_control'])
        #     if 'reward_success' in info:
        #         self.reward_components['success'].append(info['reward_success'])
            
        #     # ===== Episode结束时的处理 =====
        #     if info.get('terminal_observation') is not None or 'episode' in info:
        #         self.n_episodes += 1
                
        #         # 基础指标
        #         episode_reward = info.get('episode', {}).get('r', sum(self.current_episode_data['rewards']))
        #         episode_length = info.get('episode', {}).get('l', len(self.current_episode_data['rewards']))
        #         success = info.get('success', False)
        #         final_distance = info.get('distance', 0)
                
        #         self.episode_rewards.append(episode_reward)
        #         self.episode_lengths.append(episode_length)
        #         self.episode_successes.append(1.0 if success else 0.0)
        #         self.final_distances.append(final_distance)
        #         self.timesteps.append(self.num_timesteps)
                
        #         # 滑动窗口
        #         self.recent_rewards.append(episode_reward)
        #         self.recent_successes.append(1.0 if success else 0.0)
        #         self.recent_distances.append(final_distance)
                
        #         # 任务特定指标
        #         if 'target_drift' in info:
        #             self.target_drifts.append(info['target_drift'])
                
        #         # 计算平均速度误差
        #         if len(self.current_episode_data['distances']) > 0:
        #             avg_approach_vel = np.mean(np.diff(self.current_episode_data['distances'])) if len(self.current_episode_data['distances']) > 1 else 0
        #             self.approach_velocities.append(avg_approach_vel)
                
        #         # 每N个episode保存一次
        #         if self.n_episodes % 10 == 0:
        #             self._save_metrics()
        #             if self.n_episodes % 50 == 0:
        #                 self._plot_metrics()
                
        #         # 重置episode数据
        #         self.current_episode_data = {
        #             'rewards': [],
        #             'distances': [],
        #             'actions': [],
        #             'positions': []
        #         }
                
        #         self.previous_action = None
        
        # self.current_step_in_episode += 1
        return True
    
    def _save_metrics(self):
        """保存指标到JSON - 修复版"""
        
        # ✅ 类型转换辅助函数
        def convert_to_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        try:
            metrics = {
                'timesteps': self.timesteps,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'episode_successes': self.episode_successes,
                'final_distances': self.final_distances,
                'target_drifts': self.target_drifts,
                'approach_velocities': self.approach_velocities,
                'action_norms': self.action_norms[-1000:],  # 只保存最近1000个
                'action_changes': self.action_changes[-1000:],
                'reward_components': {
                    k: v[-1000:] for k, v in self.reward_components.items()
                },
                # 滑动窗口统计
                'recent_stats': {
                    'mean_reward': float(np.mean(self.recent_rewards)) if self.recent_rewards else 0,
                    'mean_success_rate': float(np.mean(self.recent_successes)) if self.recent_successes else 0,
                    'mean_distance': float(np.mean(self.recent_distances)) if self.recent_distances else 0
                }
            }
            
            # ✅ 转换所有数据为可序列化类型
            metrics = convert_to_serializable(metrics)
            
            filepath = os.path.join(self.log_dir, 'metrics', f'metrics_ep{self.n_episodes}.json')
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            if self.verbose > 0 and self.n_episodes % 100 == 0:
                print(f"✅ 已保存指标到: {filepath}")
                
        except Exception as e:
            print(f"⚠️ 保存指标失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_metrics(self):
        """生成所有图表"""
        print(f"\n{'='*60}")
        print(f"生成训练图表 (Episode {self.n_episodes})...")
        print(f"{'='*60}")
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
        
        # ===== 图1: 核心训练曲线 (2x2) =====
        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle(f'Training Progress (Episode {self.n_episodes})', fontsize=14, fontweight='bold')
        
        # 1.1 Episode Reward
        if len(self.episode_rewards) > 0:
            axes[0, 0].plot(self.timesteps, self.episode_rewards, alpha=0.3, color='#2E86AB', linewidth=0.5)
            if len(self.episode_rewards) >= 50:
                smoothed = self._smooth(self.episode_rewards, 50)
                axes[0, 0].plot(self.timesteps, smoothed, color='#2E86AB', linewidth=2, label='Smoothed (50ep)')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Episode Reward')
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        
        # 1.2 Success Rate
        if len(self.episode_successes) > 0:
            window_size = min(100, len(self.episode_successes))
            success_rate = self._moving_average(self.episode_successes, window_size)
            axes[0, 1].plot(self.timesteps, success_rate, color='#27AE60', linewidth=2)
            axes[0, 1].fill_between(self.timesteps, 0, success_rate, alpha=0.3, color='#27AE60')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].set_title(f'Success Rate (rolling {window_size}ep)')
            axes[0, 1].set_ylim([0, 1.05])
            axes[0, 1].grid(alpha=0.3)
        
        # 1.3 Final Distance
        if len(self.final_distances) > 0:
            axes[1, 0].plot(self.timesteps, self.final_distances, alpha=0.3, color='#E74C3C', linewidth=0.5)
            if len(self.final_distances) >= 50:
                smoothed_dist = self._smooth(self.final_distances, 50)
                axes[1, 0].plot(self.timesteps, smoothed_dist, color='#E74C3C', linewidth=2, label='Smoothed')
            axes[1, 0].axhline(y=0.02, color='green', linestyle='--', label='Success Threshold', linewidth=1.5)
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Final Distance (m)')
            axes[1, 0].set_title('Final Distance to Target')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        # 1.4 Episode Length
        if len(self.episode_lengths) > 0:
            axes[1, 1].plot(self.timesteps, self.episode_lengths, alpha=0.3, color='#9B59B6', linewidth=0.5)
            if len(self.episode_lengths) >= 50:
                smoothed_len = self._smooth(self.episode_lengths, 50)
                axes[1, 1].plot(self.timesteps, smoothed_len, color='#9B59B6', linewidth=2, label='Smoothed')
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        fig1.savefig(os.path.join(self.log_dir, 'plots', f'training_curves_ep{self.n_episodes}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # ===== 图2: 奖励分解 (堆叠面积图) =====
        if len(self.reward_components['distance']) > 100:
            fig2, ax = plt.subplots(figsize=(12, 6))
            
            # 准备数据
            n_samples = min(1000, len(self.reward_components['distance']))
            steps = np.arange(n_samples)
            
            # 堆叠数据
            components_data = []
            labels = []
            colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
            
            for i, (key, values) in enumerate(self.reward_components.items()):
                if len(values) >= n_samples:
                    components_data.append(values[-n_samples:])
                    labels.append(key.replace('_', ' ').title())
            
            if components_data:
                ax.stackplot(steps, *components_data, labels=labels, colors=colors[:len(components_data)], alpha=0.7)
                ax.set_xlabel('Steps (last 1000)')
                ax.set_ylabel('Reward Contribution')
                ax.set_title('Reward Components Breakdown')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                fig2.savefig(os.path.join(self.log_dir, 'plots', f'reward_breakdown_ep{self.n_episodes}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        # ===== 图3: 控制平滑度 =====
        if len(self.action_norms) > 100:
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
            
            # 3.1 Action Norm
            n_samples = min(500, len(self.action_norms))
            axes3[0].plot(self.action_norms[-n_samples:], alpha=0.6, color='#3498DB')
            axes3[0].set_xlabel('Steps (last 500)')
            axes3[0].set_ylabel('Action Norm')
            axes3[0].set_title('Action Magnitude')
            axes3[0].grid(alpha=0.3)
            
            # 3.2 Action Change (平滑度)
            if len(self.action_changes) > 100:
                n_samples = min(500, len(self.action_changes))
                axes3[1].plot(self.action_changes[-n_samples:], alpha=0.6, color='#E67E22')
                axes3[1].set_xlabel('Steps (last 500)')
                axes3[1].set_ylabel('Action Change')
                axes3[1].set_title('Action Smoothness (lower=smoother)')
                axes3[1].grid(alpha=0.3)
            
            plt.tight_layout()
            fig3.savefig(os.path.join(self.log_dir, 'plots', f'control_smoothness_ep{self.n_episodes}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig3)
        
        # ===== 图4: 分布直方图 =====
        if len(self.final_distances) > 50:
            fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
            fig4.suptitle('Performance Distributions', fontsize=14, fontweight='bold')
            
            # 4.1 Final Distance Distribution
            axes4[0, 0].hist(self.final_distances, bins=30, alpha=0.7, color='#E74C3C', edgecolor='black')
            axes4[0, 0].axvline(x=np.mean(self.final_distances), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.final_distances):.3f}m')
            axes4[0, 0].axvline(x=0.02, color='green', linestyle='--', linewidth=2, label='Success Threshold')
            axes4[0, 0].set_xlabel('Final Distance (m)')
            axes4[0, 0].set_ylabel('Frequency')
            axes4[0, 0].set_title('Final Distance Distribution')
            axes4[0, 0].legend()
            axes4[0, 0].grid(alpha=0.3, axis='y')
            
            # 4.2 Episode Reward Distribution
            axes4[0, 1].hist(self.episode_rewards, bins=30, alpha=0.7, color='#2E86AB', edgecolor='black')
            axes4[0, 1].axvline(x=np.mean(self.episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.1f}')
            axes4[0, 1].set_xlabel('Episode Reward')
            axes4[0, 1].set_ylabel('Frequency')
            axes4[0, 1].set_title('Episode Reward Distribution')
            axes4[0, 1].legend()
            axes4[0, 1].grid(alpha=0.3, axis='y')
            
            # 4.3 Episode Length Distribution
            axes4[1, 0].hist(self.episode_lengths, bins=30, alpha=0.7, color='#9B59B6', edgecolor='black')
            axes4[1, 0].axvline(x=np.mean(self.episode_lengths), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_lengths):.0f}')
            axes4[1, 0].set_xlabel('Episode Length (steps)')
            axes4[1, 0].set_ylabel('Frequency')
            axes4[1, 0].set_title('Episode Length Distribution')
            axes4[1, 0].legend()
            axes4[1, 0].grid(alpha=0.3, axis='y')
            
            # 4.4 Target Drift Distribution
            if len(self.target_drifts) > 10:
                axes4[1, 1].hist(self.target_drifts, bins=30, alpha=0.7, color='#27AE60', edgecolor='black')
                axes4[1, 1].axvline(x=np.mean(self.target_drifts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.target_drifts)*100:.1f}cm')
                axes4[1, 1].set_xlabel('Target Drift (m)')
                axes4[1, 1].set_ylabel('Frequency')
                axes4[1, 1].set_title('Target Drift Distribution')
                axes4[1, 1].legend()
                axes4[1, 1].grid(alpha=0.3, axis='y')
            
            plt.tight_layout()
            fig4.savefig(os.path.join(self.log_dir, 'plots', f'distributions_ep{self.n_episodes}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig4)
        
        print(f"✅ 图表已保存到: {os.path.join(self.log_dir, 'plots')}")
        print(f"{'='*60}\n")
    
    def _smooth(self, data, window=50):
        """滑动平均平滑"""
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')
    
    def _moving_average(self, data, window):
        """移动平均"""
        if len(data) < window:
            return data
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(np.mean(data[start:i+1]))
        return result
    
    def _on_training_end(self):
        """训练结束时的处理"""
        print("\n" + "="*60)
        print("训练完成，生成最终报告...")
        print("="*60)
        
        # 保存最终指标
        self._save_metrics()
        
        # 生成最终图表
        self._plot_metrics()
        
        # 生成统计报告
        self._generate_statistics_report()
        

    def _generate_statistics_report(self):
        """生成统计报告 - 修复版"""
        
        # ✅ 类型转换辅助函数（同上）
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        try:
            report = {
                'training_summary': {
                    'total_episodes': int(self.n_episodes),
                    'total_timesteps': int(self.num_timesteps),
                    'final_success_rate': float(np.mean(list(self.recent_successes))) if self.recent_successes else 0.0,
                    'final_mean_reward': float(np.mean(list(self.recent_rewards))) if self.recent_rewards else 0.0,
                    'final_mean_distance': float(np.mean(list(self.recent_distances))) if self.recent_distances else 0.0,
                },
                'overall_statistics': {
                    'mean_episode_reward': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
                    'std_episode_reward': float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
                    'mean_episode_length': float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
                    'overall_success_rate': float(np.mean(self.episode_successes)) if self.episode_successes else 0.0,
                    'mean_final_distance': float(np.mean(self.final_distances)) if self.final_distances else 0.0,
                },
                'control_statistics': {
                    'mean_action_norm': float(np.mean(self.action_norms)) if self.action_norms else 0.0,
                    'mean_action_change': float(np.mean(self.action_changes)) if self.action_changes else 0.0,
                }
            }
            
            # ✅ 确保所有数据可序列化
            report = convert_to_serializable(report)
            
            # 保存报告
            report_path = os.path.join(self.log_dir, 'training_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # 打印报告
            print("\n" + "="*60)
            print("训练统计报告")
            print("="*60)
            for category, stats in report.items():
                print(f"\n{category.upper().replace('_', ' ')}:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                    else:
                        print(f"  {key.replace('_', ' ').title()}: {value}")
            print("="*60 + "\n")
            
            print(f"✅ 完整报告已保存到: {report_path}")
            
        except Exception as e:
            print(f"⚠️ 生成统计报告失败: {e}")
            import traceback
            traceback.print_exc()