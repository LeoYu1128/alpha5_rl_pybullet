import os
import numpy as np
import json
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import threading

class ComprehensiveMonitor(BaseCallback):
    """
    ✅ Multiprocess Safe Version - Avoids Training Crashes
    
    Core Improvements:
    1. Delayed Plotting: Only plot at training end to avoid runtime I/O conflicts
    2. Thread Lock: Protects file writing
    3. Minimize matplotlib Usage: Only record data, delay plotting
    """
    
    def __init__(self, log_dir, verbose=1, window_size=100):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.window_size = window_size
        
        os.makedirs(os.path.join(log_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
        
        # ✅ Add thread lock to protect file writing
        self._file_lock = threading.Lock()
        
        # Core training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.timesteps = []
        self.final_distances = []
        self.target_drifts = []
        
        # Sliding window statistics
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_successes = deque(maxlen=window_size)
        self.recent_distances = deque(maxlen=window_size)
        
        self.n_episodes = 0
        
    def _on_step(self) -> bool:
        """✅ Lightweight Version: Only record data, no plotting"""
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                self.n_episodes += 1
                
                # Extract metrics
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                success = info.get('is_success', False)
                final_distance = info.get('distance', 0)
                
                # Record
                self.episode_rewards.append(float(episode_reward))
                self.episode_lengths.append(int(episode_length))
                self.episode_successes.append(1.0 if success else 0.0)
                self.final_distances.append(float(final_distance))
                self.timesteps.append(self.num_timesteps)
                
                # Sliding window
                self.recent_rewards.append(float(episode_reward))
                self.recent_successes.append(1.0 if success else 0.0)
                self.recent_distances.append(float(final_distance))
                
                # Task specific
                if 'target_drift' in info:
                    self.target_drifts.append(float(info['target_drift']))
                
                # ✅ Only save data, no plotting (avoid conflicts)
                if self.n_episodes % 100 == 0:
                    self._save_metrics_safe()
                
                # Print progress
                if self.verbose > 0 and self.n_episodes % 10 == 0:
                    mean_reward = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0
                    mean_success = np.mean(list(self.recent_successes)) if self.recent_successes else 0
                    mean_distance = np.mean(list(self.recent_distances)) if self.recent_distances else 0
                    print(f"Ep {self.n_episodes:4d} | "
                          f"R: {mean_reward:7.1f} | "
                          f"SR: {mean_success*100:5.1f}% | "
                          f"Dist: {mean_distance*100:5.1f}cm")
        
        return True
    
    def _save_metrics_safe(self):
        """✅ Thread-safe saving"""
        with self._file_lock:
            try:
                metrics = {
                    'timesteps': [int(x) for x in self.timesteps],
                    'episode_rewards': [float(x) for x in self.episode_rewards],
                    'episode_lengths': [int(x) for x in self.episode_lengths],
                    'episode_successes': [float(x) for x in self.episode_successes],
                    'final_distances': [float(x) for x in self.final_distances],
                    'target_drifts': [float(x) for x in self.target_drifts],
                    'recent_stats': {
                        'mean_reward': float(np.mean(list(self.recent_rewards))) if self.recent_rewards else 0.0,
                        'mean_success_rate': float(np.mean(list(self.recent_successes))) if self.recent_successes else 0.0,
                        'mean_distance': float(np.mean(list(self.recent_distances))) if self.recent_distances else 0.0
                    }
                }
                
                filepath = os.path.join(self.log_dir, 'metrics', f'metrics_ep{self.n_episodes}.json')
                with open(filepath, 'w') as f:
                    json.dump(metrics, f, indent=2)
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠️ Failed to save metrics: {e}")
    
    def _on_training_end(self):
        """✅ Only plot when training ends (safe!)"""
        print("\n" + "="*60)
        print("Training complete, generating final charts and report...")
        print("="*60)
        
        # Save final metrics
        self._save_metrics_safe()
        
        # Generate report
        self._generate_final_report()
        
        # ✅ Only plot after training is completely finished
        self._plot_all_metrics()
    
    def _generate_final_report(self):
        """Generate final statistics report"""
        if len(self.episode_rewards) == 0:
            return
        
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
                    'mean_episode_reward': float(np.mean(self.episode_rewards)),
                    'std_episode_reward': float(np.std(self.episode_rewards)),
                    'best_episode_reward': float(np.max(self.episode_rewards)),
                    'mean_episode_length': float(np.mean(self.episode_lengths)),
                    'overall_success_rate': float(np.mean(self.episode_successes)),
                    'mean_final_distance': float(np.mean(self.final_distances)),
                    'best_final_distance': float(np.min(self.final_distances)) if self.final_distances else 0.0,
                }
            }
            
            report_path = os.path.join(self.log_dir, 'training_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print("\n" + "="*60)
            print("📊 Training Statistics Report")
            print("="*60)
            print(f"\nFinal Performance (last {self.window_size} episodes):")
            print(f"  Success Rate: {report['training_summary']['final_success_rate']*100:.1f}%")
            print(f"  Mean Reward: {report['training_summary']['final_mean_reward']:.2f}")
            print(f"  Mean Distance: {report['training_summary']['final_mean_distance']*100:.1f}cm")
            
            print(f"\nOverall Statistics:")
            print(f"  Total Episodes: {report['training_summary']['total_episodes']}")
            print(f"  Total Timesteps: {report['training_summary']['total_timesteps']}")
            print(f"  Overall Success Rate: {report['overall_statistics']['overall_success_rate']*100:.1f}%")
            print(f"  Best Reward: {report['overall_statistics']['best_episode_reward']:.2f}")
            print(f"  Best Distance: {report['overall_statistics']['best_final_distance']*100:.1f}cm")
            print("="*60 + "\n")
            
            print(f"✅ Full report saved: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to generate report: {e}")
    
    def _plot_all_metrics(self):
        """✅ Plot all charts at once after training ends"""
        if len(self.episode_rewards) < 10:
            print("⚠️ Insufficient data, skipping plotting")
            return
        
        try:
            # ✅ Delayed matplotlib import, only use when plotting
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            print("Generating training charts...")
            
            sns.set_style("whitegrid")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Training Results ({self.n_episodes} Episodes)', fontsize=14, fontweight='bold')
            
            # 1. Episode Reward
            axes[0, 0].plot(self.timesteps, self.episode_rewards, alpha=0.3, color='#2E86AB', linewidth=0.5)
            if len(self.episode_rewards) >= 50:
                smoothed = self._smooth(self.episode_rewards, 50)
                axes[0, 0].plot(self.timesteps, smoothed, color='#2E86AB', linewidth=2, label='Smoothed (50ep)')
            axes[0, 0].set_title('Episode Reward')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # 2. Success Rate
            if len(self.episode_successes) >= 10:
                window = min(100, len(self.episode_successes))
                success_rate = self._moving_average(self.episode_successes, window)
                axes[0, 1].plot(self.timesteps, success_rate, color='#27AE60', linewidth=2)
                axes[0, 1].fill_between(self.timesteps, 0, success_rate, alpha=0.3, color='#27AE60')
                axes[0, 1].set_title(f'Success Rate (rolling {window})')
                axes[0, 1].set_xlabel('Timesteps')
                axes[0, 1].set_ylabel('Success Rate')
                axes[0, 1].set_ylim([0, 1.05])
                axes[0, 1].grid(alpha=0.3)
            
            # 3. Final Distance
            axes[1, 0].plot(self.timesteps, self.final_distances, alpha=0.3, color='#E74C3C', linewidth=0.5)
            if len(self.final_distances) >= 50:
                smoothed = self._smooth(self.final_distances, 50)
                axes[1, 0].plot(self.timesteps, smoothed, color='#E74C3C', linewidth=2, label='Smoothed (50ep)')
            axes[1, 0].axhline(y=0.02, color='green', linestyle='--', label='Success Threshold (2cm)', linewidth=1.5)
            axes[1, 0].set_title('Final Distance to Target')
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Distance (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
            
            # 4. Episode Length
            axes[1, 1].plot(self.timesteps, self.episode_lengths, alpha=0.3, color='#9B59B6', linewidth=0.5)
            if len(self.episode_lengths) >= 50:
                smoothed = self._smooth(self.episode_lengths, 50)
                axes[1, 1].plot(self.timesteps, smoothed, color='#9B59B6', linewidth=2, label='Smoothed (50ep)')
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(self.log_dir, 'plots', 'final_training_results.png')
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"✅ Final training chart saved: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ Failed to generate charts: {e}")
            import traceback
            traceback.print_exc()
    
    def _smooth(self, data, window=50):
        """Sliding average smoothing"""
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')
    
    def _moving_average(self, data, window):
        """Moving average"""
        if len(data) < window:
            return data
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(np.mean(data[start:i+1]))
        return result