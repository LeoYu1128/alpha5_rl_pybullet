"""
Modern Algorithm Comparison System for Reinforcement Learning Research
Author: Based on train_v8.py
Purpose: Generate publication-ready comparison data and visualizations
"""

import os
import sys
import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import glob
from scipy import stats


# Import original training function
sys.path.append(os.path.dirname(__file__))
from train_v8 import train_alpha_reach, STAGE_CONFIGS, TRAINING_STAGE
import random
import torch

def set_global_seed(seed=42):
    """Set global random seed to ensure experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Global random seed set: {seed}")

class ModernRLComparison:
    """Modern RL Algorithm Comparison System with Publication-Quality Outputs"""
    
    def __init__(self, algorithms=['SAC', 'TQC', 'CrossQ'], 
                 timesteps=500000,
                 stage=None,
                 num_envs=1,
                 save_dir='comparison_results',
                 seed=42):
        
        self.algorithms = algorithms
        self.timesteps = timesteps
        self.stage = stage if stage else TRAINING_STAGE
        self.num_envs = num_envs
        self.seed = seed
        set_global_seed(seed)
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"comparison_{self.stage}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "videos"), exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"🔬 Modern RL Algorithm Comparison System")
        print(f"{'='*80}")
        print(f"Algorithms: {self.algorithms}")
        print(f"Training Steps: {self.timesteps:,}")
        print(f"Training Stage: {self.stage}")
        print(f"Results Directory: {self.save_dir}")
        print(f"{'='*80}\n")
        
        # Store all results
        self.results = {}
        
    def run_comparison(self):
        """Run complete comparison experiment"""
        print("\n" + "="*80)
        print("🚀 Starting Algorithm Comparison Experiment")
        print("="*80 + "\n")
        
        for algo in self.algorithms:
            print(f"\n{'='*80}")
            print(f"📊 Training Algorithm: {algo}")
            print(f"{'='*80}\n")
            
            try:
                # Train algorithm with visualization enabled
                algo_results = self._train_algorithm(algo)
                self.results[algo] = algo_results
                
                # Save intermediate results
                self._save_intermediate_results(algo, algo_results)
                
                print(f"\n✅ {algo} Training Complete")
                print(f"   Final Reward: {algo_results['metrics']['final_mean_reward']:.2f} ± {algo_results['metrics']['final_std_reward']:.2f}")
                print(f"   Success Rate: {algo_results['evaluation']['success_rate']*100:.1f}%")
                print(f"   Training Time: {algo_results['metrics']['training_time_hours']:.2f}h")
                
            except Exception as e:
                print(f"\n❌ {algo} Training Failed: {e}")
                import traceback
                traceback.print_exc()
                self.results[algo] = None
        
        # Generate comparison report
        self._generate_comparison_report()
        
        # Generate all visualizations
        self._generate_modern_visualizations()
        
        # Copy GIF videos to comparison folder
        self._collect_videos()
        
        print(f"\n{'='*80}")
        print(f"✅ Comparison Complete! Results saved to: {self.save_dir}")
        print(f"{'='*80}\n")
        
        return self.results
    
    def _train_algorithm(self, algorithm: str) -> Dict:
        """Train single algorithm and collect detailed data"""
        start_time = time.time()
        
        # Run training with auto_visualize=True to generate GIFs
        model, mean_reward, std_reward, exp_dir = train_alpha_reach(
            algorithm=algorithm,
            total_timesteps=self.timesteps,
            num_envs=self.num_envs,
            auto_visualize=True,  # ✅ Enable GIF generation
            stage=self.stage,
            seed=self.seed
        )
        
        training_time = time.time() - start_time
        
        # Collect detailed metrics
        metrics = self._collect_detailed_metrics(exp_dir, algorithm)
        metrics['training_time_hours'] = training_time / 3600
        metrics['final_mean_reward'] = float(mean_reward)
        metrics['final_std_reward'] = float(std_reward)
        
        # Perform final evaluation
        eval_results = self._evaluate_algorithm(model, exp_dir, algorithm)
        
        return {
            'algorithm': algorithm,
            'model': model,
            'experiment_dir': exp_dir,
            'metrics': metrics,
            'evaluation': eval_results
        }
    
    def _collect_detailed_metrics(self, exp_dir: str, algorithm: str) -> Dict:
        """Collect detailed metrics from training logs"""
        log_dir = os.path.join(exp_dir, "logs")
        monitor_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        
        # Default metrics
        default_metrics = {
            'final_100_mean': 0.0,
            'final_100_std': 0.0,
            'final_100_max': 0.0,
            'final_100_min': 0.0,
            'convergence_timesteps': self.timesteps,
            'convergence_episodes': 0,
            'sample_efficiency': self.timesteps,
            'reward_cv': 0.0,  # Changed from inf to 0.0
            'best_reward': 0.0,
            'final_episode_length_mean': 0.0,
            'final_episode_length_std': 0.0,
            'learning_curve': {'timesteps': [], 'rewards': [], 'smoothed_rewards': [], 'success_rates': []}
        }
        
        if not monitor_files:
            print(f"⚠️ Warning: No monitor files found for {algorithm}")
            return default_metrics
        
        # Read all monitor data
        all_data = []
        for log_file in monitor_files:
            try:
                df = pd.read_csv(log_file, comment='#')
                if 'r' in df.columns:
                    df['r'] = df['r'].astype(str).str.replace('--', '-', regex=False)
                    df['r'] = pd.to_numeric(df['r'], errors='coerce')
                if 'l' in df.columns:
                    df['l'] = pd.to_numeric(df['l'], errors='coerce')
                if 't' in df.columns:
                    df['t'] = pd.to_numeric(df['t'], errors='coerce')
                df = df.dropna()
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                print(f"⚠️ Failed to read {log_file}: {e}")
        
        if not all_data:
            print(f"⚠️ Warning: No valid data for {algorithm}")
            return default_metrics
        
        combined_df = pd.concat(all_data, ignore_index=True)
        # FIXED: Do not sort - keeps temporal order
        # combined_df = combined_df.sort_values('l')
        
        # FIXED: Calculate cumulative timesteps
        # Monitor CSV 'l' column is episode length, need cumsum
        cumulative_timesteps = combined_df['l'].cumsum()
        n_episodes = len(combined_df)
        if n_episodes == 0:
            return default_metrics
        
        metrics = {}
        
        # 1. Final performance (last 100 episodes or all if less)
        n_final = min(100, n_episodes)
        final_rewards = combined_df['r'].tail(n_final)
        metrics['final_100_mean'] = float(final_rewards.mean())
        metrics['final_100_std'] = float(final_rewards.std()) if len(final_rewards) > 1 else 0.0
        metrics['final_100_max'] = float(final_rewards.max())
        metrics['final_100_min'] = float(final_rewards.min())
        
        # 2. Convergence speed
        smoothed_rewards = combined_df['r'].rolling(window=min(50, n_episodes), min_periods=1).mean()
        target_reward = metrics['final_100_mean'] * 0.9
        convergence_mask = smoothed_rewards >= target_reward
        if convergence_mask.any():
            convergence_idx = convergence_mask.idxmax()
            metrics['convergence_timesteps'] = int(cumulative_timesteps.iloc[convergence_idx])
            metrics['convergence_episodes'] = int(convergence_idx)
        else:
            metrics['convergence_timesteps'] = int(self.timesteps)
            metrics['convergence_episodes'] = n_episodes
        
        # 3. Sample efficiency
        metrics['sample_efficiency'] = metrics['convergence_timesteps']
        
        # 4. Training stability (CV)
        if metrics['final_100_mean'] != 0 and abs(metrics['final_100_mean']) > 0.01:
            metrics['reward_cv'] = float(abs(metrics['final_100_std'] / metrics['final_100_mean']))
        else:
            metrics['reward_cv'] = 0.0  # Set to 0 instead of inf
        
        # 5. Best performance
        metrics['best_reward'] = float(combined_df['r'].max())
        
        # 6. Episode length statistics
        # FIXED: Use 'l' column (episode length), not 't' column (elapsed time)
        final_lengths = combined_df['l'].tail(n_final)
        metrics['final_episode_length_mean'] = float(final_lengths.mean())
        metrics['final_episode_length_std'] = float(final_lengths.std()) if len(final_lengths) > 1 else 0.0
        
        # 7. Learning curve data (with success rate estimation)
        # Estimate success rate from rewards (reward > threshold indicates success)
        success_threshold = 0.0  # Adjust based on your reward structure
        window_size = min(50, n_episodes)
        success_rates = []
        for i in range(len(combined_df)):
            start_idx = max(0, i - window_size + 1)
            window_rewards = combined_df['r'].iloc[start_idx:i+1]
            success_rate = (window_rewards > success_threshold).mean()
            success_rates.append(float(success_rate))
        
        metrics['learning_curve'] = {
            'timesteps': cumulative_timesteps.values.tolist(),
            'rewards': combined_df['r'].values.tolist(),
            'smoothed_rewards': smoothed_rewards.values.tolist(),
            'success_rates': success_rates
        }
        
        return metrics
    
    def _evaluate_algorithm(self, model, exp_dir: str, algorithm: str) -> Dict:
        """Evaluate trained model"""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3 import SAC
        from sb3_contrib import TQC, CrossQ
        
        print(f"\n📊 Evaluating {algorithm} Model...")
        
        # Load model and environment
        stage_config = STAGE_CONFIGS.get(self.stage, STAGE_CONFIGS['stage1'])
        env_config = {k: v for k, v in stage_config.items() 
                      if k not in ['description', 'recommended_timesteps']}
        
        from envs.rl_env_v7 import AlphaReachEnv
        
        def make_eval_env():
            env = AlphaReachEnv(render_mode=None, **env_config)
            return env
        
        eval_env = DummyVecEnv([make_eval_env])
        
        # Try loading VecNormalize parameters
        model_path = os.path.join(exp_dir, "models", f"{algorithm}_final")
        vecnorm_path = os.path.join(exp_dir, "models", f"{algorithm}_vecnormalize.pkl")
        
        if os.path.exists(vecnorm_path):
            eval_env = VecNormalize.load(vecnorm_path, eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        
        # Run evaluation
        n_eval_episodes = 50
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_distances = []
        
        for episode in range(n_eval_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                if done[0]:
                    break
            
            episode_rewards.append(float(episode_reward))
            episode_lengths.append(int(episode_length))
            
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            episode_successes.append(bool(info.get('success', False)))
            episode_distances.append(float(info.get('distance', 0)))
        
        eval_env.close()
        
        # Calculate statistics
        eval_results = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths)),
            'success_rate': float(np.mean(episode_successes)),
            'mean_distance': float(np.mean(episode_distances)),
            'std_distance': float(np.std(episode_distances)),
            'n_episodes': n_eval_episodes,
            'all_rewards': episode_rewards,
            'all_successes': episode_successes,
            'all_distances': episode_distances
        }
        
        print(f"   Success Rate: {eval_results['success_rate']*100:.1f}%")
        print(f"   Mean Distance: {eval_results['mean_distance']*100:.2f}cm")
        print(f"   Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        
        return eval_results
    
    def _save_intermediate_results(self, algorithm: str, results: Dict):
        """Save intermediate results as JSON"""
        save_dict = {
            'algorithm': algorithm,
            'metrics': results['metrics'].copy(),
            'evaluation': results['evaluation']
        }
        
        # Remove large learning curve data
        if 'learning_curve' in save_dict['metrics']:
            save_dict['metrics']['learning_curve_length'] = len(save_dict['metrics']['learning_curve']['timesteps'])
            del save_dict['metrics']['learning_curve']
        
        json_path = os.path.join(self.save_dir, "data", f"{algorithm}_results.json")
        with open(json_path, 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"   💾 Results saved to: {json_path}")
    
    def _collect_videos(self):
        """Collect GIF videos from experiment folders"""
        print("\n📹 Collecting demonstration videos...")
        
        for algo, result in self.results.items():
            if result is None:
                continue
            
            exp_dir = result['experiment_dir']
            video_source = os.path.join(exp_dir, "videos")
            
            if os.path.exists(video_source):
                gif_files = glob.glob(os.path.join(video_source, "*.gif"))
                
                for gif_file in gif_files:
                    dest_name = f"{algo}_{os.path.basename(gif_file)}"
                    dest_path = os.path.join(self.save_dir, "videos", dest_name)
                    
                    import shutil
                    shutil.copy2(gif_file, dest_path)
                    print(f"   ✅ Copied: {dest_name}")
        
        print(f"   📁 Videos saved to: {os.path.join(self.save_dir, 'videos')}")
    
    def _generate_comparison_report(self):
        """Generate text format comparison report"""
        report_path = os.path.join(self.save_dir, "comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Deep Reinforcement Learning Algorithm Comparison Report\n")
            f.write("Underwater Robotic Arm Reaching Task\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Experiment Configuration:\n")
            f.write(f"  Training Stage: {self.stage}\n")
            f.write(f"  Training Steps: {self.timesteps:,}\n")
            f.write(f"  Compared Algorithms: {', '.join(self.algorithms)}\n")
            f.write(f"  Experiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Detailed report for each algorithm
            for algo in self.algorithms:
                if self.results.get(algo) is None:
                    f.write(f"{algo}: Training Failed\n\n")
                    continue
                
                result = self.results[algo]
                metrics = result['metrics']
                evaluation = result['evaluation']
                
                f.write(f"{algo} Performance Report:\n")
                f.write("-"*80 + "\n")
                
                f.write(f"\n1. Training Performance:\n")
                f.write(f"   Training Time: {metrics.get('training_time_hours', 0):.2f} hours\n")
                f.write(f"   Final Reward (last 100 eps): {metrics.get('final_100_mean', 0):.2f} ± {metrics.get('final_100_std', 0):.2f}\n")
                f.write(f"   Best Reward: {metrics.get('best_reward', 0):.2f}\n")
                f.write(f"   Reward CV: {metrics.get('reward_cv', 0):.4f}\n")
                
                f.write(f"\n2. Convergence Performance:\n")
                f.write(f"   Convergence Steps: {metrics.get('convergence_timesteps', 0):,}\n")
                f.write(f"   Convergence Episodes: {metrics.get('convergence_episodes', 0)}\n")
                f.write(f"   Sample Efficiency: {metrics.get('sample_efficiency', 0):,} steps\n")
                
                f.write(f"\n3. Evaluation Performance ({evaluation.get('n_episodes', 0)} episodes):\n")
                f.write(f"   Success Rate: {evaluation.get('success_rate', 0)*100:.1f}%\n")
                f.write(f"   Mean Distance: {evaluation.get('mean_distance', 0)*100:.2f} ± {evaluation.get('std_distance', 0)*100:.2f} cm\n")
                f.write(f"   Mean Reward: {evaluation.get('mean_reward', 0):.2f} ± {evaluation.get('std_reward', 0):.2f}\n")
                f.write(f"   Mean Episode Length: {evaluation.get('mean_length', 0):.1f} ± {evaluation.get('std_length', 0):.1f}\n")
                
                f.write("\n" + "="*80 + "\n\n")
            
            # Algorithm rankings
            f.write("Algorithm Rankings:\n")
            f.write("="*80 + "\n\n")
            
            valid_results = {algo: res for algo, res in self.results.items() if res is not None}
            
            if valid_results:
                # By success rate
                success_ranking = sorted(valid_results.items(), 
                                        key=lambda x: x[1]['evaluation']['success_rate'], 
                                        reverse=True)
                f.write("By Success Rate:\n")
                for rank, (algo, res) in enumerate(success_ranking, 1):
                    f.write(f"  {rank}. {algo}: {res['evaluation']['success_rate']*100:.1f}%\n")
                
                # By mean distance
                distance_ranking = sorted(valid_results.items(),
                                         key=lambda x: x[1]['evaluation']['mean_distance'])
                f.write("\nBy Mean Final Distance (Precision):\n")
                for rank, (algo, res) in enumerate(distance_ranking, 1):
                    f.write(f"  {rank}. {algo}: {res['evaluation']['mean_distance']*100:.2f}cm\n")
                
                # By sample efficiency
                efficiency_ranking = sorted(valid_results.items(),
                                           key=lambda x: x[1]['metrics']['sample_efficiency'])
                f.write("\nBy Sample Efficiency (Convergence Speed):\n")
                for rank, (algo, res) in enumerate(efficiency_ranking, 1):
                    f.write(f"  {rank}. {algo}: {res['metrics']['sample_efficiency']:,} steps\n")
                
                # By training stability
                stability_ranking = sorted(valid_results.items(),
                                          key=lambda x: x[1]['metrics']['reward_cv'])
                f.write("\nBy Training Stability (CV, lower is better):\n")
                for rank, (algo, res) in enumerate(stability_ranking, 1):
                    cv = res['metrics']['reward_cv']
                    f.write(f"  {rank}. {algo}: {cv:.4f}\n")
                    # 🆕 Add: Curriculum Learning Progress Section
            f.write("\n" + "="*80 + "\n")
            f.write("Curriculum Learning Progression\n")
            f.write("="*80 + "\n\n")
            
            for algo in self.algorithms:
                if self.results.get(algo) is None:
                    continue
                
                f.write(f"\n{algo}:\n")
                f.write("-"*80 + "\n")
                
                result = self.results[algo]
                exp_dir = result.get('experiment_dir', '')
                
                if exp_dir:
                    curriculum_file = os.path.join(exp_dir, 'logs', 'curriculum_history.json')
                    
                    if os.path.exists(curriculum_file):
                        try:
                            with open(curriculum_file, 'r') as cf:
                                curriculum_data = json.load(cf)
                            
                            f.write(f"  Total Episodes: {curriculum_data.get('total_episodes', 'N/A')}\n")
                            f.write(f"  Total Successes: {curriculum_data.get('total_successes', 'N/A')}\n")
                            f.write(f"  Overall Success Rate: {curriculum_data.get('overall_success_rate', 0)*100:.1f}%\n")
                            f.write(f"  Total Stage Transitions: {curriculum_data.get('total_stage_transitions', 0)}\n")
                            f.write(f"  Final Stage Reached: {curriculum_data.get('final_stage', 'N/A')}\n")
                            
                            # Episodes per stage
                            episodes_per_stage = curriculum_data.get('episodes_per_stage', {})
                            if episodes_per_stage:
                                f.write(f"\n  Episodes per Stage:\n")
                                for stage in sorted(episodes_per_stage.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                                    count = episodes_per_stage[stage]
                                    f.write(f"    Stage {stage}: {count} episodes\n")
                            
                            # Stage transition timeline
                            transitions = curriculum_data.get('transitions', [])
                            if transitions:
                                f.write(f"\n  Stage Transition Timeline:\n")
                                for i, trans in enumerate(transitions, 1):
                                    f.write(f"\n    Transition {i}: Stage {trans.get('old_stage', '?')} → Stage {trans.get('new_stage', '?')}\n")
                                    f.write(f"      Timestep: {trans.get('timestep', 0):,}\n")
                                    f.write(f"      Episode: {trans.get('episode', 0)}\n")
                                    f.write(f"      Episodes in previous stage: {trans.get('episodes_in_previous_stage', 0)}\n")
                                    
                                    stage_config = trans.get('stage_config', {})
                                    if stage_config:
                                        drift = stage_config.get('drift_strength', 0)
                                        threshold = stage_config.get('success_threshold', 0)
                                        f.write(f"      Drift strength: {drift:.3f} m/s\n")
                                        f.write(f"      Success threshold: {threshold*100:.1f} cm\n")
                            
                        except Exception as e:
                            f.write(f"  ⚠️  Error loading curriculum data: {e}\n")
                    else:
                        f.write(f"  ⚠️  Curriculum history not found\n")
                        f.write(f"  Expected location: {curriculum_file}\n")
                else:
                    f.write(f"  ⚠️  Experiment directory not available\n")
            
            f.write("\n" + "="*80 + "\n")
        print(f"\n📄 Comparison report generated: {report_path}")
    
    def _generate_modern_visualizations(self):
        """Generate modern publication-quality visualizations"""
        print(f"\n📊 Generating publication-quality visualizations...")
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'axes.linewidth': 1.0,
            'lines.linewidth': 2.0,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white'
        })
        
        valid_results = {algo: res for algo, res in self.results.items() if res is not None}
        
        if not valid_results:
            print("❌ No valid results to visualize")
            return
        
        # 1. Learning curves (Episode Return)
        self._plot_learning_curves(valid_results)
        
        # 2. Success rate over training
        self._plot_success_rate_curves(valid_results)
        
        # 3. Final performance comparison
        self._plot_final_performance(valid_results)
        
        # 4. Sample efficiency comparison
        self._plot_sample_efficiency(valid_results)
        
        # 5. Statistical comparison (with significance testing)
        self._plot_statistical_comparison(valid_results)
        
        # 6. Comprehensive radar chart
        self._plot_comprehensive_radar(valid_results)
        
        print(f"✅ All visualizations saved to: {os.path.join(self.save_dir, 'plots')}")
    
    def _plot_learning_curves(self, results: Dict):
        """Plot episode return learning curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = {'SAC': '#1f77b4', 'TQC': '#d62728', 'CrossQ': '#ff7f0e'}
        
        has_data = False
        
        # Collect all algorithm data for Y-axis range
        all_smoothed_rewards = []
        
        for algo, result in results.items():
            lc = result['metrics'].get('learning_curve', {})
            if not lc or not lc.get('timesteps'):
                print(f"   ⚠️  {algo}: No learning curve data")
                continue
            
            has_data = True
            color = colors.get(algo, '#000000')
            
            timesteps = np.array(lc['timesteps'])
            rewards = np.array(lc['rewards'])
            smoothed = np.array(lc['smoothed_rewards'])
            
            # 🆕 Add debug info
            print(f"\n   📊 {algo} Learning Curve:")
            print(f"      Points: {len(timesteps)}, Timesteps: [{timesteps[0]:.0f}, {timesteps[-1]:.0f}]")
            print(f"      Rewards: [{rewards.min():.1f}, {rewards.max():.1f}], mean={rewards.mean():.1f}")
            print(f"      Smoothed: [{smoothed.min():.1f}, {smoothed.max():.1f}]")
            
            all_smoothed_rewards.extend(smoothed)
            
            # Full curve
            ax1.plot(timesteps, rewards, alpha=0.2, color=color, linewidth=0.8)
            ax1.plot(timesteps, smoothed, label=algo, color=color, linewidth=2.0)
            
            # Last 20%
            cutoff = int(len(timesteps) * 0.8)
            if cutoff < len(timesteps):
                ax2.plot(timesteps[cutoff:], smoothed[cutoff:], 
                        label=algo, color=color, linewidth=2.0)
        
        # 🆕 Add: Plot curriculum stage markers (using priority algorithm)
        curriculum_drawn = False
        for algo in ['CrossQ', 'TQC', 'SAC']:
            if algo not in results:
                continue
            
            result = results[algo]
            exp_dir = result.get('experiment_dir', '')
            if not exp_dir:
                continue
            
            curriculum_file = os.path.join(exp_dir, 'logs', 'curriculum_history.json')
            
            if os.path.exists(curriculum_file) and not curriculum_drawn:
                try:
                    with open(curriculum_file, 'r') as f:
                        curriculum_data = json.load(f)
                    
                    valid_transitions = [t for t in curriculum_data['transitions'] 
                                    if t['old_stage'] is not None]
                    
                    if len(valid_transitions) > 0:
                        print(f"\n   ℹ️  Using curriculum stages from {algo} ({len(valid_transitions)} transitions)")
                        
                        for transition in valid_transitions:
                            timestep = transition['timestep']
                            new_stage = transition['new_stage']
                            
                            # Draw vertical dashed line
                            ax1.axvline(x=timestep, color='gray', linestyle='--', 
                                    alpha=0.4, linewidth=1.2, zorder=1)
                            
                            # Add stage label (set y_pos later)
                            # Use fixed value temporarily, will adjust later
                        
                        curriculum_drawn = True
                        stored_curriculum = curriculum_data  # Save data for later use
                        stored_algo = algo
                        
                except Exception as e:
                    print(f"   ⚠️  Could not load curriculum for {algo}: {e}")
        
        if not has_data:
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'No Learning Curve Data\n(Increase training steps)', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=14, color='red', weight='bold')
        else:
            # Set Y-axis range
            if all_smoothed_rewards:
                y_min = min(all_smoothed_rewards)
                y_max = max(all_smoothed_rewards)
                y_range = y_max - y_min if y_max != y_min else 100
                ax1.set_ylim(y_min - y_range*0.1, y_max + y_range*0.2)
            
            # Now add curriculum labels (Y-axis range is determined)
            if curriculum_drawn:
                valid_transitions = [t for t in stored_curriculum['transitions'] 
                                if t['old_stage'] is not None]
                y_pos = ax1.get_ylim()[1] * 0.95
                
                for transition in valid_transitions:
                    timestep = transition['timestep']
                    new_stage = transition['new_stage']
                    
                    ax1.text(timestep, y_pos, f'S{new_stage}', 
                            rotation=0, fontsize=9, ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='yellow', alpha=0.5, 
                                    edgecolor='gray', linewidth=0.8),
                            zorder=10)
                
                print(f"   ✓ Added {len(valid_transitions)} curriculum stage markers from {stored_algo}")
            
            ax1.set_xlabel('Training Steps', fontsize=13)
            ax1.set_ylabel('Episode Return', fontsize=13)
            ax1.set_title('Learning Curves with Curriculum Stages (Full Training)', fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            ax2.set_xlabel('Training Steps', fontsize=13)
            ax2.set_ylabel('Episode Return', fontsize=13)
            ax2.set_title('Learning Curves (Final 20%)', fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', '1_learning_curves.png'))
        plt.close()
        print("   ✅ Generated: 1_learning_curves.png")
    
    def _plot_success_rate_curves(self, results: Dict):
        """Plot success rate over training"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = {'SAC': '#1f77b4', 'TQC': '#d62728', 'CrossQ': '#ff7f0e'}
        
        has_data = False
        
        for algo, result in results.items():
            lc = result['metrics'].get('learning_curve', {})
            if not lc or not lc.get('success_rates'):
                continue
            
            has_data = True
            color = colors.get(algo, '#000000')
            
            timesteps = np.array(lc['timesteps'])
            success_rates = np.array(lc['success_rates']) * 100  # Convert to percentage
            
            # Smooth success rate curve
            window = min(100, len(success_rates) // 10)
            if window > 0:
                smoothed_sr = pd.Series(success_rates).rolling(window=window, min_periods=1).mean()
                ax.plot(timesteps, smoothed_sr, label=algo, color=color, linewidth=2.5)
        
        if not has_data:
            ax.text(0.5, 0.5, 'No Success Rate Data Available\n(Increase training steps)', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='red', weight='bold')
        else:
            ax.set_xlabel('Training Steps', fontsize=13)
            ax.set_ylabel('Success Rate (%)', fontsize=13)
            ax.set_title('Success Rate During Training', fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', '2_success_rate_curves.png'))
        plt.close()
        print("   ✅ Generated: 2_success_rate_curves.png")
    
    def _plot_final_performance(self, results: Dict):
        """Plot final performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        algorithms = list(results.keys())
        colors = [{'SAC': '#1f77b4', 'TQC': '#d62728', 'CrossQ': '#ff7f0e'}.get(a, '#777') 
                    for a in algorithms]
        
        # 1. Success Rate
        success_rates = [results[a]['evaluation']['success_rate'] * 100 for a in algorithms]
        bars1 = ax1.bar(algorithms, success_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Success Rate (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Task Success Rate', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(success_rates) * 1.2 if success_rates else 100)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Mean Final Distance (Precision)
        mean_distances = [results[a]['evaluation']['mean_distance'] * 100 for a in algorithms]
        std_distances = [results[a]['evaluation']['std_distance'] * 100 for a in algorithms]
        bars2 = ax2.bar(algorithms, mean_distances, yerr=std_distances,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                       capsize=10, error_kw={'linewidth': 2})
        for bar, mean, std in zip(bars2, mean_distances, std_distances):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Mean Distance (cm)', fontsize=13, fontweight='bold')
        ax2.set_title('Final Distance to Target (Precision)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(mean_distances) * 1.3 if mean_distances else 10)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Episode Return
        mean_rewards = [results[a]['evaluation']['mean_reward'] for a in algorithms]
        std_rewards = [results[a]['evaluation']['std_reward'] for a in algorithms]
        bars3 = ax3.bar(algorithms, mean_rewards, yerr=std_rewards,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                       capsize=10, error_kw={'linewidth': 2})
        for bar, mean, std in zip(bars3, mean_rewards, std_rewards):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax3.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
        ax3.set_title('Average Episode Return', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Training Stability (CV)
        cv_values = [results[a]['metrics']['reward_cv'] for a in algorithms]
        bars4 = ax4.bar(algorithms, cv_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        for bar, cv in zip(bars4, cv_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{cv:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Coefficient of Variation', fontsize=13, fontweight='bold')
        ax4.set_title('Training Stability (Lower is Better)', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, max(cv_values) * 1.2 if cv_values and max(cv_values) > 0 else 1)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', '3_final_performance.png'))
        plt.close()
        print("   ✅ Generated: 3_final_performance.png")
    
    def _plot_sample_efficiency(self, results: Dict):
        """Plot sample efficiency comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(results.keys())
        colors = [{'SAC': '#1f77b4', 'TQC': '#d62728', 'CrossQ': '#ff7f0e'}.get(a, '#777') 
                    for a in algorithms]
        
        convergence_steps = [results[a]['metrics']['convergence_timesteps'] / 1000 for a in algorithms]
        
        bars = ax.barh(algorithms, convergence_steps, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        for bar, steps in zip(bars, convergence_steps):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{steps:.0f}K', ha='left', va='center', fontweight='bold', fontsize=11, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Steps to Convergence (×1000)', fontsize=13, fontweight='bold')
        ax.set_title('Sample Efficiency (Lower is Better)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', '4_sample_efficiency.png'))
        plt.close()
        print("   ✅ Generated: 4_sample_efficiency.png")
    
    def _plot_statistical_comparison(self, results: Dict):
        """Plot statistical comparison with significance testing"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = {'SAC': '#1f77b4', 'TQC': '#d62728', 'CrossQ': '#ff7f0e'}
        
        algorithms = list(results.keys())
        
        # 1. Reward distributions
        reward_data = [results[a]['evaluation']['all_rewards'] for a in algorithms]
        bp1 = ax1.boxplot(reward_data, labels=algorithms, patch_artist=True,
                         widths=0.6, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        for patch, algo in zip(bp1['boxes'], algorithms):
            patch.set_facecolor(colors.get(algo, '#777'))
            patch.set_alpha(0.7)
            patch.set_linewidth(1.5)
        
        ax1.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
        ax1.set_title('Episode Return Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Distance distributions
        distance_data = [np.array(results[a]['evaluation']['all_distances']) * 100 for a in algorithms]
        bp2 = ax2.boxplot(distance_data, labels=algorithms, patch_artist=True,
                         widths=0.6, showmeans=True,
                         meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        for patch, algo in zip(bp2['boxes'], algorithms):
            patch.set_facecolor(colors.get(algo, '#777'))
            patch.set_alpha(0.7)
            patch.set_linewidth(1.5)
        
        ax2.set_ylabel('Final Distance (cm)', fontsize=13, fontweight='bold')
        ax2.set_title('Final Distance Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', '5_statistical_comparison.png'))
        plt.close()
        print("   ✅ Generated: 5_statistical_comparison.png")
    
    def _plot_comprehensive_radar(self, results: Dict):
        """Plot comprehensive performance radar chart"""
        from math import pi
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = {'SAC': '#1f77b4', 'TQC': '#d62728', 'CrossQ': '#ff7f0e'}
        
        # Define evaluation dimensions
        categories = ['Success\nRate', 'Precision', 'Sample\nEfficiency', 
                     'Stability', 'Final\nReturn']
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
        ax.grid(True)
        
        for algo, result in results.items():
            eval_res = result['evaluation']
            metrics = result['metrics']
            
            # Normalize metrics to 0-1 scale (higher is better)
            success_rate = float(eval_res.get('success_rate', 0))
            
            mean_dist = float(eval_res.get('mean_distance', 1.0))
            precision = max(0, min(1, 1 - mean_dist / 0.1))
            
            sample_eff = float(metrics.get('sample_efficiency', self.timesteps))
            efficiency = max(0, min(1, 1 - sample_eff / self.timesteps))
            
            cv = float(metrics.get('reward_cv', 1.0))
            stability = max(0, min(1, 1 - cv)) if cv < 1 else 0
            
            final_mean = float(metrics.get('final_100_mean', -10))
            performance = max(0, min(1, (final_mean + 10) / 20))
            
            values = [success_rate, precision, efficiency, stability, performance]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2.5, label=algo, 
                   color=colors.get(algo, '#000000'))
            ax.fill(angles, values, alpha=0.15, color=colors.get(algo, '#000000'))
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        plt.title('Comprehensive Performance Comparison\n(All Metrics Normalized to 0-1)', 
                 size=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'plots', '6_comprehensive_radar.png'))
        plt.close()
        print("   ✅ Generated: 6_comprehensive_radar.png")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Modern RL Algorithm Comparison System')
    parser.add_argument('--algorithms', nargs='+', 
                       default=['SAC', 'TQC', 'CrossQ'],
                       help='List of algorithms to compare')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Training steps per algorithm')
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3', 'stage4'],
                       help='Training stage')
    parser.add_argument('--num_envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--save_dir', type=str, default='comparison_results',
                       help='Results save directory')
    parser.add_argument('--seed', type=int, default=42,  # ✅ Add this line
                   help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create comparison system
    comparison = ModernRLComparison(
        algorithms=args.algorithms,
        timesteps=args.timesteps,
        stage=args.stage,
        num_envs=args.num_envs,
        save_dir=args.save_dir,
        seed=args.seed
    )
    
    # Run comparison experiment
    results = comparison.run_comparison()
    
    print("\n" + "="*80)
    print("🎉 Experiment Complete! View Results:")
    print(f"   📁 All Files: {comparison.save_dir}")
    print(f"   📄 Text Report: {os.path.join(comparison.save_dir, 'comparison_report.txt')}")
    print(f"   📊 Plots Directory: {os.path.join(comparison.save_dir, 'plots')}")
    print(f"   📹 Videos Directory: {os.path.join(comparison.save_dir, 'videos')}")
    print("="*80)


if __name__ == "__main__":
    main()