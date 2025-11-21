import os
import numpy as np
import json
import sys
from datetime import datetime
import glob

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

default_mpl_dir = os.path.join(os.path.dirname(__file__), '.mplconfig')
os.environ.setdefault('MPLCONFIGDIR', default_mpl_dir)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt
import pybullet as p
from PIL import Image
from pathlib import Path

from stable_baselines3 import SAC
from sb3_contrib import TQC, CrossQ

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import time
import random
import torch
from curriculum_callback import CurriculumMonitorCallback  # New addition
from monitor_callbacks import ComprehensiveMonitor

# ✅ Modified import: Use progressive environment
# from envs.rl_env_v13 import AlphaReachEnv
# from envs.rl_env_v5_success_copy import AlphaReachEnv
from envs.rl_env_v7 import AlphaReachEnv

# ============================================================================
# 🎯 Progressive Training Configuration - Only need to modify here!
# ============================================================================

# ===== Select Training Stage =====
# Change this to switch between different training stages!
TRAINING_STAGE = 'stage4'  # 👈 Change here! Options: 'stage1', 'stage2', 'stage3', 'stage4'

# ===== Stage Configuration Description =====
# stage1: Basic version - Static target, fixed physics parameters (Recommended 500K steps)
# stage2: Domain randomization - Random physics parameters, improved robustness (Recommended 500K steps)
# stage3: Target drift - Target slowly drifts (Recommended 500K steps)
# stage4: Curriculum learning - Progressive difficulty increase, final precision 5mm (Recommended 1M steps)

STAGE_CONFIGS = {
    'stage1': {
        'enable_target_drift': True,
        'enable_domain_randomization': True,
        'enable_curriculum': True,
        'enable_sensor_noise': True,      # ← New addition
        # 'enable_control_delay': False,     # ← New addition
        'description': 'Basic version - Static target',
        'recommended_timesteps': 500000
    },
    'stage2': {
        'enable_target_drift': True,
        'enable_domain_randomization': True,
        'enable_curriculum': True,
        # 'enable_sensor_noise': True,      # ← New addition
        # 'enable_control_delay': False,     # ← New addition
        'description': 'Domain randomization - More robust',
        'recommended_timesteps': 500000
    },
    'stage3': {
        'enable_target_drift': True,
        'enable_domain_randomization': True,
        'enable_curriculum': True,
        'enable_sensor_noise': True,      # ← New addition
        # 'enable_control_delay': True,     # ← New addition
        'description': 'Target drift - More realistic',
        'recommended_timesteps': 500000
    },
    'stage4': {
        'enable_target_drift': True,
        'enable_domain_randomization': True,
        'enable_curriculum': True,
        'enable_sensor_noise': True,      # ← New addition
        # 'enable_control_delay': True,     # ← New addition
        'description': 'Curriculum learning - Final version',
        'recommended_timesteps': 1000000
    }
}
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seed set: {seed}")
# ============================================================================


def create_experiment_folder(algorithm, timesteps, stage=None):
    """Create experiment folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_suffix = f"_{stage}" if stage else ""
    exp_name = f"{algorithm}{stage_suffix}_{timesteps}steps_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)

    return exp_dir, exp_name


def save_experiment_config(exp_dir, algorithm, config, timesteps, stage_config=None):
    """Save experiment configuration"""
    exp_config = {
        "algorithm": algorithm,
        "timesteps": timesteps,
        "timestamp": datetime.now().isoformat(),
        "config": config
    }
    if stage_config:
        exp_config["stage_config"] = stage_config
    
    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(exp_config, f, indent=2)


def plot_training_curves_enhanced(exp_dir, log_dir):
    """Enhanced training curve plotting (fixes double-dash issue)"""
    try:
        import pandas as pd
        
        log_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        if not log_files:
            print("⚠️ Warning: No monitor.csv files found")
            return False
        
        print(f"📊 Found {len(log_files)} monitor files, starting to plot...")
        
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'lines.linewidth': 2,
            'grid.alpha': 0.3
        })

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress - Underwater Alpha Arm (Progressive Training)', fontsize=18, fontweight='bold')

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        all_data = []
        for idx, log_file in enumerate(log_files):
            try:
                df = pd.read_csv(log_file, comment='#')
                
                # ========== Key Fix: Clean data ==========
                if 'r' in df.columns:
                    # Fix double-dash issue
                    df['r'] = df['r'].astype(str).str.replace('--', '-', regex=False)
                    df['r'] = pd.to_numeric(df['r'], errors='coerce')
                
                if 'l' in df.columns:
                    df['l'] = pd.to_numeric(df['l'], errors='coerce')
                
                if 't' in df.columns:
                    df['t'] = pd.to_numeric(df['t'], errors='coerce')
                
                # Drop rows that couldn't be converted
                df = df.dropna()
                # =======================================
                
                if not df.empty:
                    all_data.append(df)
                    color = colors[idx % len(colors)]
                    
                    if 'r' in df.columns and 'l' in df.columns:
                        raw_rewards = df['r'].values
                        timesteps = df['l'].values
                        smoothed_rewards = pd.Series(raw_rewards).rolling(window=50, min_periods=1).mean()
                        
                        axes[0, 0].plot(timesteps, raw_rewards, alpha=0.15, color=color, linewidth=0.5)
                        axes[0, 0].plot(timesteps, smoothed_rewards, linewidth=2.5, color=color,
                                       label=f'Env {idx+1}' if len(log_files) > 1 else 'Episode Reward')
                        
                        axes[0, 0].set_title('📈 Episode Reward', fontweight='bold', pad=10)
                        axes[0, 0].set_xlabel('Timesteps')
                        axes[0, 0].set_ylabel('Reward')
                        axes[0, 0].grid(True, alpha=0.3)
                        axes[0, 0].legend()
                    
                    if 't' in df.columns and 'l' in df.columns:
                        raw_lengths = df['t'].values
                        smoothed_lengths = pd.Series(raw_lengths).rolling(window=50, min_periods=1).mean()
                        
                        axes[0, 1].plot(timesteps, raw_lengths, alpha=0.15, color=color, linewidth=0.5)
                        axes[0, 1].plot(timesteps, smoothed_lengths, linewidth=2.5, color=color,
                                       label=f'Env {idx+1}' if len(log_files) > 1 else 'Episode Length')
                        
                        axes[0, 1].set_title('📏 Episode Length', fontweight='bold', pad=10)
                        axes[0, 1].set_xlabel('Timesteps')
                        axes[0, 1].set_ylabel('Steps')
                        axes[0, 1].grid(True, alpha=0.3)
                        axes[0, 1].legend()
                        
            except Exception as e:
                print(f"⚠️ Failed to read {log_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('l')
            
            if 'r' in combined_df.columns:
                window_rewards = combined_df['r'].rolling(window=100, min_periods=1).mean()
                axes[1, 0].plot(combined_df['l'], window_rewards, color='#27AE60', linewidth=3)
                axes[1, 0].fill_between(combined_df['l'], 
                                       combined_df['r'].rolling(100, min_periods=1).min(),
                                       combined_df['r'].rolling(100, min_periods=1).max(),
                                       alpha=0.2, color='#27AE60')
                axes[1, 0].set_title('📊 Rolling Average (window=100)', fontweight='bold', pad=10)
                axes[1, 0].set_xlabel('Timesteps')
                axes[1, 0].set_ylabel('Reward')
                axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].axis('off')
            
            summary_text = f"""
            📋 Training Summary

            Total Episodes: {len(combined_df)}
            Total Timesteps: {int(combined_df['l'].max()):,}

            Final 100 Episodes:
            Mean Reward: {float(combined_df['r'].tail(100).mean()):.2f}
            Std Reward:  {float(combined_df['r'].tail(100).std()):.2f}
            Max Reward:  {float(combined_df['r'].tail(100).max()):.2f}

            Mean Length: {float(combined_df['t'].tail(100).mean()):.1f} steps
            Best Reward: {float(combined_df['r'].max()):.2f}
            """
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                          fontsize=12, verticalalignment='center', family='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        plot_path = os.path.join(exp_dir, "plots", "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Training curves saved: {plot_path}")
        return True
        
    except ImportError:
        print("❌ Need to install pandas: pip install pandas")
        return False
    except Exception as e:
        print(f"❌ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_gif_from_model(model, env, exp_dir, episode_num=1, max_steps=400, fps=30):
    """Generate GIF from trained model - Enhanced information display version"""
    frames = []
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    episode_reward = 0
    episode_length = 0
    success_achieved = False
    
    print(f"🎬 Recording episode {episode_num}...")
    
    for step in range(max_steps):
        if hasattr(env, 'envs'):
            raw_env = env.envs[0] if hasattr(env.envs[0], '_get_end_effector_position') else env.envs[0].env
        elif hasattr(env, 'venv'):
            raw_env = env.venv.envs[0]
        else:
            raw_env = env
        
        if hasattr(raw_env, 'env'):
            raw_env = raw_env.env
        
        try:
            width, height = 640, 480
            
            ee_pos = raw_env._get_end_effector_position()
            target_pos = raw_env.target_position
            initial_target = raw_env.initial_target_position if hasattr(raw_env, 'initial_target_position') else target_pos
            
            distance_to_target = np.linalg.norm(ee_pos - target_pos)
            target_drift = np.linalg.norm(target_pos - initial_target)
            
            # Debug: Print environment info on first frame
            if step == 0:
                print(f"  Environment unwrap success:")
                print(f"     - Type: {type(raw_env).__name__}")
                print(f"     - Has curriculum_stage: {hasattr(raw_env, 'curriculum_stage')}")
                print(f"     - Has curriculum_level: {hasattr(raw_env, 'curriculum_level')}")
                if hasattr(raw_env, 'curriculum_stage'):
                    print(f"     - Curriculum stage: {raw_env.curriculum_stage}")
                if hasattr(raw_env, 'curriculum_level'):
                    print(f"     - Curriculum level: {raw_env.curriculum_level}")

            # Use curriculum_stage, fallback to curriculum_level
            curriculum_stage = getattr(raw_env, 'curriculum_stage', 
                                    getattr(raw_env, 'curriculum_level', 0))
            success_threshold = raw_env.success_threshold if hasattr(raw_env, 'success_threshold') else 0.08
            
            camera_target = [
                (ee_pos[0] + target_pos[0]) / 2,
                (ee_pos[1] + target_pos[1]) / 2,
                (ee_pos[2] + target_pos[2]) / 2
            ]
            
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=camera_target,
                distance=1.5, yaw=45, pitch=-30, roll=0,
                upAxisIndex=2,
                physicsClientId=raw_env.physics_client
            )
            
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width/height, nearVal=0.1, farVal=100.0,
                physicsClientId=raw_env.physics_client
            )
            
            img_arr = p.getCameraImage(
                width=width, height=height,
                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=raw_env.physics_client
            )
            
            rgb_array = np.array(img_arr[2], dtype=np.uint8)
            if rgb_array.ndim == 1:
                rgb_array = rgb_array.reshape((height, width, 4))
            rgb_array = rgb_array[:, :, :3]
            
            image = Image.fromarray(rgb_array, 'RGB')
            
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            try:
                font_large = ImageFont.truetype("arial.ttf", 18)
                font_small = ImageFont.truetype("arial.ttf", 14)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            from PIL import Image as PILImage
            overlay = PILImage.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            overlay_draw.rectangle([(0, 0), (width, 120)], fill=(0, 0, 0, 180))
            overlay_draw.rectangle([(0, height-180), (350, height)], fill=(0, 0, 0, 180))
            overlay_draw.rectangle([(width-350, height-120), (width, height)], fill=(0, 0, 0, 180))
            
            image = PILImage.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            y_offset = 10
            title = f"Episode {episode_num} - Step {step}/{max_steps}"
            draw.text((10, y_offset), title, fill=(255, 255, 0), font=font_large)
            y_offset += 25
            
            success_text = "✅ SUCCESS!" if success_achieved else f"Target: {distance_to_target:.3f}m"
            success_color = (0, 255, 0) if success_achieved else (255, 255, 255)
            reward_text = f"Reward: {episode_reward:.1f} | {success_text}"
            draw.text((10, y_offset), reward_text, fill=success_color, font=font_small)
            y_offset += 22
            
            distance_color = (0, 255, 0) if distance_to_target < success_threshold else (255, 165, 0) if distance_to_target < success_threshold * 2 else (255, 255, 255)
            distance_text = f"Distance: {distance_to_target*100:.1f}cm / Threshold: {success_threshold*100:.0f}cm"
            draw.text((10, y_offset), distance_text, fill=distance_color, font=font_small)
            y_offset += 22
            
            stage_colors = [(100, 200, 255), (255, 200, 100), (255, 100, 100)]
            stage_names = ["Easy", "Medium", "Hard"]
            stage_color = stage_colors[min(curriculum_stage, 2)]
            stage_text = f"Stage: {curriculum_stage} ({stage_names[min(curriculum_stage, 2)]}) "
            draw.text((10, y_offset), stage_text, fill=stage_color, font=font_small)
            
            y_offset = height - 170
            draw.text((10, y_offset), "End Effector (TCP):", fill=(100, 200, 255), font=font_small)
            y_offset += 20
            draw.text((10, y_offset), f"  X: {ee_pos[0]:+.3f}m", fill=(255, 255, 255), font=font_small)
            y_offset += 20
            draw.text((10, y_offset), f"  Y: {ee_pos[1]:+.3f}m", fill=(255, 255, 255), font=font_small)
            y_offset += 20
            draw.text((10, y_offset), f"  Z: {ee_pos[2]:+.3f}m", fill=(255, 255, 255), font=font_small)
            y_offset += 25
            
            if hasattr(raw_env, '_get_joint_positions'):
                joint_positions = raw_env._get_joint_positions()
                joint_text = f"Joints: [{', '.join([f'{np.degrees(j):.0f}°' for j in joint_positions[:4]])}]"
                draw.text((10, y_offset), joint_text, fill=(200, 200, 200), font=font_small)
            
            y_offset = height - 110
            draw.text((width-340, y_offset), "Target Position:", fill=(255, 165, 0), font=font_small)
            y_offset += 20
            draw.text((width-340, y_offset), f"  Current X: {target_pos[0]:+.3f}m", fill=(255, 200, 100), font=font_small)
            y_offset += 20
            draw.text((width-340, y_offset), f"  Current Y: {target_pos[1]:+.3f}m", fill=(255, 200, 100), font=font_small)
            y_offset += 20
            draw.text((width-340, y_offset), f"  Current Z: {target_pos[2]:+.3f}m", fill=(255, 200, 100), font=font_small)
            y_offset += 25
            
            drift_color = (255, 100, 100) if target_drift > 0.03 else (100, 255, 100)
            drift_text = f"Drift: {target_drift*100:.1f}cm"
            draw.text((width-340, y_offset), drift_text, fill=drift_color, font=font_small)
            
            frames.append(image)
            
        except Exception as e:
            if step == 0:
                print(f"⚠️ Frame capture failed: {e}")
                import traceback
                traceback.print_exc()
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(reward, (list, np.ndarray)):
            reward = reward[0]
        if isinstance(done, (list, np.ndarray)):
            done = done[0]
        if isinstance(info, list):
            info = info[0] if len(info) > 0 else {}
        
        episode_reward += reward
        episode_length += 1
        
        if info.get('success', False):
            success_achieved = True
        
        if done:
            break
    
    if frames:
        videos_dir = os.path.join(exp_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        gif_path = os.path.join(videos_dir, f"episode_{episode_num}.gif")
        
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,
                loop=0,
                optimize=True
            )
            
            final_distance = info.get('distance', 0)
            success = info.get('success', False)
            
            print(f"  ✅ GIF saved: {gif_path}")
            print(f"     Frames: {len(frames)} | Steps: {episode_length}")
            print(f"     Reward: {episode_reward:.2f} | Final distance: {final_distance*100:.1f}cm")
            print(f"     Success: {'✅ YES' if success else '❌ NO'}")
            
            return gif_path
            
        except Exception as e:
            print(f"❌ Failed to save GIF: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("❌ No frames captured")
        return None


def post_training_visualization(exp_dir, model_path, vecnorm_path, algorithm, stage_config, n_episodes=3):
    """Generate all visualizations after training completes"""
    print("\n" + "="*60)
    print("🎨 Starting post-training visualization...")
    print("="*60)
    
    try:
        algo_classes = {'SAC': SAC, 'TQC': TQC, 'CrossQ': CrossQ}
        model = algo_classes[algorithm].load(model_path)
        # ✅ Filter out parameters not needed by the environment
        env_config = {k: v for k, v in stage_config.items() 
                      if k not in ['description', 'recommended_timesteps']}
        
        # Create test environment using current stage configuration
        test_env = AlphaReachEnv(
            render_mode=None,
            **env_config
        )
        vec_env = DummyVecEnv([lambda: test_env])
        
        if os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print("✅ VecNormalize parameters loaded")
        
        gif_paths = []
        for i in range(n_episodes):
            gif_path = generate_gif_from_model(model, vec_env, exp_dir, episode_num=i+1)
            if gif_path:
                gif_paths.append(gif_path)
        
        vec_env.close()
        
        print(f"\n✅ Successfully generated {len(gif_paths)} GIFs")
        for path in gif_paths:
            print(f"   - {path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


class TrainingProgressCallback(BaseCallback):
    """Training progress callback"""
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episodes = 0
        self.successes = 0
        
    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episodes += 1
                    if info.get('success', False):
                        self.successes += 1
        return True
    
    def _on_training_end(self) -> None:
        if self.episodes > 0:
            success_rate = self.successes / self.episodes
            print(f"Training complete - Success rate: {success_rate:.2%}")


def make_env(idx: int, log_dir: str, stage_config: dict):
    """Create environment"""
    def _init():
        # ✅ Filter out non-environment parameters (description, recommended_timesteps)
        env_config = {k: v for k, v in stage_config.items() 
                      if k not in ['description', 'recommended_timesteps']}
        env = AlphaReachEnv(
            render_mode=None,
            **env_config
        )
        filename = os.path.join(log_dir, f"monitor_{idx}.monitor.csv")
        return Monitor(env, filename=filename, info_keywords=("success", "distance"))
    return _init


def train_alpha_reach(
    algorithm='SAC',
    total_timesteps=300000,
    num_envs=8,
    auto_visualize=True,
    stage=None,  # New: Optional stage parameter
    seed=42
):
    
    """Train Alpha robotic arm - Supports progressive stages"""
    set_global_seed(seed)
    # Use specified stage or global TRAINING_STAGE
    current_stage = stage if stage else TRAINING_STAGE
    
    if current_stage not in STAGE_CONFIGS:
        raise ValueError(f"Unknown stage: {current_stage}. Options: {list(STAGE_CONFIGS.keys())}")
    
    stage_config = STAGE_CONFIGS[current_stage]

    print(f"Starting Alpha robotic arm reaching task training")
    print(f"🎯 Training stage: {current_stage} - {stage_config['description']}")
    print(f"Algorithm: {algorithm}")
    print(f"Total steps: {total_timesteps}")
    print(f"Recommended steps: {stage_config['recommended_timesteps']}")
    print(f"Parallel environments: {num_envs}")
    print("="*50)

    exp_dir, exp_name = create_experiment_folder(algorithm, total_timesteps, current_stage)
    print(f"Experiment folder: {exp_dir}")

    save_path = os.path.join(exp_dir, "models")
    log_path = os.path.join(exp_dir, "logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("Checking environment...")
    # ✅ Filter out non-environment parameters
    env_config = {k: v for k, v in stage_config.items() 
                if k not in ['description', 'recommended_timesteps']}
    test_env = AlphaReachEnv(**env_config)
    check_env(test_env)
    test_env.close()
    print("✅ Environment check passed")
    
    train_env = DummyVecEnv([make_env(i, log_path, stage_config) for i in range(num_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, 
                         clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    eval_env = DummyVecEnv([make_env(0, log_path, stage_config)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False,
                        clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    eval_env.obs_rms = train_env.obs_rms

    configs = {
        # 'SAC': {
        #     'policy': 'MlpPolicy',
        #     'learning_rate': 3e-4,
        #     'buffer_size': 1000000,
        #     'batch_size': 256,
        #     'tau': 0.005,
        #     'gamma': 0.99,
        #     'train_freq': (1, 'step'),
        #     'gradient_steps': 1,
        #     'learning_starts': 10000,
        #     'ent_coef': 'auto',
        #     'target_update_interval': 1,
        #     'use_sde': False,
        #     'policy_kwargs': dict(
        #         net_arch=[256, 256],
        #         log_std_init=-3,
        #     ),
        #     'verbose': 1
        'SAC': {
            'policy': 'MlpPolicy',
            'device': 'cuda',
            'learning_rate': 3e-4,
            'buffer_size': 500000,
            'batch_size': 512,
            'tau': 0.005,
            'gamma': 0.98,
            'train_freq': (max(1, 64 // num_envs), 'step'),
            'gradient_steps': 64,
            'learning_starts': 10000,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'use_sde': False,
            'sde_sample_freq': 4,
            'policy_kwargs': dict(net_arch=[256, 256, 256]),
            'verbose': 1
        },
        'TQC': {
            'policy': 'MlpPolicy',
            'device': 'cuda',
            'learning_rate': 3e-4,
            'buffer_size': 500000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'learning_starts': 10000,
            'top_quantiles_to_drop_per_net': 2,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                n_critics=2,
                n_quantiles=25
            ),
            'verbose': 1
        },
        'CrossQ': {
            'policy': 'MlpPolicy',
            'device': 'cuda',
            'learning_rate': 3e-4,
            'buffer_size': 300000,
            'batch_size': 256,
            'gamma': 0.99,
            'train_freq': 1,
            'learning_starts': 5000,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                n_critics=2
            ),
            'verbose': 1
        }
    }
    
    save_experiment_config(exp_dir, algorithm, configs[algorithm], total_timesteps, stage_config)

    algo_classes = {'SAC': SAC, 'TQC': TQC, 'CrossQ': CrossQ}
    
    if algorithm not in algo_classes:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Please choose SAC/TQC/CrossQ")
    
    print(f"Creating {algorithm} model...")
    model = algo_classes[algorithm](
        env=train_env, 
        tensorboard_log=os.path.join(log_path, "tensorboard"),
        **configs[algorithm]
    )

    print("\nInitializing monitoring system...")
    comprehensive_monitor = ComprehensiveMonitor(
        log_dir=log_path,
        verbose=1,
        window_size=100
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, f"{algorithm}_best"),
        log_path=log_path,
        eval_freq=max(5000, total_timesteps // 20),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000, total_timesteps // 4),
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix=f"{algorithm}_checkpoint"
    )

    progress_callback = TrainingProgressCallback(eval_freq=1000)
    # 🆕 Add: Curriculum monitoring callback
    curriculum_callback = CurriculumMonitorCallback(
        log_dir=log_path,
        verbose=1
    )
    print("Starting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback, comprehensive_monitor, curriculum_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"Training complete, time elapsed: {training_time/3600:.2f} hours")

    model_path = os.path.join(save_path, f"{algorithm}_final")
    model.save(model_path)
    vecnorm_path = os.path.join(save_path, f"{algorithm}_vecnormalize.pkl")
    train_env.save(vecnorm_path)

    print("Performing final evaluation...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    train_env.close()
    eval_env.close()

    print("\n" + "="*60)
    print("📈 Generating training curves...")
    print("="*60)
    plot_training_curves_enhanced(exp_dir, log_path)
    
    if auto_visualize:
        print("\n" + "="*60)
        print("🎬 Generating demonstration GIFs...")
        print("="*60)
        post_training_visualization(exp_dir, model_path, vecnorm_path, algorithm, stage_config, n_episodes=3)

    summary = {
        'algorithm': algorithm,
        'stage': current_stage,
        'stage_description': stage_config['description'],
        'total_timesteps': total_timesteps,
        'training_time_hours': training_time/3600,
        'final_mean_reward': float(mean_reward),
        'final_std_reward': float(std_reward),
        'model_path': model_path
    }
    
    with open(os.path.join(exp_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment complete! All results saved to: {exp_dir}")
    print(f"{'='*60}")
    
    # Provide suggestions for next steps
    if current_stage == 'stage1' and mean_reward > 0:
        print("\n💡 Suggestions:")
        print("   Stage 1 training successful! You can continue to stage 2")
        print("   1. Modify the file header: TRAINING_STAGE = 'stage2'")
        print("   2. Run training again")
    elif current_stage == 'stage2' and mean_reward > -1:
        print("\n💡 Suggestions:")
        print("   Stage 2 training successful! You can continue to stage 3")
        print("   1. Modify the file header: TRAINING_STAGE = 'stage3'")
    elif current_stage == 'stage3' and mean_reward > -2:
        print("\n💡 Suggestions:")
        print("   Stage 3 training successful! You can continue to stage 4")
        print("   1. Modify the file header: TRAINING_STAGE = 'stage4'")
        print("   2. Recommend setting --timesteps 1000000 (stage 4 requires more time)")
    elif current_stage == 'stage4':
        print("\n🎉 Congratulations! Complete progressive training finished!")
        print(f"   Final model: {model_path}")

    return model, mean_reward, std_reward, exp_dir


def test_trained_model(model_path, algorithm='SAC', n_episodes=5, stage=None):
    """Test trained model"""
    print(f"Testing trained {algorithm} model...")
    print(f"Model path: {model_path}")
    
    # Use specified stage or global TRAINING_STAGE
    current_stage = stage if stage else TRAINING_STAGE
    stage_config = STAGE_CONFIGS.get(current_stage, STAGE_CONFIGS['stage1'])
    
    env = AlphaReachEnv(render_mode="human", **stage_config)
    
    algo_classes = {'SAC': SAC, 'TQC': TQC, 'CrossQ': CrossQ}
    
    if algorithm not in algo_classes:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Please choose SAC/TQC/CrossQ")
    
    model = algo_classes[algorithm].load(model_path)
    
    try:
        vec_normalize_path = model_path.replace('_final', '_vecnormalize.pkl')
        if os.path.exists(vec_normalize_path):
            print("Loading environment normalization parameters...")
            test_env = DummyVecEnv([lambda: env])
            test_env = VecNormalize.load(vec_normalize_path, test_env)
            test_env.training = False
            test_env.norm_reward = False
            use_vec_env = True
        else:
            use_vec_env = False
    except:
        use_vec_env = False
    
    successes = 0
    total_rewards = []
    distances = []
    
    for episode in range(n_episodes):
        if use_vec_env:
            obs = test_env.reset()
        else:
            obs, _ = env.reset()
            
        episode_reward = 0
        step_count = 0
        
        for step in range(500):
            if use_vec_env:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                episode_reward += reward[0]
                step_count += 1
                
                if done[0]:
                    info = info[0] if len(info) > 0 else {}
                    break
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
        
        success = info.get('success', False)
        distance = info.get('distance', 0)
        
        if success:
            successes += 1
        
        total_rewards.append(episode_reward)
        distances.append(distance)
        
        print(f"Episode {episode+1}: Success={success}, Distance={distance:.3f}m, "
              f"Reward={episode_reward:.2f}, Steps={step_count}")
    
    print("\n" + "="*50)
    print("Test Results Statistics:")
    print("="*50)
    print(f"Success Rate: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"Mean Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Mean Final Distance: {np.mean(distances):.3f} ± {np.std(distances):.3f}m")
    
    env.close()
    
    return {
        'success_rate': successes / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'mean_distance': np.mean(distances)
    }


def compare_algorithms(algorithms=['SAC', 'TQC', 'CrossQ'], timesteps=500000):
    """Compare performance of different algorithms"""
    set_global_seed(42)
    print("Starting algorithm performance comparison")
    print(f"Algorithms: {algorithms}")
    print(f"Training steps: {timesteps}")
    print(f"Current stage: {TRAINING_STAGE}")
    print("="*50)
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\nTraining {algorithm}...")
        try:
            model, mean_reward, std_reward, exp_dir = train_alpha_reach(
                algorithm=algorithm,
                total_timesteps=timesteps,
                num_envs=8,
                auto_visualize=True
            )
            results[algorithm] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'model': model,
                'exp_dir': exp_dir
            }
            print(f"✅ {algorithm} training complete")
        except Exception as e:
            print(f"❌ {algorithm} training failed: {e}")
            import traceback
            traceback.print_exc()
            results[algorithm] = {
                'mean_reward': -np.inf,
                'std_reward': 0,
                'model': None,
                'exp_dir': None
            }
    
    print("\n" + "="*60)
    print("Algorithm Performance Comparison Results:")
    print("="*60)
    print(f"{'Algorithm':<10} {'Mean Reward':<12} {'Std Dev':<10} {'Relative Perf':<10}")
    print("-" * 60)
    
    valid_rewards = [r['mean_reward'] for r in results.values() if r['mean_reward'] != -np.inf]
    if valid_rewards:
        best_reward = max(valid_rewards)
        
        for algo, result in results.items():
            if result['mean_reward'] != -np.inf:
                relative_perf = result['mean_reward'] / best_reward * 100
                print(f"{algo:<10} {result['mean_reward']:<12.2f} {result['std_reward']:<10.2f} {relative_perf:<10.1f}%")
            else:
                print(f"{algo:<10} {'Failed':<12} {'-':<10} {'-':<10}")
        
        best_algo = max(results.keys(), key=lambda x: results[x]['mean_reward'])
        if results[best_algo]['mean_reward'] != -np.inf:
            print(f"\n🏆 Best Algorithm: {best_algo} (Reward: {results[best_algo]['mean_reward']:.2f})")
    else:
        print("⚠️ All algorithms failed training")
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha Robotic Arm Progressive Training')
    parser.add_argument('--mode', choices=['train', 'test', 'compare'], 
                       default='train', help='Running mode')
    parser.add_argument('--algorithm', choices=['SAC', 'TQC', 'CrossQ'], 
                       default='SAC', help='RL algorithm')
    parser.add_argument('--timesteps', type=int, default=500000, 
                       help='Training steps')
    parser.add_argument('--num_envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--model', type=str, 
                       help='Model path for test mode')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='Number of test episodes')
    parser.add_argument('--auto_visualize', action='store_true', default=True,
                       help='Automatically generate GIFs and charts after training')
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3', 'stage4'],
                       help='Specify training stage (overrides global TRAINING_STAGE)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (ensures reproducibility)')
    
    args = parser.parse_args()
    
    # Print current stage configuration
    current_stage = args.stage if args.stage else TRAINING_STAGE
    print("\n" + "="*70)
    print(f"🎯 Current training stage: {current_stage}")
    print(f"   Description: {STAGE_CONFIGS[current_stage]['description']}")
    print(f"   Recommended steps: {STAGE_CONFIGS[current_stage]['recommended_timesteps']:,}")
    print("="*70 + "\n")
    
    if args.mode == 'train':
        print("Single algorithm training mode")
        train_alpha_reach(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            auto_visualize=args.auto_visualize,
            stage=args.stage,
            seed=args.seed
        )
        
    elif args.mode == 'test':
        if not args.model:
            model_path = f"./models/{args.algorithm}_best/best_model"
            if not os.path.exists(model_path + '.zip'):
                print(f"Error: Cannot find model file {model_path}")
                return
        else:
            model_path = args.model
            
        print("Model test mode")
        test_trained_model(
            model_path=model_path,
            algorithm=args.algorithm,
            n_episodes=args.episodes,
            stage=args.stage
        )
        
    elif args.mode == 'compare':
        print("Algorithm comparison mode")
        results = compare_algorithms(
            algorithms=['SAC', 'TQC', 'CrossQ'],
            timesteps=args.timesteps
        )


if __name__ == "__main__":
    main()