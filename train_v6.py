import os
import numpy as np
import json
import shutil
from datetime import datetime
import glob

# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 避免Matplotlib在受限环境下写入HOME目录失败
default_mpl_dir = os.path.join(os.path.dirname(__file__), '.mplconfig')
os.environ.setdefault('MPLCONFIGDIR', default_mpl_dir)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import matplotlib.pyplot as plt
import pybullet as p
from PIL import Image
from pathlib import Path

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
import time

# 导入环境
from envs.rl_env_v5 import AlphaReachEnv


def create_experiment_folder(algorithm, timesteps):
    """创建实验文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{algorithm}_{timesteps}steps_drift_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)

    return exp_dir, exp_name


def save_experiment_config(exp_dir, algorithm, config, timesteps):
    """保存实验配置"""
    exp_config = {
        "algorithm": algorithm,
        "timesteps": timesteps,
        "timestamp": datetime.now().isoformat(),
        "features": [
            "漂移目标（5cm内随机摆动）",
            "简单速度估计（历史位置差分）",
            "无夹爪控制（始终张开）",
            "17维观察空间",
            "4维动作空间",
            "自动GIF生成"
        ],
        "config": config
    }

    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(exp_config, f, indent=2)


def plot_training_curves(exp_dir, log_dir):
    """绘制和保存训练曲线"""
    try:
        log_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        if not log_files:
            log_files = glob.glob(os.path.join(log_dir, "*.csv"))
        if not log_files:
            print("警告：未找到训练日志文件")
            return

        import pandas as pd

        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'lines.linewidth': 2,
            'grid.alpha': 0.3
        })

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Progress - Alpha Arm with Drifting Target', 
                     fontsize=18, fontweight='bold')

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for idx, log_file in enumerate(log_files):
            try:
                df = pd.read_csv(log_file, comment='#')
                color = colors[idx % len(colors)]

                if 'r' in df.columns and 'l' in df.columns:
                    raw_rewards = df['r']
                    smoothed_rewards = df['r'].rolling(window=100, min_periods=1).mean()

                    axes[0, 0].plot(df['l'], raw_rewards, alpha=0.2, color=color, linewidth=0.5)
                    axes[0, 0].plot(df['l'], smoothed_rewards, linewidth=3, color=color,
                                   label=f'Env {idx+1}' if len(log_files) > 1 else 'Episode Reward')

                    axes[0, 0].set_title('Episode Reward Over Time', fontweight='bold')
                    axes[0, 0].set_xlabel('Training Timesteps')
                    axes[0, 0].set_ylabel('Cumulative Reward')
                    axes[0, 0].grid(True, alpha=0.3)
                    if len(log_files) > 1:
                        axes[0, 0].legend()

                if 't' in df.columns:
                    raw_lengths = df['t']
                    smoothed_lengths = df['t'].rolling(window=100, min_periods=1).mean()

                    axes[0, 1].plot(df['l'], raw_lengths, alpha=0.2, color=color, linewidth=0.5)
                    axes[0, 1].plot(df['l'], smoothed_lengths, linewidth=3, color=color,
                                   label=f'Env {idx+1}' if len(log_files) > 1 else 'Episode Length')

                    axes[0, 1].set_title('Episode Length Over Time', fontweight='bold')
                    axes[0, 1].set_xlabel('Training Timesteps')
                    axes[0, 1].set_ylabel('Steps per Episode')
                    axes[0, 1].grid(True, alpha=0.3)
                    if len(log_files) > 1:
                        axes[0, 1].legend()

            except Exception as e:
                print(f"读取日志文件失败 {log_file}: {e}")

        eval_log = os.path.join(log_dir, "evaluations.npz")
        if os.path.exists(eval_log):
            try:
                eval_data = np.load(eval_log)
                timesteps = eval_data['timesteps']
                results = eval_data['results']

                mean_rewards = np.mean(results, axis=1)
                std_rewards = np.std(results, axis=1)

                axes[1, 0].plot(timesteps, mean_rewards, color='#27AE60', linewidth=3,
                               label='Mean Evaluation Reward', marker='o', markersize=4)
                axes[1, 0].fill_between(timesteps,
                                       mean_rewards - std_rewards,
                                       mean_rewards + std_rewards,
                                       alpha=0.25, color='#27AE60', label='±1 Std Dev')
                axes[1, 0].set_title('Evaluation Performance', fontweight='bold')
                axes[1, 0].set_xlabel('Training Timesteps')
                axes[1, 0].set_ylabel('Evaluation Reward')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

                if 'successes' in eval_data:
                    success_rates = eval_data['successes']
                    axes[1, 1].plot(timesteps, success_rates, color='#E74C3C', linewidth=3,
                                   marker='s', markersize=4, label='Success Rate')
                    axes[1, 1].set_title('Task Success Rate', fontweight='bold')
                    axes[1, 1].set_xlabel('Training Timesteps')
                    axes[1, 1].set_ylabel('Success Rate')
                    axes[1, 1].set_ylim(0, 1.05)
                    axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].legend()
                else:
                    axes[1, 1].text(0.5, 0.5, 'Success Rate\nData Not Available',
                                   transform=axes[1, 1].transAxes, ha='center', va='center',
                                   fontsize=14, alpha=0.6)
                    axes[1, 1].set_title('Task Success Rate', fontweight='bold')

            except Exception as e:
                print(f"读取评估日志失败: {e}")
        else:
            for i in range(2):
                axes[1, i].text(0.5, 0.5, 'Evaluation Data\nNot Available',
                               transform=axes[1, i].transAxes, ha='center', va='center',
                               fontsize=14, alpha=0.6)
                axes[1, i].set_title(['Evaluation Performance', 'Task Success Rate'][i], fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(exp_dir, "plots", "training_curves.png"),
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"✅ 训练曲线已保存到: {os.path.join(exp_dir, 'plots', 'training_curves.png')}")

    except ImportError:
        print("警告：pandas未安装，无法绘制训练曲线")
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")


class TrainingProgressCallback(BaseCallback):
    """训练进度回调"""
    
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.success_rates = []
        self.distances = []
        self.episodes = 0
        self.successes = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'success' in info:
                self.episodes += 1
                if info['success']:
                    self.successes += 1
                self.distances.append(info.get('distance', 0))
        
        return True
    
    def _on_training_end(self) -> None:
        if self.episodes > 0:
            success_rate = self.successes / self.episodes
            avg_distance = np.mean(self.distances) if self.distances else 0
            print(f"\n✅ 训练完成统计:")
            print(f"   总回合数: {self.episodes}")
            print(f"   成功率: {success_rate:.2%}")
            print(f"   平均最终距离: {avg_distance:.3f}m")


def make_env(idx: int, log_dir: str, render_mode=None):
    """创建环境包装器"""
    def _init():
        env = AlphaReachEnv(render_mode=render_mode)
        filename = os.path.join(log_dir, f"monitor_{idx}.monitor.csv")
        return Monitor(env, filename=filename, info_keywords=("success", "distance"))
    return _init


def render_frame(env, episode, step, reward):
    """
    渲染单帧图像
    """
    try:
        width, height = 640, 480
        
        ee_pos = env._get_end_effector_position()
        target_pos = env.target_position
        
        camera_target = [(ee_pos[0] + target_pos[0]) / 2,
                        (ee_pos[1] + target_pos[1]) / 2,
                        (ee_pos[2] + target_pos[2]) / 2]
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera_target,
            distance=1.5,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
            physicsClientId=env.physics_client
        )
        
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=env.physics_client
        )
        
        img_arr = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=env.physics_client
        )
        
        rgb_data = img_arr[2]
        rgb_array = np.array(rgb_data, dtype=np.uint8)
        
        if rgb_array.ndim == 1:
            rgb_array = rgb_array.reshape((height, width, 4))
        
        rgb_array = rgb_array[:, :, :3]
        image = Image.fromarray(rgb_array, 'RGB')
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        current_distance = np.linalg.norm(ee_pos - target_pos)
        
        if hasattr(env, 'initial_target_position'):
            drift = np.linalg.norm(target_pos - env.initial_target_position)
            drift_text = f"Drift: {drift*100:.1f}cm"
        else:
            drift_text = ""
        
        info_text = f"Episode: {episode} | Step: {step} | Reward: {reward:.1f}"
        distance_text = f"Distance: {current_distance*100:.1f}cm | {drift_text}"
        target_text = f"Target: [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}]"
        
        texts = [info_text, distance_text, target_text]
        y_offset = 10
        
        for text in texts:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.rectangle(
                [(5, y_offset), (text_width + 15, y_offset + text_height + 5)],
                fill=(0, 0, 0, 180)
            )
            
            draw.text((10, y_offset), text, fill=(255, 255, 255), font=font)
            y_offset += text_height + 10
        
        return image
        
    except Exception as e:
        print(f"渲染帧失败: {e}")
        return None


def create_combined_gif(results, videos_dir, algorithm):
    """创建综合GIF（所有成功回合）"""
    from PIL import Image, ImageDraw, ImageFont
    
    combined_frames = []
    
    success_episodes = [
        info for info in results['episode_info'] 
        if info['success']
    ]
    
    if not success_episodes:
        return
    
    print(f"  合并 {len(success_episodes)} 个成功回合...")
    
    for idx, episode_info in enumerate(success_episodes[:3]):
        episode_num = episode_info['episode']
        gif_path = results['gif_paths'][episode_num - 1]
        
        if gif_path and os.path.exists(gif_path):
            try:
                with Image.open(gif_path) as gif:
                    for frame_idx in range(0, gif.n_frames, 2):
                        gif.seek(frame_idx)
                        frame = gif.copy().convert('RGB')
                        
                        draw = ImageDraw.Draw(frame)
                        try:
                            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                        except:
                            try:
                                font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 20)
                            except:
                                font = ImageFont.load_default()
                        
                        label = f"Success Episode {episode_num}"
                        draw.text((10, 50), label, fill=(0, 255, 0), font=font)
                        
                        combined_frames.append(frame)
                    
                    if idx < len(success_episodes) - 1:
                        black_frame = Image.new('RGB', frame.size, (0, 0, 0))
                        draw = ImageDraw.Draw(black_frame)
                        draw.text(
                            (black_frame.width//2 - 50, black_frame.height//2),
                            "Next Episode",
                            fill=(255, 255, 255),
                            font=font
                        )
                        combined_frames.extend([black_frame] * 20)
                        
            except Exception as e:
                print(f"  处理回合 {episode_num} 失败: {e}")
    
    if combined_frames:
        combined_path = os.path.join(videos_dir, f"{algorithm}_successful_episodes.gif")
        
        title_frame = Image.new('RGB', combined_frames[0].size, (0, 0, 0))
        draw = ImageDraw.Draw(title_frame)
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            try:
                title_font = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 24)
            except:
                title_font = ImageFont.load_default()
        
        title_text = f"{algorithm} - Successful Episodes"
        stats_text = f"{len(success_episodes)} successful attempts"
        
        bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = bbox[2] - bbox[0]
        title_x = (title_frame.width - title_width) // 2
        
        draw.text((title_x, title_frame.height//2 - 30), title_text, 
                 fill=(255, 255, 255), font=title_font)
        draw.text((title_x + 20, title_frame.height//2 + 10), stats_text,
                 fill=(200, 200, 200), font=title_font)
        
        final_frames = [title_frame] * 40 + combined_frames
        
        final_frames[0].save(
            combined_path,
            save_all=True,
            append_images=final_frames[1:],
            duration=33,
            loop=0,
            optimize=True
        )
        
        file_size = os.path.getsize(combined_path) / 1024 / 1024
        print(f"  ✅ 综合GIF已保存: {os.path.basename(combined_path)} ({file_size:.1f}MB)")


def generate_training_gifs(exp_dir, model_path, vecnorm_path, algorithm, n_episodes=5, max_steps=800):
    """
    训练完成后自动生成GIF
    """
    
    print("⚙️  初始化GIF录制环境...")
    
    videos_dir = os.path.join(exp_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    model = algo_classes[algorithm].load(model_path)
    
    env = AlphaReachEnv(render_mode=None)
    
    if os.path.exists(vecnorm_path):
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        use_vec = True
    else:
        use_vec = False
    
    results = {
        'total': n_episodes,
        'success_count': 0,
        'gif_paths': [],
        'episode_info': []
    }
    
    for episode in range(n_episodes):
        print(f"\n🎬 录制回合 {episode + 1}/{n_episodes}...")
        
        frames = []
        
        if use_vec:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()
        
        episode_reward = 0
        step_count = 0
        success = False
        
        for step in range(max_steps):
            if step % 2 == 0:
                try:
                    frame = render_frame(env, episode + 1, step, episode_reward)
                    if frame is not None:
                        frames.append(frame)
                except Exception as e:
                    print(f"  ⚠️  步数{step}渲染失败: {e}")
            
            if use_vec:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                episode_reward += reward[0]
                step_count += 1
                
                if done[0]:
                    info = info[0] if len(info) > 0 else {}
                    success = info.get('success', False)
                    break
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    success = info.get('success', False)
                    break
        
        if frames:
            gif_path = os.path.join(videos_dir, f"episode_{episode + 1}.gif")
            try:
                print(f"  💾 保存GIF: {len(frames)}帧...")
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=33,
                    loop=0,
                    optimize=True
                )
                
                file_size = os.path.getsize(gif_path) / 1024 / 1024
                print(f"  ✅ 保存成功: {os.path.basename(gif_path)} ({file_size:.1f}MB)")
                
                results['gif_paths'].append(gif_path)
                results['episode_info'].append({
                    'episode': episode + 1,
                    'success': success,
                    'reward': episode_reward,
                    'steps': step_count,
                    'frames': len(frames),
                    'file_size_mb': file_size
                })
                
                if success:
                    results['success_count'] += 1
                    
            except Exception as e:
                print(f"  ❌ GIF保存失败: {e}")
                results['gif_paths'].append(None)
        else:
            print(f"  ❌ 未采集到帧")
            results['gif_paths'].append(None)
    
    env.close()
    if use_vec:
        vec_env.close()
    
    if results['success_count'] > 0:
        try:
            print(f"\n🎬 生成综合GIF（所有成功回合）...")
            create_combined_gif(results, videos_dir, algorithm)
        except Exception as e:
            print(f"⚠️  综合GIF生成失败: {e}")
    
    return results


def train_alpha_reach(
    algorithm='SAC',
    total_timesteps=500000,
    num_envs=4,
    auto_visualize=True
):
    """训练Alpha机械臂（漂移目标版本）"""

    print("\n" + "="*70)
    print("🚀 开始训练Alpha机械臂 - 漂移目标版本")
    print("="*70)
    print(f"算法: {algorithm}")
    print(f"总步数: {total_timesteps:,}")
    print(f"并行环境: {num_envs}")
    print("特性: 漂移目标 + 速度估计 + 无夹爪 + 自动GIF")
    print("="*70 + "\n")

    exp_dir, exp_name = create_experiment_folder(algorithm, total_timesteps)
    print(f"📁 实验文件夹: {exp_dir}\n")

    save_path = os.path.join(exp_dir, "models")
    log_path = os.path.join(exp_dir, "logs")

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print("🔍 检查环境...")
    test_env = AlphaReachEnv()
    check_env(test_env)
    test_env.close()
    print("✅ 环境检查通过\n")
    
    print(f"🏗️  创建 {num_envs} 个训练环境...")
    train_env = DummyVecEnv([make_env(i, log_path) for i in range(num_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99
    )
    print("✅ 训练环境创建完成\n")

    print("🏗️  创建评估环境...")
    eval_env = DummyVecEnv([make_env(0, log_path)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        clip_obs=10.0
    )
    eval_env.obs_rms = train_env.obs_rms
    print("✅ 评估环境创建完成\n")

    configs = {
        'SAC': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': (1, 'step'),
            'gradient_steps': 1,
            'learning_starts': 20000,
            'ent_coef': 'auto_0.1',
            'target_update_interval': 1,
            'use_sde': False,
            'policy_kwargs': dict(net_arch=[400, 400, 300]),
            'verbose': 1
        },
        'PPO': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': max(256, 2048 // num_envs),
            'batch_size': 128,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': dict(net_arch=[400, 400, 300]),
            'verbose': 1
        },
        'TD3': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': (1, 'step'),
            'gradient_steps': 1,
            'learning_starts': 20000,
            'target_policy_noise': 0.1,
            'target_noise_clip': 0.3,
            'policy_delay': 2,
            'policy_kwargs': dict(net_arch=[400, 400, 300]),
            'verbose': 1
        }
    }
    
    save_experiment_config(exp_dir, algorithm, configs[algorithm], total_timesteps)
    
    print(f"⚙️  算法配置 ({algorithm}):")
    for key, value in configs[algorithm].items():
        if key != 'policy_kwargs':
            print(f"   {key}: {value}")
    print(f"   网络结构: {configs[algorithm]['policy_kwargs']['net_arch']}\n")

    print(f"🤖 创建 {algorithm} 模型...")
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    model = algo_classes[algorithm](
        env=train_env, 
        tensorboard_log=os.path.join(log_path, "tensorboard"),
        **configs[algorithm]
    )
    print("✅ 模型创建完成\n")

    print("⚙️  设置回调函数...")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, f"{algorithm}_best"),
        log_path=log_path,
        eval_freq=max(10000, total_timesteps // 50),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000, total_timesteps // 10),
        save_path=os.path.join(save_path, "checkpoints"),
        name_prefix=f"{algorithm}_checkpoint"
    )

    progress_callback = TrainingProgressCallback(eval_freq=1000)
    print("✅ 回调函数设置完成\n")
    
    print("="*70)
    print("🎯 开始训练...")
    print("="*70)
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"\n✅ 训练完成！耗时: {training_time/3600:.2f} 小时\n")

    print("💾 保存最终模型...")
    model_path = os.path.join(save_path, f"{algorithm}_final")
    model.save(model_path)
    train_env.save(os.path.join(save_path, f"{algorithm}_vecnormalize.pkl"))
    print(f"✅ 模型已保存: {model_path}.zip\n")

    print("📊 进行最终评估...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )

    print(f"\n{'='*70}")
    print("📈 最终评估结果:")
    print(f"{'='*70}")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"{'='*70}\n")

    print("📦 模型保存情况:")
    models_dir = os.path.join(exp_dir, 'models')
    if os.path.exists(os.path.join(models_dir, f"{algorithm}_final.zip")):
        print(f"  ✅ 最终模型: {algorithm}_final.zip")
    if os.path.exists(os.path.join(models_dir, f"{algorithm}_best.zip")):
        print(f"  ✅ 最佳模型: {algorithm}_best.zip")
    if os.path.exists(os.path.join(models_dir, f"{algorithm}_vecnormalize.pkl")):
        print(f"  ✅ 标准化参数: {algorithm}_vecnormalize.pkl")
    print()

    train_env.close()
    eval_env.close()

    print("📊 绘制训练曲线...")
    plot_training_curves(exp_dir, log_path)

    training_summary = {
        'algorithm': algorithm,
        'total_timesteps': total_timesteps,
        'training_time_hours': training_time / 3600,
        'final_mean_reward': float(mean_reward),
        'final_std_reward': float(std_reward),
        'experiment_name': exp_name,
        'model_path': model_path,
        'features': [
            '漂移目标（5cm内）',
            '速度估计（历史位置差分）',
            '无夹爪控制',
            f'优化超参数（{algorithm}）'
        ]
    }

    with open(os.path.join(exp_dir, "training_summary.json"), 'w') as f:
        json.dump(training_summary, f, indent=2)

    # ✅✅✅ 自动生成GIF ✅✅✅
    if auto_visualize:
        print(f"\n{'='*70}")
        print("🎬 开始生成训练结果GIF...")
        print(f"{'='*70}\n")
        
        try:
            gif_results = generate_training_gifs(
                exp_dir=exp_dir,
                model_path=model_path,
                vecnorm_path=os.path.join(save_path, f"{algorithm}_vecnormalize.pkl"),
                algorithm=algorithm,
                n_episodes=5,
                max_steps=800
            )
            
            if gif_results['success_count'] > 0:
                print(f"\n✅ 成功生成 {gif_results['success_count']}/{gif_results['total']} 个GIF")
                print(f"📁 GIF保存位置: {os.path.join(exp_dir, 'videos')}")
                
                for gif_path in gif_results['gif_paths']:
                    if gif_path and os.path.exists(gif_path):
                        file_size = os.path.getsize(gif_path) / 1024 / 1024
                        print(f"  📹 {os.path.basename(gif_path)} ({file_size:.1f}MB)")
            else:
                print("⚠️  未能生成GIF文件")
                
        except Exception as e:
            print(f"❌ GIF生成失败: {e}")
            import traceback
            traceback.print_exc()
            print("训练已完成，但GIF生成遇到问题")

    print(f"\n{'='*70}")
    print(f"🎉 实验完成！所有结果已保存到: {exp_dir}")
    print(f"{'='*70}")
    print(f"包含内容:")
    print(f"  📁 模型权重: {os.path.join(exp_dir, 'models')}")
    print(f"  📁 训练日志: {os.path.join(exp_dir, 'logs')}")
    print(f"  📁 训练曲线: {os.path.join(exp_dir, 'plots')}")
    print(f"  📁 训练GIF: {os.path.join(exp_dir, 'videos')}")
    print(f"  📄 配置文件: config.json")
    print(f"  📄 训练总结: training_summary.json")
    print(f"{'='*70}\n")

    return model, mean_reward, std_reward, exp_dir


def test_trained_model(model_path, vecnorm_path=None, algorithm='SAC', n_episodes=5):
    """测试训练好的模型"""
    
    print("\n" + "="*70)
    print(f"🧪 测试训练好的{algorithm}模型")
    print("="*70)
    print(f"模型路径: {model_path}")
    if vecnorm_path:
        print(f"归一化参数: {vecnorm_path}")
    print(f"测试回合数: {n_episodes}")
    print("="*70 + "\n")
    
    env = AlphaReachEnv(render_mode="human")
    
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    model = algo_classes[algorithm].load(model_path)
    
    if vecnorm_path and os.path.exists(vecnorm_path):
        print("✅ 加载环境标准化参数...")
        test_env = DummyVecEnv([lambda: env])
        test_env = VecNormalize.load(vecnorm_path, test_env)
        test_env.training = False
        test_env.norm_reward = False
        use_vec_env = True
    else:
        print("⚠️  未找到标准化参数，使用原始环境")
        use_vec_env = False
    
    successes = 0
    total_rewards = []
    distances = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        print(f"\n{'='*70}")
        print(f"🎮 回合 {episode+1}/{n_episodes}")
        print(f"{'='*70}")
        
        if use_vec_env:
            obs = test_env.reset()
        else:
            obs, _ = env.reset()
            
        episode_reward = 0
        step_count = 0
        max_drift = 0.0
        
        for step in range(800):
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
            
            if hasattr(env, 'initial_target_position'):
                drift = np.linalg.norm(env.target_position - env.initial_target_position)
                max_drift = max(max_drift, drift)
            
            if step % 50 == 0:
                print(f"  步数 {step}: 距离={info.get('distance', 0):.3f}m, "
                      f"漂移={drift*100:.1f}cm")
        
        success = info.get('success', False)
        distance = info.get('distance', 0)
        
        if success:
            successes += 1
        
        total_rewards.append(episode_reward)
        distances.append(distance)
        episode_lengths.append(step_count)
        
        print(f"\n📊 回合 {episode+1} 结果:")
        print(f"  ✅ 成功: {'是' if success else '否'}")
        print(f"  📏 最终距离: {distance:.3f}m")
        print(f"  🎁 总奖励: {episode_reward:.2f}")
        print(f"  ⏱️  步数: {step_count}")
        print(f"  🌊 最大漂移: {max_drift*100:.1f}cm")
    
    print(f"\n{'='*70}")
    print("📊 测试结果统计")
    print(f"{'='*70}")
    print(f"总回合数: {n_episodes}")
    print(f"成功率: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均最终距离: {np.mean(distances):.3f} ± {np.std(distances):.3f}m")
    print(f"平均回合长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}步")
    
    if successes > 0:
        success_distances = [d for i, d in enumerate(distances) 
                            if total_rewards[i] > 0]
        if success_distances:
            print(f"成功时平均距离: {np.mean(success_distances):.3f}m")
    print(f"{'='*70}\n")
    
    env.close()
    
    return {
        'success_rate': successes / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'mean_distance': np.mean(distances),
        'mean_length': np.mean(episode_lengths)
    }


def compare_algorithms(algorithms=['SAC', 'PPO', 'TD3'], timesteps=500000):
    """比较不同算法的性能"""
    
    print("\n" + "="*70)
    print("🔬 算法性能比较")
    print("="*70)
    print(f"算法: {algorithms}")
    print(f"训练步数: {timesteps:,}")
    print("="*70 + "\n")
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*70}")
        print(f"🚀 正在训练 {algorithm}...")
        print(f"{'='*70}")
        
        try:
            model, mean_reward, std_reward, exp_dir = train_alpha_reach(
                algorithm=algorithm,
                total_timesteps=timesteps,
                num_envs=4,
                auto_visualize=False
            )
            results[algorithm] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'model': model,
                'exp_dir': exp_dir
            }
            print(f"✅ {algorithm} 训练完成")
        except Exception as e:
            print(f"❌ {algorithm} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            results[algorithm] = {
                'mean_reward': -np.inf,
                'std_reward': 0,
                'model': None,
                'exp_dir': None
            }
    
    print(f"\n{'='*70}")
    print("📊 算法性能比较结果")
    print(f"{'='*70}")
    print(f"{'算法':<8} {'平均奖励':<12} {'标准差':<10} {'相对性能':<10}")
    print("-" * 70)
    
    best_reward = max([r['mean_reward'] for r in results.values() 
                       if r['mean_reward'] != -np.inf])
    
    for algo, result in results.items():
        if result['mean_reward'] != -np.inf:
            relative_perf = result['mean_reward'] / best_reward * 100
            print(f"{algo:<8} {result['mean_reward']:<12.2f} "
                  f"{result['std_reward']:<10.2f} {relative_perf:<10.1f}%")
        else:
            print(f"{algo:<8} {'失败':<12} {'-':<10} {'-':<10}")
    
    best_algo = max(results.keys(), key=lambda x: results[x]['mean_reward'])
    if results[best_algo]['mean_reward'] != -np.inf:
        print(f"\n🏆 最佳算法: {best_algo} "
              f"(奖励: {results[best_algo]['mean_reward']:.2f})")
    
    print(f"{'='*70}\n")
    
    comparison_dir = os.path.join("experiments", "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)
    
    comparison_file = os.path.join(
        comparison_dir, 
        f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'timesteps': timesteps,
        'results': {
            algo: {
                'mean_reward': float(r['mean_reward']),
                'std_reward': float(r['std_reward']),
                'exp_dir': r['exp_dir']
            }
            for algo, r in results.items()
        },
        'best_algorithm': best_algo
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"💾 比较结果已保存到: {comparison_file}\n")
    
    return results


def train_with_curriculum(algorithm='SAC', total_timesteps=300000):
    """课程学习：从简单到困难"""
    
    print("\n" + "="*70)
    print("🎓 课程学习训练模式")
    print("="*70)
    print(f"算法: {algorithm}")
    print(f"总步数: {total_timesteps:,}")
    print("阶段: 3个（静态 → 小漂移 → 正常漂移）")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{algorithm}_curriculum_{total_timesteps}steps_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    save_path = os.path.join(exp_dir, "models")
    log_path = os.path.join(exp_dir, "logs")
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    steps_per_stage = total_timesteps // 3
    
    print("\n" + "="*70)
    print("📚 阶段1: 静态目标（无漂移）")
    print("="*70)
    
    def make_env_stage1(idx):
        def _init():
            env = AlphaReachEnv(render_mode=None)
            env.max_target_drift = 0.0
            env.drift_noise_strength = 0.0
            filename = os.path.join(log_path, f"stage1_monitor_{idx}.monitor.csv")
            return Monitor(env, filename=filename, info_keywords=("success", "distance"))
        return _init
    
    env_stage1 = DummyVecEnv([make_env_stage1(i) for i in range(4)])
    env_stage1 = VecNormalize(env_stage1, norm_obs=True, norm_reward=False)
    
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    
    config = {
        'policy': 'MlpPolicy',
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'batch_size': 256,
        'gamma': 0.99,
        'verbose': 1,
        'tensorboard_log': os.path.join(log_path, "tensorboard")
    }
    
    if algorithm == 'SAC':
        config.update({
            'tau': 0.005,
            'train_freq': (1, 'step'),
            'gradient_steps': 1,
            'learning_starts': 10000,
            'ent_coef': 'auto_0.1',
            'use_sde': False,
            'policy_kwargs': dict(net_arch=[400, 400, 300])
        })
    elif algorithm == 'PPO':
        config.update({
            'n_steps': 512,
            'batch_size': 128,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'policy_kwargs': dict(net_arch=[400, 400, 300])
        })
    elif algorithm == 'TD3':
        config.update({
            'tau': 0.005,
            'train_freq': (1, 'step'),
            'gradient_steps': 1,
            'learning_starts': 10000,
            'target_policy_noise': 0.1,
            'target_noise_clip': 0.3,
            'policy_delay': 2,
            'policy_kwargs': dict(net_arch=[400, 400, 300])
        })
    
    model = algo_classes[algorithm](env=env_stage1, **config)
    
    print(f"🎯 训练阶段1: {steps_per_stage:,}步")
    model.learn(total_timesteps=steps_per_stage, progress_bar=True)
    print("✅ 阶段1完成\n")
    
    print("="*70)
    print("📚 阶段2: 小漂移（2cm内）")
    print("="*70)
    
    def make_env_stage2(idx):
        def _init():
            env = AlphaReachEnv(render_mode=None)
            env.max_target_drift = 0.02
            env.drift_noise_strength = 0.005
            filename = os.path.join(log_path, f"stage2_monitor_{idx}.monitor.csv")
            return Monitor(env, filename=filename, info_keywords=("success", "distance"))
        return _init
    
    env_stage2 = DummyVecEnv([make_env_stage2(i) for i in range(4)])
    env_stage2 = VecNormalize(env_stage2, norm_obs=True, norm_reward=False)
    env_stage2.obs_rms = env_stage1.obs_rms
    
    model.set_env(env_stage2)
    
    print(f"🎯 训练阶段2: {steps_per_stage:,}步")
    model.learn(total_timesteps=steps_per_stage, progress_bar=True, reset_num_timesteps=False)
    print("✅ 阶段2完成\n")
    
    print("="*70)
    print("📚 阶段3: 正常漂移（5cm内）")
    print("="*70)
    
    def make_env_stage3(idx):
        def _init():
            env = AlphaReachEnv(render_mode=None)
            env.max_target_drift = 0.05
            env.drift_noise_strength = 0.01
            filename = os.path.join(log_path, f"stage3_monitor_{idx}.monitor.csv")
            return Monitor(env, filename=filename, info_keywords=("success", "distance"))
        return _init
    
    env_stage3 = DummyVecEnv([make_env_stage3(i) for i in range(4)])
    env_stage3 = VecNormalize(env_stage3, norm_obs=True, norm_reward=False)
    env_stage3.obs_rms = env_stage2.obs_rms
    
    model.set_env(env_stage3)
    
    print(f"🎯 训练阶段3: {steps_per_stage:,}步")
    model.learn(total_timesteps=steps_per_stage, progress_bar=True, reset_num_timesteps=False)
    print("✅ 阶段3完成\n")
    
    model_path = os.path.join(save_path, f"{algorithm}_curriculum_final")
    model.save(model_path)
    env_stage3.save(os.path.join(save_path, f"{algorithm}_curriculum_vecnormalize.pkl"))
    
    print(f"💾 课程学习模型已保存: {model_path}.zip\n")
    
    eval_env = DummyVecEnv([make_env_stage3(0)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = env_stage3.obs_rms
    
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    
    print("="*70)
    print("📊 课程学习最终评估")
    print("="*70)
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print("="*70 + "\n")
    
    env_stage1.close()
    env_stage2.close()
    env_stage3.close()
    eval_env.close()
    
    return model, mean_reward, std_reward, exp_dir


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha机械臂训练 v6 - 漂移目标版本')
    parser.add_argument('--mode', choices=['train', 'test', 'compare', 'curriculum'], 
                       default='train', help='运行模式')
    parser.add_argument('--algorithm', choices=['SAC', 'PPO', 'TD3'], 
                       default='SAC', help='RL算法')
    parser.add_argument('--timesteps', type=int, default=500000, 
                       help='训练步数')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='并行环境数量')
    parser.add_argument('--model', type=str, 
                       help='测试模式下的模型路径')
    parser.add_argument('--vecnorm', type=str,
                       help='测试模式下的VecNormalize参数路径')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='测试回合数')
    parser.add_argument('--no-gif', action='store_true',
                       help='训练后不生成GIF')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🤖 Alpha机械臂强化学习训练系统 v6")
    print("="*70)
    print("新特性:")
    print("  ✅ 目标漂移（5cm内随机摆动）")
    print("  ✅ 速度估计（从历史位置计算）")
    print("  ✅ 无夹爪控制（始终张开）")
    print("  ✅ 优化超参数（针对动态目标）")
    print("  ✅ 17维观察空间（含速度信息）")
    print("  ✅ 4维动作空间（4个关节）")
    print("  ✅ 自动GIF生成（训练完成后）")
    print("="*70 + "\n")
    
    if args.mode == 'train':
        print("🎯 模式: 单算法训练")
        train_alpha_reach(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            auto_visualize=not args.no_gif
        )
        
    elif args.mode == 'test':
        print("🧪 模式: 测试训练好的模型")
        if not args.model:
            model_path = f"./models/{args.algorithm}_final"
            if not os.path.exists(model_path + '.zip'):
                print(f"❌ 错误: 找不到模型文件 {model_path}")
                print(f"请先训练模型或指定正确的模型路径")
                return
        else:
            model_path = args.model
        
        vecnorm_path = args.vecnorm if args.vecnorm else model_path.replace('_final', '_vecnormalize.pkl')
            
        test_trained_model(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            algorithm=args.algorithm,
            n_episodes=args.episodes
        )
        
    elif args.mode == 'compare':
        print("🔬 模式: 算法性能比较")
        results = compare_algorithms(
            algorithms=['SAC', 'PPO', 'TD3'],
            timesteps=args.timesteps
        )
    
    elif args.mode == 'curriculum':
        print("🎓 模式: 课程学习训练")
        train_with_curriculum(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps
        )


if __name__ == "__main__":
    main()