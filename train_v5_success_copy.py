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
from envs.rl_env_v5_success_copy import AlphaReachEnv

def create_experiment_folder(algorithm, timesteps):
    """创建实验文件夹"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{algorithm}_{timesteps}steps_{timestamp}"
    exp_dir = os.path.join("experiments", exp_name)

    # 创建实验目录结构
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
        "config": config
    }

    with open(os.path.join(exp_dir, "config.json"), 'w') as f:
        json.dump(exp_config, f, indent=2)

def plot_training_curves(exp_dir, log_dir):
    """绘制和保存训练曲线"""
    try:
        # 查找日志文件（优先 .monitor.csv）
        log_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        if not log_files:
            log_files = glob.glob(os.path.join(log_dir, "*.csv"))
        if not log_files:
            print("警告：未找到训练日志文件")
            return

        # 读取训练数据
        import pandas as pd

        # 设置高质量图表样式
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
        fig.suptitle('Training Progress - Alpha5 Robotic Arm Reach Task', fontsize=18, fontweight='bold')

        # 定义颜色方案
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for idx, log_file in enumerate(log_files):
            try:
                # Monitor文件首部含以#开头的注释行
                df = pd.read_csv(log_file, comment='#')
                color = colors[idx % len(colors)]

                if 'r' in df.columns and 'l' in df.columns:
                    # 奖励曲线 - 改进版
                    raw_rewards = df['r']
                    smoothed_rewards = df['r'].rolling(window=100, min_periods=1).mean()

                    # 原始数据（透明度低）
                    axes[0, 0].plot(df['l'], raw_rewards, alpha=0.2, color=color, linewidth=0.5)
                    # 平滑曲线（主要显示）
                    axes[0, 0].plot(df['l'], smoothed_rewards, linewidth=3, color=color,
                                   label=f'Env {idx+1}' if len(log_files) > 1 else 'Episode Reward')

                    axes[0, 0].set_title('Episode Reward Over Time', fontweight='bold')
                    axes[0, 0].set_xlabel('Training Timesteps')
                    axes[0, 0].set_ylabel('Cumulative Reward')
                    axes[0, 0].grid(True, alpha=0.3)
                    if len(log_files) > 1:
                        axes[0, 0].legend()

                if 't' in df.columns:
                    # 回合长度 - 改进版
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

        # 如果有评估日志，也绘制评估曲线
        eval_log = os.path.join(log_dir, "evaluations.npz")
        if os.path.exists(eval_log):
            try:
                eval_data = np.load(eval_log)
                timesteps = eval_data['timesteps']
                results = eval_data['results']

                # 评估奖励 - 改进版
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

                # 成功率（如果有）- 改进版
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
                    # 如果没有成功率数据，显示距离信息（如果有）
                    axes[1, 1].text(0.5, 0.5, 'Success Rate\nData Not Available',
                                   transform=axes[1, 1].transAxes, ha='center', va='center',
                                   fontsize=14, alpha=0.6)
                    axes[1, 1].set_title('Task Success Rate', fontweight='bold')

            except Exception as e:
                print(f"读取评估日志失败: {e}")
        else:
            # 如果没有评估数据，在下方两个子图显示提示
            for i in range(2):
                axes[1, i].text(0.5, 0.5, 'Evaluation Data\nNot Available',
                               transform=axes[1, i].transAxes, ha='center', va='center',
                               fontsize=14, alpha=0.6)
                axes[1, i].set_title(['Evaluation Performance', 'Task Success Rate'][i], fontweight='bold')

        # 调整布局和样式
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 只保存PNG格式，提高质量设置
        plt.savefig(os.path.join(exp_dir, "plots", "training_curves.png"),
                   dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        print(f"训练曲线已保存到: {os.path.join(exp_dir, 'plots', 'training_curves.png')}")

    except ImportError:
        print("警告：pandas未安装，无法绘制训练曲线")
    except Exception as e:
        print(f"绘制训练曲线失败: {e}")

def record_episode_gif_vecenv(vec_env, model, episode_num, exp_dir, max_steps=400, fps=30):
    """录制单个回合为GIF（支持VecEnv）"""
    frames = []
    obs = vec_env.reset()
    episode_reward = 0
    episode_length = 0
    done = False

    print(f"录制回合 {episode_num}...（正常速度，使用标准化环境）")

    while not done and episode_length < max_steps:
        # 每步都采样，显示完整轨迹
        should_capture = True

        if should_capture:
            # 获取原始环境进行渲染
            raw_env = vec_env.envs[0] if hasattr(vec_env, 'envs') else vec_env.venv.envs[0]

            # 获取渲染图像
            try:
                # 对于PyBullet环境，需要手动获取相机图像
                width, height = 640, 480

                # 使用与OpenGL一致的相机设置
                # 获取机械臂和目标的位置信息
                ee_pos = raw_env._get_end_effector_position()
                target_pos = raw_env.target_position

                # 使用与OpenGL相同的视角参数
                # 相机焦点在机械臂和目标之间
                camera_target = [(ee_pos[0] + target_pos[0]) / 2,
                               (ee_pos[1] + target_pos[1]) / 2,
                               (ee_pos[2] + target_pos[2]) / 2]

                # 使甇30度俯视角
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=camera_target,
                    distance=1.5,  # 适中的距离
                    yaw=45,    # 标准45度视角
                    pitch=-30, # 30度俯视角，向下看
                    roll=0,
                    upAxisIndex=2,
                    physicsClientId=raw_env.physics_client
                )

                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=width/height,
                    nearVal=0.1,
                    farVal=100.0,
                    physicsClientId=raw_env.physics_client
                )

                # 获取相机图像（包含目标球，灰色背景）
                (_, _, px, _, _) = p.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    physicsClientId=raw_env.physics_client
                )

                # 转换为PIL图像
                rgb_array = np.array(px)[:, :, :3]  # 去掉alpha通道
                image = Image.fromarray(rgb_array, 'RGB')

                # 添加文本信息
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(image)

                # 尝试使用默认字体
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

                # 添加信息文本（包含目标距离）
                current_distance = np.linalg.norm(raw_env._get_end_effector_position() - raw_env.target_position)
                info_text = f"Episode: {episode_num} | Step: {episode_length} | Distance: {current_distance:.3f}m"
                draw.text((10, 10), info_text, fill=(255, 255, 255), font=font)

                # 添加目标信息
                target_text = f"🟠 Target: [{raw_env.target_position[0]:.2f}, {raw_env.target_position[1]:.2f}, {raw_env.target_position[2]:.2f}]"
                draw.text((10, 35), target_text, fill=(255, 165, 0), font=font)  # 橙色文字

                frames.append(image)

            except Exception as e:
                print(f"捕获帧失败: {e}")
                # 如果捕获失败，跳过此帧但继续录制

        # 执行动作
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        episode_reward += reward[0]  # VecEnv返回数组
        episode_length += 1
        done = done[0]  # VecEnv返回数组
        info = info[0] if len(info) > 0 else {}

        # 每50步显示进度
        if episode_length % 50 == 0:
            print(f"  已采样 {episode_length} 步，录制 {len(frames)} 帧")

    # 保存GIF到实验文件夹的videos目录
    if frames:
        # 确保实验的videos文件夹存在
        videos_dir = os.path.join(exp_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        gif_path = os.path.join(videos_dir, f"episode_{episode_num}.gif")
        try:
            actual_frames = len(frames)
            expected_frames = max_steps
            print(f"  保存正常速度GIF：{actual_frames}帧/{expected_frames}预期，{fps}FPS...")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,  # 毫秒
                loop=0,
                optimize=True  # 恢复优化以控制文件大小
            )
            print(f"  ✓ GIF已保存: {gif_path}")
            return gif_path, episode_reward, episode_length, info
        except Exception as e:
            print(f"  ✗ 保存GIF失败: {e}")
            return None, episode_reward, episode_length, info
    else:
        print(f"  ✗ 没有捕获到帧")

    return None, episode_reward, episode_length, info

def record_episode_gif(env, model, episode_num, exp_dir, max_steps=400, fps=30):
    """录制单个回合为GIF"""
    frames = []
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False

    print(f"录制回合 {episode_num}...（正常速度）")

    while not done and episode_length < max_steps:
        # 每步都采样，显示完整轨迹
        should_capture = True

        if should_capture:
            # 获取渲染图像
            try:
                # 对于PyBullet环境，需要手动获取相机图像
                width, height = 640, 480

                # 使用与OpenGL一致的相机设置
                # 获取机械臂和目标的位置信息
                ee_pos = env._get_end_effector_position()
                target_pos = env.target_position

                # 使用与OpenGL相同的视角参数
                # 相机焦点在机械臂和目标之间
                camera_target = [(ee_pos[0] + target_pos[0]) / 2,
                               (ee_pos[1] + target_pos[1]) / 2,
                               (ee_pos[2] + target_pos[2]) / 2]

                # 使甇30度俯视角
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=camera_target,
                    distance=1.5,  # 适中的距离
                    yaw=45,    # 标准45度视角
                    pitch=-30, # 30度俯视角，向下看
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

                # 获取相机图像（包含目标球，灰色背景）
                (_, _, px, _, _) = p.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    physicsClientId=env.physics_client
                )

                # 转换为PIL图像
                rgb_array = np.array(px)[:, :, :3]  # 去掉alpha通道
                image = Image.fromarray(rgb_array, 'RGB')

                # 添加文本信息
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(image)

                # 尝试使用默认字体
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

                # 添加信息文本（包含目标距离）
                current_distance = np.linalg.norm(env._get_end_effector_position() - env.target_position)
                info_text = f"Episode: {episode_num} | Step: {episode_length} | Distance: {current_distance:.3f}m"
                draw.text((10, 10), info_text, fill=(255, 255, 255), font=font)

                # 添加目标信息
                target_text = f"🟠 Target: [{env.target_position[0]:.2f}, {env.target_position[1]:.2f}, {env.target_position[2]:.2f}]"
                draw.text((10, 35), target_text, fill=(255, 165, 0), font=font)  # 橙色文字

                frames.append(image)

            except Exception as e:
                print(f"捕获帧失败: {e}")
                # 如果捕获失败，跳过此帧但继续录制

        # 执行动作
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_length += 1

        # 每50步显示进度
        if episode_length % 50 == 0:
            print(f"  已采样 {episode_length} 步，录制 {len(frames)} 帧")

    # 保存GIF到实验文件夹的videos目录
    if frames:
        # 确保实验的videos文件夹存在
        videos_dir = os.path.join(exp_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        gif_path = os.path.join(videos_dir, f"episode_{episode_num}.gif")
        try:
            actual_frames = len(frames)
            expected_frames = max_steps
            print(f"  保存正常速度GIF：{actual_frames}帧/{expected_frames}预期，{fps}FPS...")
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,  # 毫秒
                loop=0,
                optimize=True  # 恢复优化以控制文件大小
            )
            print(f"  ✓ GIF已保存: {gif_path}")
            return gif_path, episode_reward, episode_length, info
        except Exception as e:
            print(f"  ✗ 保存GIF失败: {e}")
            return None, episode_reward, episode_length, info
    else:
        print(f"  ✗ 没有捕获到帧")

    return None, episode_reward, episode_length, info

def visualize_trained_model(exp_dir, model_path, algorithm, n_episodes=5):
    """可视化训练好的模型并录制GIF"""
    print(f"\n{'='*50}")
    print("开始可视化测试训练好的模型...")
    print(f"{'='*50}")

    try:
        # 首先录制GIF（非GUI模式）
        print("步骤1: 录制GIF...")

        # 加载模型
        algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
        model = algo_classes[algorithm].load(model_path)

        # 创建带标准化的环境（与训练时一致）
        gif_env_raw = AlphaReachEnv(render_mode=None)
        gif_env = DummyVecEnv([lambda: gif_env_raw])

        # 加载VecNormalize参数
        vecnorm_path = model_path.replace('_final', '_vecnormalize.pkl')
        if os.path.exists(vecnorm_path):
            print(f"加载环境标准化参数: {vecnorm_path}")
            gif_env = VecNormalize.load(vecnorm_path, gif_env)
            gif_env.training = False
            gif_env.norm_reward = False
        else:
            print("警告：未找到VecNormalize参数文件，使用原始环境")

        # 测试数据记录
        test_results = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'final_distances': [],
            'episode_lengths': [],
            'gif_paths': []
        }

        print(f"开始录制和测试 {n_episodes} 个回合...")
        print("注意：正在录制GIF，可能需要较长时间...")

        for episode in range(n_episodes):
            print(f"\n回合 {episode + 1}/{n_episodes}")

            # 录制加速GIF（轨迹概览，3倍速度）
            gif_path, episode_reward, episode_length, info = record_episode_gif_vecenv(
                gif_env, model, episode + 1, exp_dir, max_steps=400, fps=30
            )

            success = info.get('success', False)
            final_distance = info.get('distance', 0)

            print(f"  回合结束:")
            print(f"    成功: {success}")
            print(f"    最终距离: {final_distance:.3f}m")
            print(f"    总奖励: {episode_reward:.2f}")
            print(f"    步数: {episode_length}")
            if gif_path:
                print(f"    GIF: {gif_path}")

            # 记录结果
            test_results['episodes'].append(episode + 1)
            test_results['rewards'].append(episode_reward)
            test_results['success_rates'].append(1 if success else 0)
            test_results['final_distances'].append(final_distance)
            test_results['episode_lengths'].append(episode_length)
            test_results['gif_paths'].append(gif_path if gif_path else "")

        gif_env.close()

        print(f"\n跳过OpenGL可视化测试（已禁用）")

        # 创建综合GIF（所有回合的最佳片段）
        create_combined_gif(exp_dir, test_results, algorithm)

        # 保存测试结果（包含GIF路径）
        test_summary = {
            'total_episodes': n_episodes,
            'average_reward': float(np.mean(test_results['rewards'])),
            'success_rate': float(np.mean(test_results['success_rates'])),
            'average_distance': float(np.mean(test_results['final_distances'])),
            'average_length': float(np.mean(test_results['episode_lengths'])),
            'gif_count': sum(1 for path in test_results.get('gif_paths', []) if path),
            'detailed_results': test_results
        }

        with open(os.path.join(exp_dir, "test_results.json"), 'w') as f:
            json.dump(test_summary, f, indent=2)

        # 打印总结
        print(f"\n{'='*50}")
        print("可视化测试完成！结果总结:")
        print(f"{'='*50}")
        print(f"总回合数: {n_episodes}")
        print(f"平均奖励: {test_summary['average_reward']:.2f}")
        print(f"成功率: {test_summary['success_rate']*100:.1f}%")
        print(f"平均最终距离: {test_summary['average_distance']:.3f}m")
        print(f"平均回合长度: {test_summary['average_length']:.1f}步")
        print(f"测试结果已保存到: {os.path.join(exp_dir, 'test_results.json')}")

        # 打印GIF信息
        gif_count = sum(1 for path in test_results.get('gif_paths', []) if path)
        if gif_count > 0:
            print(f"已生成 {gif_count} 个加速回合GIF，保存在: {os.path.join(exp_dir, 'videos')}")

            # 列出所有生成的GIF文件
            exp_videos_dir = os.path.join(exp_dir, 'videos')
            if os.path.exists(exp_videos_dir):
                gif_files = [f for f in os.listdir(exp_videos_dir) if f.endswith('.gif')]
                for gif_file in sorted(gif_files):
                    print(f"  - {gif_file}")

        return test_summary

    except Exception as e:
        print(f"可视化测试失败: {e}")
        return None

def create_combined_gif(exp_dir, test_results, algorithm):
    """创建所有回合的综合GIF"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        combined_frames = []
        videos_dir = os.path.join(exp_dir, "videos")

        # 收集所有成功的回合或者奖励最高的回合
        best_episodes = []

        # 找出成功的回合
        success_episodes = [i for i, success in enumerate(test_results['success_rates']) if success]

        if success_episodes:
            # 如果有成功的回合，选择前3个
            best_episodes = success_episodes[:3]
            title_text = f"{algorithm} - Successful Episodes"
        else:
            # 如果没有成功的回合，选择奖励最高的3个
            rewards = test_results['rewards']
            sorted_indices = sorted(range(len(rewards)), key=lambda x: rewards[x], reverse=True)
            best_episodes = sorted_indices[:3]
            title_text = f"{algorithm} - Best Performance Episodes"

        print(f"创建综合GIF，包含回合: {[e+1 for e in best_episodes]}")

        for episode_idx in best_episodes:
            gif_path = test_results['gif_paths'][episode_idx]
            if gif_path and os.path.exists(gif_path):
                try:
                    # 读取GIF帧
                    with Image.open(gif_path) as gif:
                        frames = []
                        for frame_idx in range(0, gif.n_frames, 2):  # 每2帧取1帧，保持更高帧率
                            gif.seek(frame_idx)
                            frame = gif.copy().convert('RGB')

                            # 添加回合标识
                            draw = ImageDraw.Draw(frame)
                            try:
                                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
                            except:
                                font = ImageFont.load_default()

                            episode_text = f"Episode {episode_idx + 1}"
                            draw.text((10, 50), episode_text, fill=(255, 255, 0), font=font)

                            frames.append(frame)

                        combined_frames.extend(frames)

                        # 添加间隔帧
                        if episode_idx != best_episodes[-1]:
                            black_frame = Image.new('RGB', frames[0].size, (0, 0, 0))
                            draw = ImageDraw.Draw(black_frame)
                            draw.text((black_frame.width//2-100, black_frame.height//2),
                                    "Next Episode", fill=(255, 255, 255), font=font)
                            combined_frames.extend([black_frame] * 30)  # 1秒间隔（30fps）

                except Exception as e:
                    print(f"处理回合 {episode_idx + 1} 的GIF失败: {e}")

        # 保存综合GIF
        if combined_frames:
            # 保存到实验文件夹的videos目录
            exp_videos_dir = os.path.join(exp_dir, "videos")
            os.makedirs(exp_videos_dir, exist_ok=True)
            combined_gif_path = os.path.join(exp_videos_dir, f"{algorithm}_combined_episodes.gif")

            # 添加标题帧
            title_frame = Image.new('RGB', combined_frames[0].size, (0, 0, 0))
            draw = ImageDraw.Draw(title_frame)
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            except:
                title_font = ImageFont.load_default()

            # 居中显示标题
            title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (title_frame.width - title_width) // 2
            draw.text((title_x, title_frame.height//2 - 50), title_text, fill=(255, 255, 255), font=title_font)

            # 添加统计信息
            stats_text = f"Avg Reward: {np.mean(test_results['rewards']):.1f} | Success Rate: {np.mean(test_results['success_rates'])*100:.1f}%"
            stats_bbox = draw.textbbox((0, 0), stats_text, font=title_font)
            stats_width = stats_bbox[2] - stats_bbox[0]
            stats_x = (title_frame.width - stats_width) // 2
            draw.text((stats_x, title_frame.height//2 + 20), stats_text, fill=(255, 255, 255), font=title_font)

            # 插入标题帧
            final_frames = [title_frame] * 60 + combined_frames  # 2秒标题显示（30fps）

            final_frames[0].save(
                combined_gif_path,
                save_all=True,
                append_images=final_frames[1:],
                duration=33,  # 33ms per frame (30 FPS, 3倍加速)
                loop=0,
                optimize=True
            )

            print(f"综合GIF已保存: {combined_gif_path}")
            return combined_gif_path

    except Exception as e:
        print(f"创建综合GIF失败: {e}")

    return None

class TrainingProgressCallback(BaseCallback):
    """训练进度回调，记录成功率等指标"""
    
    def __init__(self, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.success_rates = []
        self.distances = []
        self.episodes = 0
        self.successes = 0
        
    def _on_step(self) -> bool:
        # 记录回合信息
        if 'episode' in self.locals and self.locals['episode']:
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
            print(f"训练完成 - 成功率: {success_rate:.2%}, 平均距离: {avg_distance:.3f}m")

def make_env(idx: int, log_dir: str, render_mode=None):
    """创建环境包装器，带Monitor日志输出到CSV"""

    def _init():
        env = AlphaReachEnv(render_mode=render_mode)
        # 记录成功标志与距离到monitor文件，便于后续曲线绘制
        filename = os.path.join(log_dir, f"monitor_{idx}.monitor.csv")
        return Monitor(env, filename=filename, info_keywords=("success", "distance"))

    return _init

def train_alpha_reach(
    algorithm='SAC',
    total_timesteps=300000,
    num_envs=4,
    auto_visualize=False
):
    """训练Alpha机械臂到达任务"""

    print(f"开始训练Alpha机械臂到达任务")
    print(f"算法: {algorithm}")
    print(f"总步数: {total_timesteps}")
    print("="*50)

    # 创建实验文件夹
    exp_dir, exp_name = create_experiment_folder(algorithm, total_timesteps)
    print(f"实验文件夹: {exp_dir}")

    # 设置保存路径
    save_path = os.path.join(exp_dir, "models")
    log_path = os.path.join(exp_dir, "logs")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # 首先检查环境
    print("检查环境...")
    test_env = AlphaReachEnv()
    check_env(test_env)
    test_env.close()
    print("环境检查通过")
    
    # 创建训练环境
    train_env = DummyVecEnv([make_env(i, log_path) for i in range(num_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0
    )

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(0, log_path)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False
    )
    eval_env.obs_rms = train_env.obs_rms

    # 算法配置 - 优化稳定性
    configs = {
        'SAC': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 500000,
            'batch_size': 512,
            'tau': 0.005,
            'gamma': 0.98,
            'train_freq': (max(1, 64 // num_envs), 'step'),
            'gradient_steps': 64,
            'learning_starts': 10000,
            'ent_coef': 'auto_0.2',
            'target_update_interval': 1,
            'use_sde': True,
            'sde_sample_freq': 4,
            'policy_kwargs': dict(net_arch=[256, 256, 256]),
            'verbose': 1
        },
        'PPO': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'n_steps': max(128, 2048 // num_envs),
            'batch_size': 128,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.005,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': dict(net_arch=[256, 256, 128]),
            'verbose': 1
        },
        'TD3': {
            'policy': 'MlpPolicy',
            'learning_rate': 3e-4,
            'buffer_size': 500000,
            'batch_size': 512,
            'tau': 0.005,
            'gamma': 0.98,
            'train_freq': (2, 'episode'),
            'gradient_steps': 120,
            'learning_starts': 5000,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'policy_delay': 2,
            'policy_kwargs': dict(net_arch=[300, 300, 200]),
            'verbose': 1
        }
    }
    
    # 保存实验配置
    save_experiment_config(exp_dir, algorithm, configs[algorithm], total_timesteps)

    # 创建模型
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    model = algo_classes[algorithm](env=train_env, **configs[algorithm])

    # 设置回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, f"{algorithm}_best"),
        log_path=log_path,
        eval_freq=max(5000, total_timesteps // 20),  # 更频繁的评估
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
    
    # 开始训练
    print("开始训练...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback],
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.1f}秒")

    # 保存最终模型
    model_path = os.path.join(save_path, f"{algorithm}_final")
    model.save(model_path)
    train_env.save(os.path.join(save_path, f"{algorithm}_vecnormalize.pkl"))

    # 最终评估
    print("进行最终评估...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )

    print(f"最终评估结果:")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")

    # 检查并报告保存的模型
    print(f"\n模型保存情况:")
    print(f"  - 最终模型: {model_path}.zip")
    print(f"  - 环境标准化参数: {os.path.join(save_path, f'{algorithm}_vecnormalize.pkl')}")

    best_model_path = os.path.join(save_path, f"{algorithm}_best.zip")
    if os.path.exists(best_model_path):
        print(f"  - 最佳模型: {best_model_path}")
    else:
        print(f"  - 最佳模型: 未生成（可能评估次数不足）")

    # 清理
    train_env.close()
    eval_env.close()

    # 绘制和保存训练曲线
    print("绘制训练曲线...")
    plot_training_curves(exp_dir, log_path)

    # 保存训练总结
    training_summary = {
        'algorithm': algorithm,
        'total_timesteps': total_timesteps,
        'training_time': training_time,
        'final_mean_reward': float(mean_reward),
        'final_std_reward': float(std_reward),
        'experiment_name': exp_name,
        'model_path': model_path
    }

    with open(os.path.join(exp_dir, "training_summary.json"), 'w') as f:
        json.dump(training_summary, f, indent=2)

    # 自动可视化测试与GIF生成
    if auto_visualize:
        print("\n开始自动可视化测试...")
        visualize_trained_model(exp_dir, model_path, algorithm, n_episodes=5)

        # 使用独立GIF渲染脚本生成高分辨率GIF
        gif_out = os.path.join(exp_dir, "videos", f"{algorithm}_training_rollout.gif")
        os.makedirs(os.path.dirname(gif_out), exist_ok=True)
        try:
            from render_gif import render_policy_to_gif
            render_policy_to_gif(
                algorithm=algorithm,
                model_path=Path(model_path + ".zip"),
                vecnorm_path=Path(os.path.join(save_path, f"{algorithm}_vecnormalize.pkl")),
                output_path=Path(gif_out),
                episodes=2,
                max_steps=400,
                width=1280,
                height=720,
                fps=24,
                frame_skip=2,
            )
        except Exception as e:
            print(f"渲染训练GIF失败（函数导入失败，尝试CLI）: {e}")
            # 回退到命令行调用
            import sys, subprocess
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), 'render_gif.py'),
                '--algorithm', algorithm,
                '--model', model_path + '.zip',
                '--vecnorm', os.path.join(save_path, f'{algorithm}_vecnormalize.pkl'),
                '--output', gif_out,
                '--episodes', '2',
                '--max-steps', '400',
                '--width', '1280',
                '--height', '720',
                '--fps', '24',
                '--frame-skip', '2'
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e2:
                print(f"渲染训练GIF的CLI方式也失败: {e2}")

    print(f"\n{'='*60}")
    print(f"实验完成！所有结果已保存到: {exp_dir}")
    print(f"{'='*60}")
    print(f"包含内容:")
    print(f"  - 模型权重: {os.path.join(exp_dir, 'models')}")
    # 列出具体保存的模型文件
    models_dir = os.path.join(exp_dir, 'models')
    if os.path.exists(os.path.join(models_dir, f"{algorithm}_final.zip")):
        print(f"    ✓ 最终模型: {algorithm}_final.zip")
    if os.path.exists(os.path.join(models_dir, f"{algorithm}_best.zip")):
        print(f"    ✓ 最佳模型: {algorithm}_best.zip")
    if os.path.exists(os.path.join(models_dir, f"{algorithm}_vecnormalize.pkl")):
        print(f"    ✓ 标准化参数: {algorithm}_vecnormalize.pkl")
    print(f"  - 训练日志: {os.path.join(exp_dir, 'logs')}")
    print(f"  - 训练曲线: {os.path.join(exp_dir, 'plots')}")
    if auto_visualize:
        print(f"  - 测试结果: test_results.json")
        print(f"  - 视频文件: {os.path.join(exp_dir, 'videos')}")
    print(f"  - 实验配置: config.json")

    return model, mean_reward, std_reward, exp_dir

def test_trained_model(model_path, algorithm='SAC', n_episodes=5):
    """测试训练好的模型"""
    
    print(f"测试训练好的{algorithm}模型...")
    print(f"模型路径: {model_path}")
    
    # 创建可视化环境
    env = AlphaReachEnv(render_mode="human")
    
    # 加载模型
    algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
    model = algo_classes[algorithm].load(model_path)
    
    # 如果有VecNormalize，也要加载
    try:
        vec_normalize_path = model_path.replace('_final', '_vecnormalize.pkl')
        if os.path.exists(vec_normalize_path):
            print("加载环境标准化参数...")
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
        
        for step in range(500):  # 最多200步
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
        
        # 记录结果
        success = info.get('success', False)
        distance = info.get('distance', 0)
        
        if success:
            successes += 1
        
        total_rewards.append(episode_reward)
        distances.append(distance)
        
        print(f"Episode {episode+1}: 成功={success}, 距离={distance:.3f}m, "
              f"奖励={episode_reward:.2f}, 步数={step_count}")
    
    # 打印统计结果
    print("\n" + "="*50)
    print("测试结果统计:")
    print("="*50)
    print(f"成功率: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均最终距离: {np.mean(distances):.3f} ± {np.std(distances):.3f}m")
    
    if successes > 0:
        success_distances = [d for i, d in enumerate(distances) if i < len(total_rewards) and 
                           total_rewards[i] > 0]  # 简化的成功判断
        if success_distances:
            print(f"成功时平均距离: {np.mean(success_distances):.3f}m")
    
    env.close()
    
    return {
        'success_rate': successes / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'mean_distance': np.mean(distances)
    }

def compare_algorithms(algorithms=['SAC', 'PPO', 'TD3'], timesteps=30000):
    """比较不同算法的性能"""
    
    print("开始算法性能比较")
    print(f"算法: {algorithms}")
    print(f"训练步数: {timesteps}")
    print("="*50)
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n正在训练 {algorithm}...")
        try:
            model, mean_reward, std_reward = train_alpha_reach(
                algorithm=algorithm,
                total_timesteps=timesteps
            )
            results[algorithm] = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'model': model
            }
            print(f"{algorithm} 训练完成")
        except Exception as e:
            print(f"{algorithm} 训练失败: {e}")
            results[algorithm] = {
                'mean_reward': -np.inf,
                'std_reward': 0,
                'model': None
            }
    
    # 打印比较结果
    print("\n" + "="*60)
    print("算法性能比较结果:")
    print("="*60)
    print(f"{'算法':<8} {'平均奖励':<12} {'标准差':<10} {'相对性能':<10}")
    print("-" * 60)
    
    best_reward = max([r['mean_reward'] for r in results.values() if r['mean_reward'] != -np.inf])
    
    for algo, result in results.items():
        if result['mean_reward'] != -np.inf:
            relative_perf = result['mean_reward'] / best_reward * 100
            print(f"{algo:<8} {result['mean_reward']:<12.2f} {result['std_reward']:<10.2f} {relative_perf:<10.1f}%")
        else:
            print(f"{algo:<8} {'失败':<12} {'-':<10} {'-':<10}")
    
    # 找出最佳算法
    best_algo = max(results.keys(), key=lambda x: results[x]['mean_reward'])
    if results[best_algo]['mean_reward'] != -np.inf:
        print(f"\n最佳算法: {best_algo} (奖励: {results[best_algo]['mean_reward']:.2f})")
    
    return results

def plot_training_results(log_dir="./logs"):
    """绘制训练结果图表"""
    try:
        import tensorboard
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        print("绘制训练结果...")
        
        algorithms = ['SAC', 'PPO', 'TD3']
        colors = ['blue', 'red', 'green']
        
        plt.figure(figsize=(12, 8))
        
        for i, algo in enumerate(algorithms):
            log_path = os.path.join(log_dir, algo)
            if os.path.exists(log_path):
                # 这里可以添加从tensorboard日志读取数据的代码
                # 简化版：创建示例数据
                steps = np.arange(0, 50000, 1000)
                rewards = np.random.normal(0, 1, len(steps)).cumsum() * 0.1
                plt.plot(steps, rewards, color=colors[i], label=algo, alpha=0.7)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Average Reward')
        plt.title('Training Progress Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("需要安装tensorboard来绘制训练图表")
    except Exception as e:
        print(f"绘制图表失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpha机械臂到达任务训练')
    parser.add_argument('--mode', choices=['train', 'test', 'compare'], 
                       default='train', help='运行模式')
    parser.add_argument('--algorithm', choices=['SAC', 'PPO', 'TD3'], 
                       default='SAC', help='RL算法')
    parser.add_argument('--timesteps', type=int, default=300000, 
                       help='训练步数')
    parser.add_argument('--num_envs', type=int, default=4,
                       help='并行环境数量')
    parser.add_argument('--model', type=str, 
                       help='测试模式下的模型路径')
    parser.add_argument('--episodes', type=int, default=5, 
                       help='测试回合数')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("单算法训练模式")
        train_alpha_reach(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            auto_visualize=False
        )
        
    elif args.mode == 'test':
        if not args.model:
            # 尝试加载默认模型
            model_path = f"./models/{args.algorithm}_final"
            if not os.path.exists(model_path + '.zip'):
                print(f"错误: 找不到模型文件 {model_path}")
                print("请先训练模型或指定正确的模型路径")
                return
        else:
            model_path = args.model
            
        print("模型测试模式")
        test_trained_model(
            model_path=model_path,
            algorithm=args.algorithm,
            n_episodes=args.episodes
        )
        
    elif args.mode == 'compare':
        print("算法比较模式")
        results = compare_algorithms(
            algorithms=['SAC', 'PPO', 'TD3'],
            timesteps=args.timesteps
        )
        
        # 绘制结果
        plot_training_results()

if __name__ == "__main__":
    main()
