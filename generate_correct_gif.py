import os
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
import sys

# 设置matplotlib配置目录
default_mpl_dir = os.path.join(os.path.dirname(__file__), '.mplconfig')
os.environ.setdefault('MPLCONFIGDIR', default_mpl_dir)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

from stable_baselines3 import SAC
from sb3_contrib import TQC, CrossQ
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.rl_env_v7 import AlphaReachEnv


def generate_gif_corrected(model_path, vecnorm_path, output_path, algorithm='CrossQ', 
                          n_episodes=3, max_steps=500, fps=30):
    """
    生成正确的GIF，使用4cm成功阈值
    
    参数:
        model_path: 模型路径
        vecnorm_path: VecNormalize参数路径
        output_path: 输出目录
        algorithm: 算法名称 ('SAC', 'TQC', 'CrossQ')
        n_episodes: 生成的回合数
        max_steps: 每回合最大步数
        fps: 帧率
    """
    
    print("="*70)
    print(f"🎬 正在生成正确的GIF (成功阈值: 4cm)")
    print(f"   模型: {model_path}")
    print(f"   算法: {algorithm}")
    print(f"   回合数: {n_episodes}")
    print("="*70)
    
    # 加载模型
    algo_classes = {'SAC': SAC, 'TQC': TQC, 'CrossQ': CrossQ}
    if algorithm not in algo_classes:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    model = algo_classes[algorithm].load(model_path)
    
    # 创建环境 - 使用stage4配置
    stage4_config = {
        'enable_target_drift': True,
        'enable_domain_randomization': True,
        'enable_curriculum': True,
        'enable_sensor_noise': True,
    }
    
    test_env = AlphaReachEnv(render_mode=None, **stage4_config)
    vec_env = DummyVecEnv([lambda: test_env])
    
    # 加载VecNormalize参数
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("✅ 已加载VecNormalize参数")
    else:
        print("⚠️ 未找到VecNormalize参数文件")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 生成每个回合的GIF
    gif_paths = []
    for episode_num in range(1, n_episodes + 1):
        gif_path = _generate_single_episode_gif(
            model, vec_env, output_path, episode_num, max_steps, fps
        )
        if gif_path:
            gif_paths.append(gif_path)
    
    vec_env.close()
    
    print(f"\n✅ 成功生成 {len(gif_paths)} 个GIF:")
    for path in gif_paths:
        print(f"   📁 {path}")
    
    return gif_paths


def _generate_single_episode_gif(model, env, output_dir, episode_num, max_steps, fps):
    """生成单个回合的GIF"""
    
    frames = []
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    episode_reward = 0
    episode_length = 0
    success_achieved = False
    
    # ✅ 正确的成功阈值：4cm
    CORRECT_SUCCESS_THRESHOLD = 0.04
    
    print(f"\n🎬 录制回合 {episode_num}...")
    
    # 获取环境引用
    if hasattr(env, 'envs'):
        raw_env = env.envs[0]
        if hasattr(raw_env, 'env'):
            raw_env = raw_env.env
    elif hasattr(env, 'venv'):
        raw_env = env.venv.envs[0]
        if hasattr(raw_env, 'env'):
            raw_env = raw_env.env
    else:
        raw_env = env
    
    for step in range(max_steps):
        # ✅ 先执行动作（如果不是第一步）
        if step > 0:
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
        
        try:
            width, height = 640, 480
            
            # ✅ 获取最新状态（在动作执行后）
            ee_pos = raw_env._get_end_effector_position()
            target_pos = raw_env.target_position
            initial_target = getattr(raw_env, 'initial_target_position', target_pos)
            
            distance_to_target = np.linalg.norm(ee_pos - target_pos)
            target_drift = np.linalg.norm(target_pos - initial_target)
            
            # ✅ 检查是否成功（使用当前距离）
            if distance_to_target < CORRECT_SUCCESS_THRESHOLD:
                success_achieved = True
            
            # 获取课程阶段
            curriculum_stage = getattr(raw_env, 'curriculum_stage', 
                                      getattr(raw_env, 'curriculum_level', 0))
            
            # 设置相机
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
            
            # 获取图像
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
            
            # 添加信息叠加层
            draw = ImageDraw.Draw(image)
            
            try:
                font_large = ImageFont.truetype("arial.ttf", 18)
                font_small = ImageFont.truetype("arial.ttf", 14)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            
            # 创建半透明背景
            from PIL import Image as PILImage
            overlay = PILImage.new('RGBA', image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # 背景矩形
            overlay_draw.rectangle([(0, 0), (width, 120)], fill=(0, 0, 0, 180))
            overlay_draw.rectangle([(0, height-180), (350, height)], fill=(0, 0, 0, 180))
            overlay_draw.rectangle([(width-350, height-120), (width, height)], fill=(0, 0, 0, 180))
            
            image = PILImage.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(image)
            
            # ✅ 使用正确的4cm阈值判断成功
            is_success = distance_to_target < CORRECT_SUCCESS_THRESHOLD
            if is_success:
                success_achieved = True
            
            # 顶部信息
            y_offset = 10
            title = f"Episode {episode_num} - Step {step}/{max_steps}"
            draw.text((10, y_offset), title, fill=(255, 255, 0), font=font_large)
            y_offset += 25
            
            # 成功状态显示
            success_text = "✅ SUCCESS!" if success_achieved else f"Target: {distance_to_target:.3f}m"
            success_color = (0, 255, 0) if success_achieved else (255, 255, 255)
            reward_text = f"Reward: {episode_reward:.1f} | {success_text}"
            draw.text((10, y_offset), reward_text, fill=success_color, font=font_small)
            y_offset += 22
            
            # ✅ 显示正确的4cm阈值
            distance_color = (0, 255, 0) if distance_to_target < CORRECT_SUCCESS_THRESHOLD else \
                           (255, 165, 0) if distance_to_target < CORRECT_SUCCESS_THRESHOLD * 2 else \
                           (255, 255, 255)
            distance_text = f"Distance: {distance_to_target*100:.1f}cm / Threshold: {CORRECT_SUCCESS_THRESHOLD*100:.0f}cm"
            draw.text((10, y_offset), distance_text, fill=distance_color, font=font_small)
            y_offset += 22
            
            # 课程阶段
            stage_colors = [(100, 200, 255), (255, 200, 100), (255, 100, 100)]
            stage_names = ["Easy", "Medium", "Hard"]
            stage_color = stage_colors[min(curriculum_stage, 2)]
            stage_text = f"Stage: {curriculum_stage} ({stage_names[min(curriculum_stage, 2)]})"
            draw.text((10, y_offset), stage_text, fill=stage_color, font=font_small)
            
            # 左下角：末端执行器信息
            y_offset = height - 170
            draw.text((10, y_offset), "End Effector (TCP):", fill=(100, 200, 255), font=font_small)
            y_offset += 20
            draw.text((10, y_offset), f"  X: {ee_pos[0]:+.3f}m", fill=(255, 255, 255), font=font_small)
            y_offset += 20
            draw.text((10, y_offset), f"  Y: {ee_pos[1]:+.3f}m", fill=(255, 255, 255), font=font_small)
            y_offset += 20
            draw.text((10, y_offset), f"  Z: {ee_pos[2]:+.3f}m", fill=(255, 255, 255), font=font_small)
            y_offset += 25
            
            # 关节角度
            if hasattr(raw_env, '_get_joint_positions'):
                joint_positions = raw_env._get_joint_positions()
                joint_text = f"Joints: [{', '.join([f'{np.degrees(j):.0f}°' for j in joint_positions[:4]])}]"
                draw.text((10, y_offset), joint_text, fill=(200, 200, 200), font=font_small)
            
            # 右下角：目标位置信息
            y_offset = height - 110
            draw.text((width-340, y_offset), "Target Position:", fill=(255, 165, 0), font=font_small)
            y_offset += 20
            draw.text((width-340, y_offset), f"  Current X: {target_pos[0]:+.3f}m", fill=(255, 200, 100), font=font_small)
            y_offset += 20
            draw.text((width-340, y_offset), f"  Current Y: {target_pos[1]:+.3f}m", fill=(255, 200, 100), font=font_small)
            y_offset += 20
            draw.text((width-340, y_offset), f"  Current Z: {target_pos[2]:+.3f}m", fill=(255, 200, 100), font=font_small)
            y_offset += 25
            
            # 目标漂移
            drift_color = (255, 100, 100) if target_drift > 0.03 else (100, 255, 100)
            drift_text = f"Drift: {target_drift*100:.1f}cm"
            draw.text((width-340, y_offset), drift_text, fill=drift_color, font=font_small)
            
            frames.append(image)
            
        except Exception as e:
            if step == 0:
                print(f"⚠️ 捕获帧失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 执行动作
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
        
        # ⚠️ 注意：这里不能依赖环境的done信号，因为环境可能使用了错误的阈值
        # 我们需要重新计算当前距离来判断是否真正成功
        current_ee_pos = raw_env._get_end_effector_position()
        current_target_pos = raw_env.target_position
        current_distance = np.linalg.norm(current_ee_pos - current_target_pos)
        
        # ✅ 使用正确的4cm阈值判断成功
        if current_distance < CORRECT_SUCCESS_THRESHOLD:
            success_achieved = True
        
        if done:
            break
    
    # 保存GIF
    if frames:
        gif_path = os.path.join(output_dir, f"corrected_episode_{episode_num}.gif")
        
        try:
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,
                loop=0,
                optimize=True
            )
            
            final_distance = distance_to_target
            
            print(f"  ✅ GIF已保存: {gif_path}")
            print(f"     帧数: {len(frames)} | 步数: {episode_length}")
            print(f"     奖励: {episode_reward:.2f} | 最终距离: {final_distance*100:.1f}cm")
            print(f"     成功: {'✅ YES (< 4cm)' if success_achieved else '❌ NO'}")
            
            return gif_path
            
        except Exception as e:
            print(f"❌ 保存GIF失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("❌ 没有捕获到帧")
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成正确阈值的GIF (4cm)')
    parser.add_argument('--model', type=str, required=True,
                       help='模型路径 (例如: ./experiments/.../models/CrossQ_final.zip)')
    parser.add_argument('--vecnorm', type=str, required=True,
                       help='VecNormalize参数路径 (例如: ./experiments/.../models/CrossQ_vecnormalize.pkl)')
    parser.add_argument('--output', type=str, default='./corrected_gifs',
                       help='输出目录')
    parser.add_argument('--algorithm', type=str, default='CrossQ',
                       choices=['SAC', 'TQC', 'CrossQ'],
                       help='算法名称')
    parser.add_argument('--episodes', type=int, default=3,
                       help='生成的回合数')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='每回合最大步数')
    parser.add_argument('--fps', type=int, default=30,
                       help='GIF帧率')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误: 找不到模型文件 {args.model}")
        return
    
    if not os.path.exists(args.vecnorm):
        print(f"⚠️ 警告: 找不到VecNormalize文件 {args.vecnorm}")
    
    # 生成GIF
    generate_gif_corrected(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        output_path=args.output,
        algorithm=args.algorithm,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps
    )


if __name__ == "__main__":
    # 如果没有命令行参数，使用默认路径（你提供的CrossQ模型）
    if len(sys.argv) == 1:
        print("\n使用默认路径生成GIF...\n")
        
        model_path = r"D:\thesis\alpha5_rl_pybullet\experiments\CrossQ_stage4_600000steps_20251115_191718\models\CrossQ_final.zip"
        vecnorm_path = r"D:\thesis\alpha5_rl_pybullet\experiments\CrossQ_stage4_600000steps_20251115_191718\models\CrossQ_vecnormalize.pkl"
        output_path = r"D:\thesis\alpha5_rl_pybullet\experiments\CrossQ_stage4_600000steps_20251115_191718\videos_corrected"
        
        generate_gif_corrected(
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            output_path=output_path,
            algorithm='CrossQ',
            n_episodes=3,
            max_steps=500,
            fps=30
        )
    else:
        main()