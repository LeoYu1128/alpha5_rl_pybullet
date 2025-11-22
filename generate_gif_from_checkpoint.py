import os
import sys
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from stable_baselines3 import SAC
from sb3_contrib import TQC, CrossQ
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


from envs.rl_env_v9 import AlphaReachEnv


def generate_gif_from_checkpoint(
    model_path,
    algorithm='SAC',
    vecnorm_path=None,
    output_dir=None,
    n_episodes=3,
    max_steps=400,
    fps=30
):
    """
    从训练好的checkpoint生成GIF演示
    """
    
    print("="*60)
    print(f"🎬 从Checkpoint生成GIF")
    print("="*60)
    print(f"模型路径: {model_path}")
    print(f"算法: {algorithm}")
    print(f"回合数: {n_episodes}")
    print("="*60 + "\n")
    
    # 1. 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return
    
    # 2. 确定输出目录
    if output_dir is None:
        exp_dir = Path(model_path).parent.parent
        output_dir = exp_dir / "videos"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}\n")
    
    # 3. 加载模型
    print(f"⏳ 加载 {algorithm} 模型...")
    algo_classes = {
        'SAC': SAC,
        'TQC': TQC,
        'CrossQ': CrossQ
    }
    
    if algorithm not in algo_classes:
        print(f"❌ 不支持的算法: {algorithm}")
        return
    
    try:
        model = algo_classes[algorithm].load(model_path)
        print(f"✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 4. 创建环境
    print("⏳ 创建仿真环境...")
    test_env = AlphaReachEnv(render_mode=None)
    vec_env = DummyVecEnv([lambda: test_env])
    
    # 5. 加载VecNormalize
    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"⏳ 加载环境标准化参数: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("✅ VecNormalize加载成功")
    else:
        print("⚠️  未找到VecNormalize参数，使用原始环境")
    
    print(f"✅ 环境准备完成\n")
    
    # 6. 生成GIF
    gif_paths = []
    
    for episode_num in range(1, n_episodes + 1):
        print(f"🎥 录制回合 {episode_num}/{n_episodes}...")
        
        frames = []
        obs = vec_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # ===== ✅ 关键修复：正确获取原始环境 =====
        try:
            # 尝试不同的方式获取原始环境
            if hasattr(vec_env, 'venv'):
                # VecNormalize包装的情况
                raw_env = vec_env.venv.envs[0]
            elif hasattr(vec_env, 'envs'):
                # DummyVecEnv的情况
                raw_env = vec_env.envs[0]
            else:
                # 直接是环境
                raw_env = vec_env
            
            # 如果还是Monitor包装，继续解包
            if hasattr(raw_env, 'env'):
                raw_env = raw_env.env
                
        except Exception as e:
            print(f"  ⚠️  获取原始环境失败: {e}")
            raw_env = test_env
        
        # ===== ✅ 关键修复：初始化info字典 =====
        info = {'success': False, 'distance': 0.0}
        
        for step in range(max_steps):
            # 捕获帧
            try:
                width, height = 640, 480
                ee_pos = raw_env._get_end_effector_position()
                target_pos = raw_env.target_position
                
                # 相机设置
                camera_target = [
                    (ee_pos[0] + target_pos[0]) / 2,
                    (ee_pos[1] + target_pos[1]) / 2,
                    (ee_pos[2] + target_pos[2]) / 2
                ]
                
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=camera_target,
                    distance=1.5,
                    yaw=45,
                    pitch=-30,
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
                
                img_arr = p.getCameraImage(
                    width=width,
                    height=height,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    physicsClientId=raw_env.physics_client
                )
                
                # 转换为PIL图像
                rgb_array = np.array(img_arr[2], dtype=np.uint8)
                if rgb_array.ndim == 1:
                    rgb_array = rgb_array.reshape((height, width, 4))
                rgb_array = rgb_array[:, :, :3]
                
                image = Image.fromarray(rgb_array, 'RGB')
                
                # 添加文字信息
                draw = ImageDraw.Draw(image)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                distance = np.linalg.norm(ee_pos - target_pos)
                info_text = f"Episode: {episode_num} | Step: {step} | Distance: {distance:.3f}m"
                draw.text((10, 10), info_text, fill=(255, 255, 255), font=font)
                
                reward_text = f"Reward: {episode_reward:.1f}"
                draw.text((10, 35), reward_text, fill=(255, 255, 0), font=font)
                
                frames.append(image)
                
            except Exception as e:
                if step == 0:
                    print(f"  ⚠️  捕获帧失败: {e}")
            
            # 执行动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info_list = vec_env.step(action)
            
            # ===== ✅ 关键修复：正确处理info =====
            # VecEnv返回的是列表
            if isinstance(reward, (list, np.ndarray)):
                episode_reward += float(reward[0])
            else:
                episode_reward += float(reward)
            
            episode_length += 1
            
            # 处理done标志
            if isinstance(done, (list, np.ndarray)):
                done = bool(done[0])
            else:
                done = bool(done)
            
            # 处理info字典
            if isinstance(info_list, list) and len(info_list) > 0:
                info = info_list[0]
            elif isinstance(info_list, dict):
                info = info_list
            else:
                info = {'success': False, 'distance': 0.0}
            
            if done:
                break
        
        # 保存GIF
        if frames:
            gif_path = os.path.join(output_dir, f"episode_{episode_num}.gif")
            
            try:
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=1000//fps,
                    loop=0,
                    optimize=True
                )
                
                # ===== ✅ 关键修复：安全获取info值 =====
                success = info.get('success', False) if isinstance(info, dict) else False
                final_distance = info.get('distance', 0.0) if isinstance(info, dict) else 0.0
                
                print(f"  ✅ GIF已保存: {gif_path}")
                print(f"     帧数: {len(frames)} | 步数: {episode_length}")
                print(f"     奖励: {episode_reward:.2f} | 距离: {final_distance:.3f}m")
                print(f"     成功: {'✅' if success else '❌'}")
                
                gif_paths.append(gif_path)
                
            except Exception as e:
                print(f"  ❌ 保存GIF失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ❌ 没有捕获到帧")
        
        print()
    
    # 清理环境
    vec_env.close()
    
    # 总结
    print("="*60)
    print(f"🎉 完成！成功生成 {len(gif_paths)} 个GIF")
    print("="*60)
    for path in gif_paths:
        print(f"  📹 {path}")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从训练好的checkpoint生成GIF演示')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='实验文件夹路径')
    parser.add_argument('--algorithm', type=str, default='SAC', 
                       choices=['SAC', 'TQC', 'CrossQ'],
                       help='算法名称')
    parser.add_argument('--model_name', type=str, default='final',
                       choices=['final', 'best'],
                       help='使用final模型还是best模型')
    parser.add_argument('--episodes', type=int, default=3,
                       help='生成的回合数')
    parser.add_argument('--max_steps', type=int, default=400,
                       help='每个回合最大步数')
    parser.add_argument('--fps', type=int, default=30,
                       help='GIF帧率')
    
    args = parser.parse_args()
    
    # 构建路径
    exp_dir = Path(args.exp_dir)
    model_path = exp_dir / "models" / f"{args.algorithm}_{args.model_name}.zip"
    vecnorm_path = exp_dir / "models" / f"{args.algorithm}_vecnormalize.pkl"
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print(f"\n可用的模型文件:")
        models_dir = exp_dir / "models"
        if models_dir.exists():
            for f in models_dir.glob("*.zip"):
                print(f"  - {f.name}")
        return
    
    # 生成GIF
    generate_gif_from_checkpoint(
        model_path=str(model_path),
        algorithm=args.algorithm,
        vecnorm_path=str(vecnorm_path) if vecnorm_path.exists() else None,
        output_dir=None,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps
    )


if __name__ == "__main__":
    main()