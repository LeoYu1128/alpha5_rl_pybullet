#!/usr/bin/env python3
"""
Stage4 模型测试与可视化工具
功能：
1. 加载已训练的模型
2. 在Stage4环境中测试
3. 生成详细的GIF演示（显示漂移、噪声、域随机化等信息）
4. 统计性能指标

用法：
python test_models_stage4.py --models path/to/model1.zip path/to/model2.zip --episodes 5
"""

import os
import sys
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import json

import pybullet as p
from PIL import Image, ImageDraw, ImageFont

# 导入必要的模块
from stable_baselines3 import SAC
from sb3_contrib import TQC, CrossQ
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 导入环境
sys.path.append(os.path.dirname(__file__))
from envs.rl_env_v7 import AlphaReachEnv
from train_v8 import STAGE_CONFIGS


class Stage4ModelTester:
    """Stage4 环境下的模型测试器"""
    
    def __init__(self, output_dir='stage4_test_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage4 配置
        self.stage_config = STAGE_CONFIGS['stage4']
        self.env_config = {k: v for k, v in self.stage_config.items() 
                          if k not in ['description', 'recommended_timesteps']}
        
        print(f"\n{'='*80}")
        print(f"🔬 Stage4 模型测试器")
        print(f"{'='*80}")
        print(f"输出目录: {self.output_dir}")
        print(f"Stage4 配置:")
        print(f"  - 目标漂移: {self.env_config.get('enable_target_drift', False)}")
        print(f"  - 域随机化: {self.env_config.get('enable_domain_randomization', False)}")
        print(f"  - 传感器噪声: {self.env_config.get('enable_sensor_noise', False)}")
        print(f"  - 课程学习: {self.env_config.get('enable_curriculum', False)}")
        print(f"{'='*80}\n")
    
    def load_model(self, model_path, algorithm='SAC'):
        """加载训练好的模型"""
        algo_classes = {'SAC': SAC, 'TQC': TQC, 'CrossQ': CrossQ}
        
        if algorithm not in algo_classes:
            raise ValueError(f"不支持的算法: {algorithm}")
        
        print(f"📦 加载 {algorithm} 模型: {model_path}")
        
        try:
            model = algo_classes[algorithm].load(model_path)
            print(f"   ✅ 模型加载成功")
            return model
        except Exception as e:
            print(f"   ❌ 模型加载失败: {e}")
            return None
    
    def test_model(self, model, algorithm='SAC', model_name='model', 
                   n_episodes=5, max_steps=500):
        """测试模型并生成统计数据"""
        
        print(f"\n{'='*60}")
        print(f"🎯 测试模型: {model_name}")
        print(f"{'='*60}")
        
        # 创建测试环境
        test_env = AlphaReachEnv(render_mode=None, **self.env_config)
        vec_env = DummyVecEnv([lambda: test_env])
        
        # 尝试加载VecNormalize参数
        model_dir = Path(model_name).parent if '/' in model_name or '\\' in model_name else Path('.')
        vecnorm_path = model_dir / f"{algorithm}_vecnormalize.pkl"
        
        if vecnorm_path.exists():
            print(f"   📊 加载VecNormalize: {vecnorm_path}")
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        
        # 统计数据
        results = {
            'model_name': model_name,
            'algorithm': algorithm,
            'episodes': [],
            'success_rate': 0.0,
            'mean_reward': 0.0,
            'mean_distance': 0.0,
            'mean_steps': 0.0
        }
        
        for episode in range(n_episodes):
            obs = vec_env.reset()
            episode_reward = 0
            episode_steps = 0
            success = False
            final_distance = 0.0
            
            # 记录环境状态用于GIF
            env_states = []
            
            for step in range(max_steps):
                # 获取环境状态信息
                raw_env = test_env
                env_state = self._capture_env_state(raw_env)
                env_states.append(env_state)
                
                # 执行动作
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                
                episode_reward += reward[0]
                episode_steps += 1
                
                if done[0]:
                    if isinstance(info, list) and len(info) > 0:
                        info = info[0]
                    success = info.get('success', False)
                    final_distance = info.get('distance', 0.0)
                    break
            
            # 记录episode结果
            episode_result = {
                'episode': episode + 1,
                'reward': float(episode_reward),
                'steps': episode_steps,
                'success': success,
                'distance': float(final_distance)
            }
            results['episodes'].append(episode_result)
            
            print(f"   Episode {episode+1}/{n_episodes}: "
                  f"成功={success}, 奖励={episode_reward:.2f}, "
                  f"距离={final_distance*100:.2f}cm, 步数={episode_steps}")
            
            # 生成GIF
            if episode < 3:  # 只为前3个episode生成GIF
                self._generate_enhanced_gif(
                    raw_env, model, vec_env, 
                    algorithm, model_name, episode + 1,
                    max_steps=max_steps
                )
        
        vec_env.close()
        
        # 计算统计指标
        results['success_rate'] = float(np.mean([e['success'] for e in results['episodes']]))
        results['mean_reward'] = float(np.mean([e['reward'] for e in results['episodes']]))
        results['mean_distance'] = float(np.mean([e['distance'] for e in results['episodes']]))
        results['mean_steps'] = float(np.mean([e['steps'] for e in results['episodes']]))
        
        print(f"\n📊 测试结果统计:")
        print(f"   成功率: {results['success_rate']*100:.1f}%")
        print(f"   平均奖励: {results['mean_reward']:.2f}")
        print(f"   平均距离: {results['mean_distance']*100:.2f}cm")
        print(f"   平均步数: {results['mean_steps']:.1f}")
        
        # 保存结果
        result_file = os.path.join(self.output_dir, f"{model_name}_results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   💾 结果已保存: {result_file}")
        
        return results
    
    def _capture_env_state(self, env):
        """捕获环境当前状态信息"""
        state = {}
        
        try:
            # 基础信息
            state['step'] = env.current_step
            state['ee_pos'] = env._get_end_effector_position().copy()
            state['target_pos'] = env.target_position.copy()
            state['initial_target_pos'] = env.initial_target_position.copy()
            
            # 计算距离和漂移
            state['distance'] = float(np.linalg.norm(state['ee_pos'] - state['target_pos']))
            state['drift'] = float(np.linalg.norm(state['target_pos'] - state['initial_target_pos']))
            
            # 域随机化参数
            state['drag_coef'] = env.drag_coefficient
            state['water_density'] = env.water_density
            state['turbulence'] = env.turbulence_strength
            
            # 课程学习信息
            state['curriculum_level'] = env.curriculum_level
            state['success_threshold'] = env.success_threshold
            
            # 水流信息
            state['current_velocity'] = env.current_velocity_actual.copy()
            
            # 传感器噪声状态
            state['sensor_noise_enabled'] = env.enable_sensor_noise
            
        except Exception as e:
            print(f"⚠️ 状态捕获警告: {e}")
        
        return state
    
    def _generate_enhanced_gif(self, env, model, vec_env, algorithm, model_name, 
                               episode_num, max_steps=500, fps=30):
        """生成增强版GIF，显示详细的Stage4信息"""
        
        frames = []
        obs = vec_env.reset()
        
        episode_reward = 0
        episode_length = 0
        success_achieved = False
        
        print(f"   🎬 生成GIF Episode {episode_num}...")
        
        for step in range(max_steps):
            # 捕获图像
            try:
                width, height = 800, 600  # 更大的分辨率
                
                # 获取当前状态
                ee_pos = env._get_end_effector_position()
                target_pos = env.target_position
                initial_target = env.initial_target_position
                
                distance = np.linalg.norm(ee_pos - target_pos)
                drift = np.linalg.norm(target_pos - initial_target)
                
                # 设置相机
                camera_target = [(ee_pos[0] + target_pos[0]) / 2,
                                (ee_pos[1] + target_pos[1]) / 2,
                                (ee_pos[2] + target_pos[2]) / 2]
                
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=camera_target,
                    distance=1.8, yaw=45, pitch=-25, roll=0,
                    upAxisIndex=2,
                    physicsClientId=env.physics_client
                )
                
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60, aspect=width/height, 
                    nearVal=0.1, farVal=100.0,
                    physicsClientId=env.physics_client
                )
                
                img_arr = p.getCameraImage(
                    width=width, height=height,
                    viewMatrix=view_matrix, 
                    projectionMatrix=proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    physicsClientId=env.physics_client
                )
                
                rgb_array = np.array(img_arr[2], dtype=np.uint8)
                if rgb_array.ndim == 1:
                    rgb_array = rgb_array.reshape((height, width, 4))
                rgb_array = rgb_array[:, :, :3]
                
                image = Image.fromarray(rgb_array, 'RGB')
                
                # 添加信息覆盖层
                image = self._add_stage4_info_overlay(
                    image, env, algorithm, model_name, episode_num,
                    step, max_steps, episode_reward, success_achieved,
                    distance, drift
                )
                
                frames.append(image)
                
            except Exception as e:
                if step == 0:
                    print(f"   ⚠️ 捕获帧失败: {e}")
            
            # 执行动作
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            
            if info.get('success', False):
                success_achieved = True
            
            if done[0]:
                break
        
        # 保存GIF
        if frames:
            gif_name = f"{model_name}_episode{episode_num}.gif"
            gif_path = os.path.join(self.output_dir, gif_name)
            
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000//fps,
                loop=0,
                optimize=True
            )
            
            print(f"      ✅ GIF已保存: {gif_path}")
            print(f"         帧数: {len(frames)} | 步数: {episode_length}")
            print(f"         奖励: {episode_reward:.2f} | 成功: {'✅' if success_achieved else '❌'}")
    
    def _add_stage4_info_overlay(self, image, env, algorithm, model_name, 
                                  episode_num, step, max_steps, reward, 
                                  success, distance, drift):
        """在图像上添加Stage4详细信息覆盖层"""
        
        draw = ImageDraw.Draw(image)
        
        try:
            font_title = ImageFont.truetype("arial.ttf", 20)
            font_large = ImageFont.truetype("arial.ttf", 16)
            font_normal = ImageFont.truetype("arial.ttf", 14)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font_title = ImageFont.load_default()
            font_large = font_title
            font_normal = font_title
            font_small = font_title
        
        width, height = image.size
        
        # 创建半透明背景
        from PIL import Image as PILImage
        overlay = PILImage.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # 顶部信息条
        overlay_draw.rectangle([(0, 0), (width, 140)], fill=(0, 0, 0, 200))
        
        # 左侧信息栏
        overlay_draw.rectangle([(0, height-300), (400, height)], fill=(0, 0, 0, 200))
        
        # 右侧信息栏
        overlay_draw.rectangle([(width-400, height-220), (width, height)], fill=(0, 0, 0, 200))
        
        # 合并覆盖层
        image = PILImage.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # ========== 顶部：基本信息 ==========
        y = 10
        title = f"🎯 Stage4 Test: {algorithm} - {model_name} | Episode {episode_num}"
        draw.text((10, y), title, fill=(255, 255, 0), font=font_title)
        y += 30
        
        progress = f"Step {step}/{max_steps} | Reward: {reward:.1f}"
        draw.text((10, y), progress, fill=(255, 255, 255), font=font_large)
        y += 25
        
        # 任务状态
        status_color = (0, 255, 0) if success else (255, 255, 255)
        status_text = "✅ SUCCESS!" if success else f"Target Distance: {distance*100:.1f}cm"
        draw.text((10, y), status_text, fill=status_color, font=font_large)
        y += 25
        
        # 成功阈值
        threshold = env.success_threshold
        threshold_color = (0, 255, 0) if distance < threshold else (255, 165, 0) if distance < threshold * 2 else (255, 255, 255)
        threshold_text = f"Distance: {distance*100:.1f}cm / Threshold: {threshold*100:.0f}cm"
        draw.text((10, y), threshold_text, fill=threshold_color, font=font_normal)
        y += 25
        
        # 课程等级
        curriculum_level = env.curriculum_level
        level_colors = [(100, 255, 100), (255, 255, 100), (255, 200, 100), 
                       (255, 150, 100), (255, 100, 100), (255, 50, 50)]
        level_names = ["Lvl0-Easy", "Lvl1", "Lvl2", "Lvl3", "Lvl4", "Lvl5-Hard"]
        level_color = level_colors[min(curriculum_level, 5)]
        level_text = f"Curriculum: {level_names[min(curriculum_level, 5)]}"
        draw.text((10, y), level_text, fill=level_color, font=font_normal)
        
        # ========== 左下：机械臂状态 ==========
        y = height - 290
        draw.text((10, y), "🤖 Robot State:", fill=(100, 200, 255), font=font_large)
        y += 25
        
        ee_pos = env._get_end_effector_position()
        draw.text((10, y), f"End Effector (TCP):", fill=(150, 150, 255), font=font_normal)
        y += 20
        draw.text((20, y), f"X: {ee_pos[0]:+.3f}m", fill=(200, 200, 200), font=font_small)
        y += 18
        draw.text((20, y), f"Y: {ee_pos[1]:+.3f}m", fill=(200, 200, 200), font=font_small)
        y += 18
        draw.text((20, y), f"Z: {ee_pos[2]:+.3f}m", fill=(200, 200, 200), font=font_small)
        y += 25
        
        # 关节状态
        if hasattr(env, '_get_joint_positions'):
            joint_positions = env._get_joint_positions()
            joint_text = f"Joints: [{', '.join([f'{np.degrees(j):.0f}°' for j in joint_positions[:4]])}]"
            draw.text((10, y), joint_text, fill=(180, 180, 180), font=font_small)
        
        # ========== 右下：Stage4特性 ==========
        y = height - 210
        draw.text((width-390, y), "⚙️ Stage4 Features:", fill=(255, 200, 100), font=font_large)
        y += 25
        
        # 目标漂移
        drift_color = (255, 100, 100) if drift > 0.05 else (100, 255, 100) if drift < 0.02 else (255, 200, 100)
        drift_text = f"🎯 Target Drift: {drift*100:.2f}cm"
        draw.text((width-390, y), drift_text, fill=drift_color, font=font_normal)
        y += 20
        
        # 水流速度
        if hasattr(env, 'current_velocity_actual'):
            current_vel = np.linalg.norm(env.current_velocity_actual)
            vel_text = f"🌊 Water Current: {current_vel:.3f}m/s"
            draw.text((width-390, y), vel_text, fill=(100, 200, 255), font=font_normal)
        y += 20
        
        # 域随机化
        if env.enable_domain_randomization:
            dr_text = f"🎲 Drag Coef: {env.drag_coefficient:.3f}"
            draw.text((width-390, y), dr_text, fill=(255, 180, 100), font=font_small)
            y += 18
            
            density_text = f"   Density: {env.water_density:.0f}kg/m³"
            draw.text((width-390, y), density_text, fill=(180, 180, 180), font=font_small)
            y += 18
            
            turb_text = f"   Turbulence: {env.turbulence_strength:.3f}"
            draw.text((width-390, y), turb_text, fill=(180, 180, 180), font=font_small)
        else:
            draw.text((width-390, y), "🎲 Domain Rand: OFF", fill=(150, 150, 150), font=font_normal)
        y += 20
        
        # 传感器噪声
        if env.enable_sensor_noise:
            noise_text = f"📡 Sensor Noise: ON"
            noise_color = (255, 150, 150)
        else:
            noise_text = f"📡 Sensor Noise: OFF"
            noise_color = (150, 150, 150)
        draw.text((width-390, y), noise_text, fill=noise_color, font=font_normal)
        
        return image


def main():
    parser = argparse.ArgumentParser(description='Stage4模型测试与GIF生成工具')
    parser.add_argument('--models', nargs='+', required=True,
                       help='模型路径列表（.zip文件）')
    parser.add_argument('--algorithms', nargs='+',
                       help='对应的算法名称（SAC/TQC/CrossQ），如未指定则从路径推断')
    parser.add_argument('--episodes', type=int, default=5,
                       help='每个模型测试的episode数量')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='每个episode的最大步数')
    parser.add_argument('--output_dir', type=str, default='stage4_test_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 推断算法名称
    if args.algorithms is None:
        args.algorithms = []
        for model_path in args.models:
            if 'SAC' in model_path.upper():
                args.algorithms.append('SAC')
            elif 'TQC' in model_path.upper():
                args.algorithms.append('TQC')
            elif 'CROSSQ' in model_path.upper():
                args.algorithms.append('CrossQ')
            else:
                args.algorithms.append('SAC')  # 默认
    
    # 确保算法数量匹配
    if len(args.algorithms) != len(args.models):
        args.algorithms = args.algorithms * len(args.models)
        args.algorithms = args.algorithms[:len(args.models)]
    
    # 创建测试器
    tester = Stage4ModelTester(output_dir=args.output_dir)
    
    # 测试所有模型
    all_results = []
    for model_path, algorithm in zip(args.models, args.algorithms):
        model_name = Path(model_path).stem
        
        # 加载模型
        model = tester.load_model(model_path, algorithm)
        if model is None:
            continue
        
        # 测试模型
        results = tester.test_model(
            model, algorithm, model_name,
            n_episodes=args.episodes,
            max_steps=args.max_steps
        )
        all_results.append(results)
    
    # 生成对比报告
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"📊 对比统计")
        print(f"{'='*80}")
        print(f"{'模型':<30} {'成功率':<12} {'平均奖励':<12} {'平均距离':<12}")
        print(f"{'-'*80}")
        
        for result in all_results:
            print(f"{result['model_name']:<30} "
                  f"{result['success_rate']*100:>10.1f}% "
                  f"{result['mean_reward']:>11.2f} "
                  f"{result['mean_distance']*100:>10.2f}cm")
    
    print(f"\n{'='*80}")
    print(f"✅ 测试完成！")
    print(f"📁 结果保存在: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # 示例用法
    if len(sys.argv) == 1:
        print("\n使用示例：")
        print("-" * 80)
        print("# 1. 测试单个模型")
        print("python test_models_stage4.py --models experiments/CrossQ_stage4_600000steps_20251114_232027/models/CrossQ_final.zip --episodes 5")
        print()
        print("# 2. 对比多个模型")
        print("python test_models_stage4.py \\")
        print("    --models \\")
        print("        experiments/SAC_stage4_600000steps_20251114_132607/models/SAC_final.zip \\")
        print("        experiments/TQC_stage4_600000steps_20251114_175245/models/TQC_final.zip \\")
        print("        experiments/CrossQ_stage4_600000steps_20251114_232027/models/CrossQ_final.zip \\")
        print("    --episodes 5")
        print()
        print("# 3. 指定算法（如果路径中无法推断）")
        print("python test_models_stage4.py \\")
        print("    --models model1.zip model2.zip \\")
        print("    --algorithms SAC TQC \\")
        print("    --episodes 5")
        print("-" * 80)
        sys.exit(0)
    
    main()