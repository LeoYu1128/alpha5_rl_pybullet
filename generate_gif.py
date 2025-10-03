#!/usr/bin/env python3
"""
简单的GIF生成脚本
用法: python generate_gif.py --model 模型路径 --episodes 回合数 --output 输出目录
"""

import os
import argparse
from pathlib import Path

# 设置环境变量避免OpenMP冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from train_v5_success import record_episode_gif_vecenv, create_combined_gif
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.rl_env_v5_success import AlphaReachEnv
import json
import numpy as np

def generate_gif_simple(model_path, episodes, output_dir, algorithm='SAC', fps=30):
    """
    生成GIF的简化函数（不包含OpenGL可视化）

    Args:
        model_path: 模型文件路径 (.zip文件)
        episodes: 要录制的回合数
        output_dir: 输出目录
        algorithm: 算法类型 (SAC, PPO, TD3)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建临时实验目录结构
    exp_dir = output_dir
    os.makedirs(os.path.join(exp_dir, "videos"), exist_ok=True)

    # 从model_path推断模型文件路径（去掉.zip后缀）
    if model_path.endswith('.zip'):
        model_base_path = model_path[:-4]
    else:
        model_base_path = model_path
        model_path = model_path + '.zip'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"开始生成GIF...")
    print(f"模型路径: {model_path}")
    print(f"回合数: {episodes}")
    print(f"输出目录: {output_dir}")
    print("="*50)

    try:
        # 加载模型
        algo_classes = {'SAC': SAC, 'PPO': PPO, 'TD3': TD3}
        model = algo_classes[algorithm].load(model_path)

        # 创建带标准化的环境（与训练时一致）
        gif_env_raw = AlphaReachEnv(render_mode=None)
        gif_env = DummyVecEnv([lambda: gif_env_raw])

        # 加载VecNormalize参数
        model_dir = os.path.dirname(model_base_path)
        model_name = os.path.basename(model_base_path).split('_')[0]  # 提取算法名
        vecnorm_path = os.path.join(model_dir, f"{model_name}_vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            print(f"✓ 加载环境标准化参数: {vecnorm_path}")
            gif_env = VecNormalize.load(vecnorm_path, gif_env)
            gif_env.training = False
            gif_env.norm_reward = False
        else:
            print("⚠️  未找到VecNormalize参数文件，使用原始环境")

        # 测试数据记录
        test_results = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'final_distances': [],
            'episode_lengths': [],
            'gif_paths': []
        }

        print(f"开始录制 {episodes} 个回合的GIF...")
        print("注意：正在录制GIF，可能需要较长时间...")

        # 录制每个回合
        for episode in range(episodes):
            print(f"\n📹 回合 {episode + 1}/{episodes}")

            # 录制GIF
            gif_path, episode_reward, episode_length, info = record_episode_gif_vecenv(
                gif_env, model, episode + 1, exp_dir, max_steps=400, fps=fps
            )

            success = info.get('success', False)
            final_distance = info.get('distance', 0)

            print(f"  回合结束:")
            print(f"    {'✅' if success else '❌'} 成功: {success}")
            print(f"    📏 最终距离: {final_distance:.3f}m")
            print(f"    🎯 总奖励: {episode_reward:.2f}")
            print(f"    📊 步数: {episode_length}")
            if gif_path:
                size = os.path.getsize(gif_path) / 1024  # KB
                print(f"    🎬 GIF: {os.path.basename(gif_path)} ({size:.1f}KB)")

            # 记录结果
            test_results['episodes'].append(episode + 1)
            test_results['rewards'].append(float(episode_reward))
            test_results['success_rates'].append(1 if success else 0)
            test_results['final_distances'].append(float(final_distance))
            test_results['episode_lengths'].append(episode_length)
            test_results['gif_paths'].append(gif_path if gif_path else "")

        gif_env.close()

        # 创建综合GIF（所有回合的最佳片段）
        print(f"\n🎬 创建综合GIF...")
        create_combined_gif(exp_dir, test_results, algorithm)

        # 保存测试结果
        test_summary = {
            'total_episodes': episodes,
            'average_reward': float(np.mean(test_results['rewards'])),
            'success_rate': float(np.mean(test_results['success_rates'])),
            'average_distance': float(np.mean(test_results['final_distances'])),
            'average_length': float(np.mean(test_results['episode_lengths'])),
            'gif_count': sum(1 for path in test_results.get('gif_paths', []) if path),
            'detailed_results': test_results
        }

        with open(os.path.join(exp_dir, "test_results.json"), 'w') as f:
            json.dump(test_summary, f, indent=2)

        print(f"\n{'='*50}")
        print("🎉 GIF生成成功！")
        print(f"{'='*50}")
        print(f"📈 总回合数: {episodes}")
        print(f"🎯 平均奖励: {test_summary['average_reward']:.2f}")
        print(f"✅ 成功率: {test_summary['success_rate']*100:.1f}%")
        print(f"📏 平均最终距离: {test_summary['average_distance']:.3f}m")
        print(f"📊 平均回合长度: {test_summary['average_length']:.1f}步")

        # 列出生成的GIF文件
        videos_dir = os.path.join(output_dir, "videos")
        if os.path.exists(videos_dir):
            gif_files = [f for f in os.listdir(videos_dir) if f.endswith('.gif')]
            print(f"\n🎬 生成的GIF文件 ({len(gif_files)}个):")
            for gif_file in sorted(gif_files):
                gif_path = os.path.join(videos_dir, gif_file)
                size = os.path.getsize(gif_path) / 1024  # KB
                print(f"  - {gif_file} ({size:.1f}KB)")

        print(f"\n📁 输出目录: {output_dir}/videos/")

        return True

    except Exception as e:
        print(f"❌ GIF生成过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='生成机械臂训练模型的GIF动画')
    parser.add_argument('--model', '-m', required=True,
                       help='模型文件路径 (例如: models/SAC_final.zip 或 models/SAC_final)')
    parser.add_argument('--episodes', '-e', type=int, default=3,
                       help='要录制的回合数 (默认: 3)')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录 (例如: ./gif_output)')
    parser.add_argument('--algorithm', '-a', choices=['SAC', 'PPO', 'TD3'], default='SAC',
                       help='算法类型 (默认: SAC)')
    parser.add_argument('--fps', '-f', type=int, default=10,
                       help='GIF播放帧率 (默认: 30, 数值越小播放越慢)')

    args = parser.parse_args()

    # 转换为绝对路径
    model_path = os.path.abspath(args.model)
    output_dir = os.path.abspath(args.output)

    print(f"Alpha5机械臂GIF生成器")
    print(f"{'='*50}")

    # 生成GIF
    success = generate_gif_simple(
        model_path=model_path,
        episodes=args.episodes,
        output_dir=output_dir,
        algorithm=args.algorithm,
        fps=args.fps
    )

    if success:
        print(f"\n✅ 成功完成GIF生成！")
        print(f"📁 查看结果: {output_dir}/videos/")
    else:
        print(f"\n❌ GIF生成失败，请检查模型路径和参数")
        exit(1)

if __name__ == "__main__":
    main()