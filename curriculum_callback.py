"""
Curriculum Stage Monitoring Callback
监控课程学习阶段转换
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import json
import os

class CurriculumMonitorCallback(BaseCallback):
    """
    监控Curriculum Learning进展的Callback
    记录每次stage转换的时间点、episode数、成功率等
    """
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.curriculum_history = []
        self.last_stage = None
        self.episode_count = 0
        self.success_count = 0
        self.stage_episode_count = {}  # 每个stage的episode计数
        
    def _on_step(self) -> bool:
        # 获取环境
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]
            
            # 解包VecNormalize和Monitor层
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            
            # 检查是否有curriculum_stage属性
            if hasattr(unwrapped_env, 'curriculum_stage'):
                current_stage = unwrapped_env.curriculum_stage
                
                # 检测stage变化
                if current_stage != self.last_stage:
                    # 计算上一个stage的统计信息
                    if self.last_stage is not None:
                        episodes_in_last_stage = self.stage_episode_count.get(self.last_stage, 0)
                    else:
                        episodes_in_last_stage = 0
                    
                    stage_info = {
                        'timestep': self.num_timesteps,
                        'episode': self.episode_count,
                        'old_stage': self.last_stage,
                        'new_stage': current_stage,
                        'episodes_in_previous_stage': episodes_in_last_stage,
                        'stage_config': {}
                    }
                    
                    # 获取stage配置信息
                    if hasattr(unwrapped_env, 'curriculum_config'):
                        stage_config = unwrapped_env.curriculum_config.get(current_stage, {})
                        stage_info['stage_config'] = {
                            'drift_strength': float(stage_config.get('drift_strength', 0)),
                            'success_threshold': float(stage_config.get('success_threshold', 0)),
                            'workspace_radius': float(stage_config.get('workspace_radius', 0))
                        }
                    
                    self.curriculum_history.append(stage_info)
                    
                    # 重置当前stage的episode计数
                    self.stage_episode_count[current_stage] = 0
                    
                    if self.verbose > 0:
                        print(f"\n🎓 Curriculum Stage Transition at timestep {self.num_timesteps}:")
                        print(f"   Stage {self.last_stage} → Stage {current_stage}")
                        print(f"   Episodes in previous stage: {episodes_in_last_stage}")
                        if stage_info['stage_config']:
                            print(f"   New drift strength: {stage_info['stage_config']['drift_strength']:.3f} m/s")
                            print(f"   Success threshold: {stage_info['stage_config']['success_threshold']*100:.1f} cm")
                    
                    self.last_stage = current_stage
                
                # 统计当前stage的episodes
                if current_stage is not None:
                    if current_stage not in self.stage_episode_count:
                        self.stage_episode_count[current_stage] = 0
        
        # 统计episode数和成功数
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_count += 1
                    if self.last_stage is not None:
                        self.stage_episode_count[self.last_stage] = self.stage_episode_count.get(self.last_stage, 0) + 1
                    
                    if info.get('success', False):
                        self.success_count += 1
        
        return True
    
    def _on_training_end(self) -> None:
        """训练结束时保存curriculum历史"""
        
        # 计算最终stage的episodes
        if self.last_stage is not None:
            final_stage_episodes = self.stage_episode_count.get(self.last_stage, 0)
        else:
            final_stage_episodes = 0
        
        summary = {
            'total_episodes': self.episode_count,
            'total_successes': self.success_count,
            'overall_success_rate': self.success_count / max(self.episode_count, 1),
            'total_stage_transitions': len(self.curriculum_history),
            'final_stage': self.last_stage,
            'episodes_per_stage': self.stage_episode_count,
            'transitions': self.curriculum_history
        }
        
        # 保存到JSON文件
        save_path = os.path.join(self.log_dir, "curriculum_history.json")
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print("📊 Curriculum Learning Summary")
            print(f"{'='*60}")
            print(f"✅ Curriculum history saved to: {save_path}")
            print(f"   Total stage transitions: {len(self.curriculum_history)}")
            print(f"   Final stage reached: {self.last_stage}")
            print(f"   Total episodes: {self.episode_count}")
            print(f"   Overall success rate: {summary['overall_success_rate']*100:.1f}%")
            print(f"\n   Episodes per stage:")
            for stage, count in sorted(self.stage_episode_count.items()):
                print(f"      Stage {stage}: {count} episodes")