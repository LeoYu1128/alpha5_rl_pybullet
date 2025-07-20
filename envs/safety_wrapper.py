"""
安全包装器 - 为Alpha机器人环境提供额外的安全保护
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SafetyWrapper(gym.Wrapper):
    """安全包装器，确保机器人操作安全"""
    
    def __init__(self, env, 
                 enable_emergency_stop=True,
                 enable_soft_limits=True,
                 enable_collision_recovery=True,
                 max_recovery_attempts=3):
        super().__init__(env)
        
        self.enable_emergency_stop = enable_emergency_stop
        self.enable_soft_limits = enable_soft_limits
        self.enable_collision_recovery = enable_collision_recovery
        self.max_recovery_attempts = max_recovery_attempts
        
        # 安全监控
        self.emergency_stop_triggered = False
        self.recovery_mode = False
        self.recovery_attempts = 0
        
        # 历史记录（用于异常检测）
        self.position_history = []
        self.velocity_history = []
        self.torque_history = []
        self.history_length = 10
        
        # 安全阈值
        self.max_position_change = 0.5  # 弧度
        self.max_velocity = 2.0  # 弧度/秒
        self.max_torque = 50.0  # Nm
        self.max_acceleration = 5.0  # 弧度/秒²
        
        # 软限位（比硬限位更保守）
        self.soft_limit_margin = 0.1  # 弧度
        
    def reset(self, **kwargs):
        """重置环境并清除安全状态"""
        self.emergency_stop_triggered = False
        self.recovery_mode = False
        self.recovery_attempts = 0
        self.position_history.clear()
        self.velocity_history.clear()
        self.torque_history.clear()
        
        return self.env.reset(**kwargs)
        
    def step(self, action):
        """执行动作并进行安全检查"""
        # 检查紧急停止
        if self.emergency_stop_triggered:
            print("WARNING: Emergency stop is active!")
            action = np.zeros_like(action)
            
        # 恢复模式
        if self.recovery_mode:
            action = self._get_recovery_action()
            
        # 应用软限位
        if self.enable_soft_limits:
            action = self._apply_soft_limits(action)
            
        # 检查动作安全性
        if self._is_action_safe(action):
            obs, reward, terminated, truncated, info = self.env.step(action)
        else:
            # 不安全的动作，使用安全动作
            safe_action = self._get_safe_action(action)
            obs, reward, terminated, truncated, info = self.env.step(safe_action)
            reward -= 10  # 惩罚不安全行为
            
        # 更新历史记录
        self._update_history(obs)
        
        # 安全检查
        safety_status = self._perform_safety_checks(obs, info)
        
        # 更新信息
        info['safety_status'] = safety_status
        info['emergency_stop'] = self.emergency_stop_triggered
        info['recovery_mode'] = self.recovery_mode
        
        # 如果检测到严重问题，触发紧急停止
        if safety_status['severity'] == 'critical':
            self._trigger_emergency_stop()
            terminated = True
            
        # 恢复模式检查
        if self.recovery_mode and safety_status['severity'] == 'safe':
            self.recovery_mode = False
            self.recovery_attempts = 0
            print("Recovery successful!")
            
        return obs, reward, terminated, truncated, info
        
    def _apply_soft_limits(self, action):
        """应用软限位"""
        # 获取当前关节位置
        joint_positions = self.env._get_joint_positions()
        
        # 预测下一步位置
        predicted_positions = joint_positions + action * 0.1  # 假设动作是速度
        
        # 检查并限制
        limited_action = action.copy()
        for i, joint_name in enumerate(self.env.joint_names):
            limits = self.env.joint_limits[joint_name]
            soft_lower = limits['lower'] + self.soft_limit_margin
            soft_upper = limits['upper'] - self.soft_limit_margin
            
            if predicted_positions[i] < soft_lower:
                limited_action[i] = max(limited_action[i], 0)
            elif predicted_positions[i] > soft_upper:
                limited_action[i] = min(limited_action[i], 0)
                
        return limited_action
        
    def _is_action_safe(self, action):
        """检查动作是否安全"""
        # 检查动作范围
        if np.any(np.abs(action) > 1.0):
            return False
            
        # 检查动作变化率
        if hasattr(self, 'prev_action'):
            action_change = np.abs(action - self.prev_action)
            if np.any(action_change > 0.2):  # 最大变化率
                return False
                
        self.prev_action = action.copy()
        return True
        
    def _get_safe_action(self, action):
        """获取安全的替代动作"""
        # 限制动作幅度
        safe_action = np.clip(action, -0.5, 0.5)
        
        # 平滑动作变化
        if hasattr(self, 'prev_action'):
            max_change = 0.1
            safe_action = np.clip(
                safe_action,
                self.prev_action - max_change,
                self.prev_action + max_change
            )
            
        return safe_action
        
    def _get_recovery_action(self):
        """获取恢复动作"""
        # 缓慢返回到安全位置
        current_positions = self.env._get_joint_positions()
        safe_positions = np.array(self.env.initial_joint_positions)
        
        # PD控制返回安全位置
        error = safe_positions - current_positions
        action = np.clip(0.5 * error, -0.2, 0.2)
        
        return action
        
    def _update_history(self, obs):
        """更新历史记录"""
        # 提取关节信息
        joint_positions = obs[:5]
        joint_velocities = obs[5:10]
        joint_torques = obs[-5:]
        
        # 更新历史
        self.position_history.append(joint_positions)
        self.velocity_history.append(joint_velocities)
        self.torque_history.append(joint_torques)
        
        # 限制历史长度
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            self.torque_history.pop(0)
            
    def _perform_safety_checks(self, obs, info):
        """执行安全检查"""
        safety_status = {
            'severity': 'safe',
            'issues': []
        }
        
        # 1. 检查关节位置异常
        if len(self.position_history) > 1:
            position_change = np.abs(self.position_history[-1] - self.position_history[-2])
            if np.any(position_change > self.max_position_change):
                safety_status['issues'].append('excessive_position_change')
                safety_status['severity'] = 'warning'
                
        # 2. 检查速度异常
        if len(self.velocity_history) > 0:
            velocities = self.velocity_history[-1]
            if np.any(np.abs(velocities) > self.max_velocity):
                safety_status['issues'].append('excessive_velocity')
                safety_status['severity'] = 'warning'
                
        # 3. 检查力矩异常
        if len(self.torque_history) > 0:
            torques = self.torque_history[-1]
            if np.any(np.abs(torques) > self.max_torque):
                safety_status['issues'].append('excessive_torque')
                safety_status['severity'] = 'critical'
                
        # 4. 检查加速度异常
        if len(self.velocity_history) > 1:
            dt = 1/240.0  # 仿真时间步
            acceleration = (self.velocity_history[-1] - self.velocity_history[-2]) / dt
            if np.any(np.abs(acceleration) > self.max_acceleration):
                safety_status['issues'].append('excessive_acceleration')
                safety_status['severity'] = 'warning'
                
        # 5. 检查碰撞
        if 'safety_penalty' in info and info['safety_penalty'] > 5:
            safety_status['issues'].append('collision_detected')
            safety_status['severity'] = 'critical'
            
        # 6. 检查奇异性
        if 'jacobian_condition' in info and info['jacobian_condition'] > 50:
            safety_status['issues'].append('near_singularity')
            safety_status['severity'] = 'warning'
            
        # 决定是否进入恢复模式
        if safety_status['severity'] == 'warning' and len(safety_status['issues']) > 2:
            if self.enable_collision_recovery and self.recovery_attempts < self.max_recovery_attempts:
                self.recovery_mode = True
                self.recovery_attempts += 1
                print(f"Entering recovery mode (attempt {self.recovery_attempts})")
                
        return safety_status
        
    def _trigger_emergency_stop(self):
        """触发紧急停止"""
        if self.enable_emergency_stop:
            self.emergency_stop_triggered = True
            print("EMERGENCY STOP TRIGGERED!")
            print("System requires manual reset")
            
    def reset_emergency_stop(self):
        """手动重置紧急停止"""
        self.emergency_stop_triggered = False
        self.recovery_mode = False
        self.recovery_attempts = 0
        print("Emergency stop reset. System ready.")
        
    def get_safety_report(self):
        """获取安全报告"""
        report = {
            'emergency_stop': self.emergency_stop_triggered,
            'recovery_mode': self.recovery_mode,
            'recovery_attempts': self.recovery_attempts,
            'history_stats': {}
        }
        
        # 计算历史统计
        if self.position_history:
            positions = np.array(self.position_history)
            report['history_stats']['position_std'] = np.std(positions, axis=0)
            
        if self.velocity_history:
            velocities = np.array(self.velocity_history)
            report['history_stats']['max_velocity'] = np.max(np.abs(velocities))
            report['history_stats']['avg_velocity'] = np.mean(np.abs(velocities))
            
        if self.torque_history:
            torques = np.array(self.torque_history)
            report['history_stats']['max_torque'] = np.max(np.abs(torques))
            report['history_stats']['avg_torque'] = np.mean(np.abs(torques))
            
        return report