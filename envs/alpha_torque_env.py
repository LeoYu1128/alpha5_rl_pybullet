"""
纯力矩控制RL环境 - 完整版
基于成功的力矩控制测试结果
"""

import pybullet as p
import pybullet_data
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AlphaRobotTorqueEnv(gym.Env):
    """纯力矩控制Alpha机器人RL环境"""
    
    def __init__(self, render_mode="human", max_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        # # 初始关节位置（安全的中间位置）
        # self.safe_initial_positions = [
        #     3.0,    # joint_1: 基座旋转（中间位置）
        #     3.0,    # joint_2: 肩部（稍微抬起）
        #     1.0,    # joint_3: 肘部（弯曲）
        #     0.0,    # joint_4: 腕部旋转（中间）
        #     0.01,   # joint_5: 夹爪（微开）
        # ]
        # 🎯 基于测试验证的参数
        self.max_torques = np.array([54.36, 54.36, 47.112, 33.069, 28.992])
        self.joint_limits = {
            'lower': np.array([0.032, 0.0174533, 0.0174533, -3.14159, 0.0013]),
            'upper': np.array([6.02, 3.40339, 3.40339, 3.14159, 0.0133])
        }
        
        # 🔧 安全的初始位置（经过验证）
        self.safe_initial_positions = np.array([3.0, 3.0, 2.0, 0.0, 0.01])
        
        # 🎯 渐进式训练：从小力矩开始
        self.training_stage = 1
        self.torque_scale = 0.3  # 开始时使用30%力矩
        
        # 🎯 简单的目标列表
        self.targets = [
            np.array([0.2, 0.0, 0.2]),      # 目标1：正前方
            np.array([0.15, 0.15, 0.2]),    # 目标2：右前方  
            np.array([0.22, -0.08, 0.20]),   # 目标3：左前上方
        ]
        self.current_target_idx = 0
        self.success_count = 0
        self.required_successes = 5  # 连续成功10次换目标
        
        # 连接PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            # 🔧 禁用鼠标拖动，避免干扰RL训练
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.resetDebugVisualizerCamera(0.8, 45, -30, [0, 0, 0.2])
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 初始化场景
        self._setup_scene()
        
        # 🎮 动作空间：纯力矩控制 [-1, 1] → [-max_torque, +max_torque]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # 👀 观察空间：[关节位置(5), 关节速度(5), 末端位置(3), 目标位置(3)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(16,),
            dtype=np.float32
        )
        
        print(f"✅ 纯力矩控制RL环境初始化完成")
        print(f"🎯 初始力矩缩放: {self.torque_scale:.1%}")
        
    def _setup_scene(self):
        """设置场景"""
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载机器人
        robot_path = os.path.join(os.path.dirname(__file__), 
                                 "../alpha_description/urdf/alpha_robot_for_pybullet.urdf")
        if not os.path.exists(robot_path):
            robot_path = "alpha_robot_for_pybullet.urdf"
            
        self.robot_id = p.loadURDF(
            robot_path,
            basePosition=[0, 0, 0.02],
            useFixedBase=True
        )
        
        # 🌊 设置真实的水下阻尼（保持原始参数）
        for i in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, i,
                           jointDamping=0.7,  # 原始阻尼
                           lateralFriction=0.5)
        
        # 获取关节信息
        self.joint_indices = [2, 3, 4, 5, 7]  # 基于诊断结果
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
        self.tcp_index = 11  # TCP索引
        
        # 创建目标
        self._create_target()
        
        print(f"✅ 场景设置完成: {len(self.joint_indices)} 个关节")
        
    def _create_target(self):
        """创建目标"""
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.025, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=self.targets[0]
        )
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # 🔧 重置到安全初始位置
        for i, (joint_idx, initial_pos) in enumerate(zip(self.joint_indices, self.safe_initial_positions)):
            p.resetJointState(self.robot_id, joint_idx, initial_pos, targetVelocity=0)
            
        # 设置当前目标位置
        self.current_target = self.targets[self.current_target_idx].copy()
        p.resetBasePositionAndOrientation(
            self.target_id, self.current_target, [0, 0, 0, 1]
        )
        
        # 稳定仿真
        for _ in range(60):
            p.stepSimulation()
            
        return self._get_observation(), {}
        
    def step(self, action):
        """执行动作 - 纯力矩控制"""
        self.current_step += 1
        
        # 🎮 纯力矩控制：action [-1,1] → torque [-max_torque, +max_torque]
        commanded_torques = action * self.max_torques * self.torque_scale
        
        # 🛡️ 安全检查：防止关节超限时施加错误方向的力矩
        safe_torques = self._apply_safety_limits(commanded_torques)
        
        # 🔧 应用力矩控制
        for i, (joint_idx, torque) in enumerate(zip(self.joint_indices, safe_torques)):
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.TORQUE_CONTROL,
                force=torque
            )
            
        # 仿真步进
        for _ in range(4):
            p.stepSimulation()
            
        # 获取新状态
        observation = self._get_observation()
        
        # 🏆 计算奖励
        reward = self._compute_reward(safe_torques)
        
        # 检查终止条件
        success = self._is_success()
        safety_violation = self._check_safety_violation()
        
        terminated = success or safety_violation
        truncated = self.current_step >= self.max_steps
        
        # 🎯 目标切换和难度递增逻辑
        if success:
            self.success_count += 1
            if self.success_count >= self.required_successes:
                self._advance_training()
        elif terminated:
            self.success_count = 0
            
        info = {
            'success': success,
            'distance': self._get_distance_to_target(),
            'target_index': self.current_target_idx,
            'consecutive_successes': self.success_count,
            'torque_scale': self.torque_scale,
            'training_stage': self.training_stage,
            'safety_violation': safety_violation,
            'commanded_torques': commanded_torques,
            'safe_torques': safe_torques
        }
        
        return observation, reward, terminated, truncated, info
        
    def _apply_safety_limits(self, torques):
        """应用安全限制"""
        current_positions = self._get_joint_positions()
        current_velocities = self._get_joint_velocities()
        
        safe_torques = torques.copy()
        
        for i in range(5):
            pos = current_positions[i]
            vel = current_velocities[i]
            lower = self.joint_limits['lower'][i]
            upper = self.joint_limits['upper'][i]
            
            # 位置安全检查
            safety_margin = 0.1  # 10cm安全边距
            
            if pos > upper - safety_margin:
                # 接近上限，只允许负向力矩
                safe_torques[i] = min(0, torques[i])
            elif pos < lower + safety_margin:
                # 接近下限，只允许正向力矩  
                safe_torques[i] = max(0, torques[i])
                
            # 速度安全检查
            max_safe_velocity = 3.0  # rad/s
            if abs(vel) > max_safe_velocity:
                # 速度过快，施加阻尼力矩
                damping_torque = -5.0 * vel
                safe_torques[i] = damping_torque
                
        return safe_torques
        
    def _get_observation(self):
        """获取观察"""
        # 关节状态
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        
        # 末端执行器位置
        ee_state = p.getLinkState(self.robot_id, self.tcp_index)
        ee_pos = np.array(ee_state[0])
        
        # 组合观察
        obs = np.concatenate([
            joint_positions,      # 5
            joint_velocities,     # 5
            ee_pos,              # 3
            self.current_target   # 3
        ]).astype(np.float32)
        
        return obs
        
    def _get_joint_positions(self):
        """获取关节位置"""
        positions = []
        for joint_idx in self.joint_indices:
            pos, _, _, _ = p.getJointState(self.robot_id, joint_idx)
            positions.append(pos)
        return np.array(positions)
        
    def _get_joint_velocities(self):
        """获取关节速度"""
        velocities = []
        for joint_idx in self.joint_indices:
            _, vel, _, _ = p.getJointState(self.robot_id, joint_idx)
            velocities.append(vel)
        return np.array(velocities)
        
    def _compute_reward(self, torques):
        """计算奖励 - 针对力矩控制优化"""
        ee_state = p.getLinkState(self.robot_id, self.tcp_index)
        ee_pos = np.array(ee_state[0])
        
        # 距离奖励（主要奖励）
        distance = np.linalg.norm(ee_pos - self.current_target)
        distance_reward = -distance * 10  # 放大距离奖励
        
        # 成功奖励
        success_reward = 0
        if distance < 0.03:  # 3cm内算成功
            success_reward = 100
            
        # 能耗惩罚（鼓励高效的力矩使用）
        energy_penalty = -0.01 * np.sum(np.square(torques / self.max_torques))
        
        # 平滑性奖励（鼓励平滑的运动）
        joint_velocities = self._get_joint_velocities()
        smoothness_reward = -0.001 * np.sum(np.square(joint_velocities))
        
        # 安全奖励（保持在关节限制内）
        joint_positions = self._get_joint_positions()
        safety_reward = 0
        for i, pos in enumerate(joint_positions):
            lower = self.joint_limits['lower'][i]
            upper = self.joint_limits['upper'][i]
            # 在安全范围内给予小奖励
            if lower + 0.1 < pos < upper - 0.1:
                safety_reward += 0.1
                
        total_reward = distance_reward + success_reward + energy_penalty + smoothness_reward + safety_reward
        
        return total_reward
        
    def _get_distance_to_target(self):
        """获取到目标的距离"""
        ee_state = p.getLinkState(self.robot_id, self.tcp_index)
        ee_pos = np.array(ee_state[0])
        return np.linalg.norm(ee_pos - self.current_target)
        
    def _is_success(self):
        """检查是否成功"""
        return self._get_distance_to_target() < 0.03
        
    def _check_safety_violation(self):
        """检查安全违规"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        
        # 检查位置限制
        for i, pos in enumerate(joint_positions):
            lower = self.joint_limits['lower'][i]
            upper = self.joint_limits['upper'][i]
            if pos < lower or pos > upper:
                return True
                
        # 检查速度限制
        for vel in joint_velocities:
            if abs(vel) > 5.0:  # 5 rad/s 是危险速度
                return True
                
        return False
        
    def _advance_training(self):
        """推进训练：切换目标或增加难度"""
        if self.current_target_idx < len(self.targets) - 1:
            # 切换到下一个目标
            self.current_target_idx += 1
            self.success_count = 0
            print(f"🎯 切换到目标 {self.current_target_idx + 1}")
        else:
            # 所有目标都掌握了，增加训练难度
            if self.torque_scale < 1.0:
                self.torque_scale = min(1.0, self.torque_scale + 0.2)
                self.training_stage += 1
                self.current_target_idx = 0  # 重新开始目标循环
                self.success_count = 0
                print(f"🚀 训练升级！阶段 {self.training_stage}, 力矩缩放: {self.torque_scale:.1%}")
            else:
                # 达到最高难度，继续训练
                self.current_target_idx = 0
                self.success_count = 0
                print(f"🏆 达到最高训练难度！继续强化训练")
        
    def render(self):
        """渲染"""
        if self.render_mode == "human":
            pass  # GUI自动渲染
            
    def close(self):
        """关闭环境"""
        p.disconnect()


# # 🚀 纯力矩控制训练脚本
# def train_pure_torque_control():
#     """训练纯力矩控制"""
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.vec_env import DummyVecEnv
#     from stable_baselines3.common.monitor import Monitor
#     from stable_baselines3.common.callbacks import EvalCallback
    
#     print("🚀 开始纯力矩控制RL训练...")
#     print("💡 这是真正的端到端学习！")
#     print("🎯 RL将直接学习 observation → torque 的映射")
    
#     # 创建环境
#     def make_env():
#         env = AlphaRobotTorqueEnv(render_mode=None)
#         env = Monitor(env)
#         return env
    
#     env = DummyVecEnv([make_env])
    
#     # 创建评估环境
#     eval_env = DummyVecEnv([make_env])
    
#     # 创建模型 - 适合力矩控制的参数
#     model = PPO(
#         "MlpPolicy",
#         env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         n_epochs=10,
#         gamma=0.99,
#         gae_lambda=0.95,
#         clip_range=0.2,
#         ent_coef=0.01,  # 适当的探索
#         vf_coef=0.5,
#         max_grad_norm=0.5,
#         verbose=1,
#         tensorboard_log="./alpha_torque_tensorboard/",
#         device = "cpu"  # 使用CPU训练
#     )
    
#     # 评估回调
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path='./alpha_torque_best/',
#         log_path='./alpha_torque_logs/',
#         eval_freq=5000,
#         n_eval_episodes=5,
#         deterministic=True,
#         render=False
#     )
    
#     # 训练
#     print("开始训练...")
#     model.learn(
#         total_timesteps=200000,  # 更多步数，因为力矩控制更复杂
#         callback=eval_callback
#     )
    
#     # 保存模型
#     model.save("alpha_pure_torque_model")
#     print("✅ 纯力矩控制模型已保存")
    
#     # 测试训练结果
#     print("🧪 测试训练结果...")
#     test_env = AlphaRobotTorqueEnv(render_mode="human")
    
#     obs, _ = test_env.reset()
#     episode_reward = 0
    
#     for step in range(1000):
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         episode_reward += reward
        
#         if step % 100 == 0:
#             print(f"Step {step}: Distance = {info['distance']:.4f}m, "
#                   f"Torque scale = {info['torque_scale']:.1%}, "
#                   f"Stage = {info['training_stage']}")
        
#         if terminated or truncated:
#             print(f"Episode finished! Success: {info['success']}, "
#                   f"Total reward: {episode_reward:.2f}")
#             obs, _ = test_env.reset()
#             episode_reward = 0
            
#     test_env.close()


def test_environment():
    """测试环境基本功能"""
    print("🤖 纯力矩控制Alpha RL环境测试...")
    env = AlphaRobotTorqueEnv(render_mode="human")
    
    obs, _ = env.reset()
    print("环境已启动，随机动作测试...")
    print("💡 观察：RL需要学会在重力作用下控制关节")
    print(f"观察空间维度: {env.observation_space.shape}")
    print(f"动作空间维度: {env.action_space.shape}")
    
    total_reward = 0
    for i in range(1000):
        # 随机动作测试
        action = env.action_space.sample() * 0.1  # 小幅度动作避免太剧烈
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if i % 100 == 0:
            print(f"Step {i}: Distance = {info['distance']:.4f}m, "
                  f"Success count = {info['consecutive_successes']}, "
                  f"Safety violation = {info['safety_violation']}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            print(f"Success: {info['success']}")
            print(f"Total reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0
            
        # 检查目标切换
        if 'target_index' in info and i > 0:
            print(f"Current target: {info['target_index'] + 1}")
            
    env.close()
    print("✅ 环境测试完成")


if __name__ == "__main__":
    # import sys
    
    # if len(sys.argv) > 1 and sys.argv[1] == "train":
    #     train_pure_torque_control()
    # else:
    # 默认运行测试环境
    test_environment()