import pybullet as p
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AlphaRobotEnvRealistic(gym.Env): 
    """Alpha Robot 环境 - 带真实物理参数"""
    
    def __init__(self, render_mode="human", max_steps=1000,
                 enable_water_dynamics=True,
                 enable_actuator_model=True):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # 物理仿真开关
        self.enable_water_dynamics = enable_water_dynamics
        self.enable_actuator_model = enable_actuator_model
        
        # 关节限制（真实参数）
        self.joint_limits = {
            'joint_1': {'lower': 0.032, 'upper': 6.02, 'effort': 54.36, 'velocity': 2.0},
            'joint_2': {'lower': 0.0174533, 'upper': 3.40339, 'effort': 54.36, 'velocity': 2.0},
            'joint_3': {'lower': 0.0174533, 'upper': 3.40339, 'effort': 47.112, 'velocity': 2.0},
            'joint_4': {'lower': -3.14159, 'upper': 3.14159, 'effort': 33.069, 'velocity': 2.0},
            'joint_5': {'lower': 0.0013, 'upper': 0.0133, 'effort': 28.992, 'velocity': 1.0}
        }
        
        # 水下动力学参数（来自alpha_controller.py）
        self.water_density = 1025.0  # kg/m³
        self.gravity = 9.81
        self.link_masses = np.array([0.367, 0.161, 0.38, 0.142, 0.355])
        self.buoyancy_masses = np.array([0.205588, 0.031936, 0.217564, 0.031936, 0.133732])
        self.linear_damping = np.array([2.8, 2.8, 2.8, 2.8, 40.0])
        self.quadratic_damping = np.array([0.026, 0.026, 0.026, 0.026, 0.1])
        
        # 执行器模型参数（真实电机参数）
        self.motor_kt = np.array([0.0134, 0.0134, 0.0134, 0.0271, 0.01])  # Nm/A
        self.gear_ratios = np.array([1754.4, 1754.4, 1754.4, 340.4, 1.0])
        self.motor_resistance = np.array([1.0, 1.0, 1.0, 0.5, 0.1])  # 简化的电阻值
        self.max_current = np.array([10.0, 10.0, 10.0, 15.0, 5.0])  # 最大电流 A
        
        # 连接PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.2]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # 设置重力
        p.setGravity(0, 0, -self.gravity)
        p.setTimeStep(1./240.)
        
        # 初始化场景
        self._setup_scene()
        
        # 动作空间：直接扭矩控制 [-1, 1] 映射到最大扭矩
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # 观察空间：[关节位置(5), 关节速度(5), 关节扭矩(5), 目标关节位置(5)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20,),
            dtype=np.float32
        )
        
        # 目标和历史
        self.target_joint_positions = None
        self.applied_torques = np.zeros(5)
        
    def _setup_scene(self):
        """设置场景"""
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载机器人
        robot_path = "path/to/alpha_robot.urdf"  # 您的URDF路径
        self.robot_id = p.loadURDF(
            robot_path,
            basePosition=[0, 0, 0.02],
            useFixedBase=True
        )
        
        # 设置真实的动力学参数
        self._setup_realistic_dynamics()
        
        # 获取关节信息
        self._setup_joints()
        
    def _setup_realistic_dynamics(self):
        """设置真实的动力学参数"""
        for i in range(p.getNumJoints(self.robot_id)):
            # 设置真实的质量（如果URDF中不准确）
            if i < len(self.link_masses):
                p.changeDynamics(
                    self.robot_id, i,
                    mass=self.link_masses[i]
                )
            
            # 设置水下环境的阻尼
            if self.enable_water_dynamics:
                # 基础阻尼（PyBullet的jointDamping）
                p.changeDynamics(
                    self.robot_id, i,
                    jointDamping=0.7,  # 我们自己计算阻尼
                    lateralFriction=0.3,
                    spinningFriction=0.1,
                    rollingFriction=0.1
                )
                
    def _compute_water_forces(self, joint_velocities, joint_positions):
        """计算水下环境的力（基于真实参数）"""
        if not self.enable_water_dynamics:
            return np.zeros(5)
            
        water_torques = np.zeros(5)
        
        for i in range(5):
            # 1. 浮力扭矩（简化计算）
            # 实际应该基于链接的姿态和浮心位置
            buoyancy_torque = self.buoyancy_masses[i] * self.gravity * 0.1 * np.sin(joint_positions[i])
            
            # 2. 阻尼扭矩
            # 线性阻尼：τ = -b * ω
            linear_damping_torque = -self.linear_damping[i] * joint_velocities[i]
            
            # 二次阻尼：τ = -c * ω * |ω|
            quadratic_damping_torque = -self.quadratic_damping[i] * joint_velocities[i] * abs(joint_velocities[i])
            
            # 总水下扭矩
            water_torques[i] = buoyancy_torque + linear_damping_torque + quadratic_damping_torque
            
        return water_torques
        
    def _compute_actuator_torque(self, commanded_torques, joint_velocities):
        """计算真实的执行器输出（考虑电机特性）"""
        if not self.enable_actuator_model:
            return commanded_torques
            
        actual_torques = np.zeros(5)
        
        for i in range(5):
            # 反电动势效应
            back_emf = self.motor_kt[i] * self.gear_ratios[i] * joint_velocities[i]
            
            # 可用电压（简化）
            available_voltage = 24.0 - back_emf  # 假设24V供电
            
            # 电流限制
            desired_current = commanded_torques[i] / (self.motor_kt[i] * self.gear_ratios[i])
            max_current_from_voltage = available_voltage / self.motor_resistance[i]
            
            actual_current = np.clip(
                desired_current,
                -min(self.max_current[i], max_current_from_voltage),
                min(self.max_current[i], max_current_from_voltage)
            )
            
            # 实际输出扭矩
            actual_torques[i] = actual_current * self.motor_kt[i] * self.gear_ratios[i]
            
        return actual_torques
        
    def step(self, action):
        """执行动作（直接扭矩控制）"""
        self.current_step += 1
        
        # 将动作映射到扭矩
        commanded_torques = self._action_to_torques(action)
        
        # 获取当前状态
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        
        # 计算水下力
        water_torques = self._compute_water_forces(joint_velocities, joint_positions)
        
        # 计算实际电机输出
        motor_torques = self._compute_actuator_torque(commanded_torques, joint_velocities)
        
        # 总扭矩 = 电机扭矩 + 水下力
        total_torques = motor_torques + water_torques
        
        # 应用扭矩
        for i, (joint_idx, torque) in enumerate(zip(self.joint_indices, total_torques)):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torque
            )
            
        # 记录实际应用的扭矩
        self.applied_torques = total_torques
        
        # 仿真步进
        for _ in range(4):
            p.stepSimulation()
            
        # 获取新状态
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查终止
        terminated = self._is_success()
        truncated = self.current_step >= self.max_steps
        
        info = {
            'commanded_torques': commanded_torques,
            'motor_torques': motor_torques,
            'water_torques': water_torques,
            'total_torques': total_torques,
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'success': terminated
        }
        
        return observation, reward, terminated, truncated, info
        
    def _action_to_torques(self, action):
        """将归一化动作映射到扭矩命令"""
        torques = []
        for i, joint_name in enumerate(self.joint_names):
            max_torque = self.joint_limits[joint_name]['effort']
            # 使用80%的最大扭矩作为限制
            torque = action[i] * max_torque * 0.8
            torques.append(torque)
        return np.array(torques)
        
    def _get_observation(self):
        """获取观察（包含扭矩信息）"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            self.applied_torques,  # 实际应用的扭矩
            self.target_joint_positions
        ])
        
        return observation.astype(np.float32)
        
    def _calculate_reward(self):
        """计算奖励（针对扭矩控制优化）"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        
        # 位置误差
        position_errors = joint_positions - self.target_joint_positions
        position_reward = -np.sum(np.square(position_errors))
        
        # 速度惩罚（鼓励平稳运动）
        velocity_penalty = -0.1 * np.sum(np.square(joint_velocities))
        
        # 扭矩效率惩罚（鼓励低能耗）
        torque_penalty = -0.01 * np.sum(np.square(self.applied_torques))
        
        # 成功奖励
        if np.all(np.abs(position_errors) < 0.05):
            success_bonus = 100.0
        else:
            success_bonus = 0.0
            
        return position_reward + velocity_penalty + torque_penalty + success_bonus
