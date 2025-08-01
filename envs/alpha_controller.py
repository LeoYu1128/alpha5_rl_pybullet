import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R


class AlphaRobotController:

    
    def __init__(self, robot_id, joint_indices, joint_names):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.joint_names = joint_names
        
        # 初始化控制器状态
        self._init_control_parameters()
        self._init_kinematics()
        self._init_actuator_model()
        self._init_underwater_dynamics()
        
        # 控制历史
        self.integral_errors = np.zeros(len(joint_indices))
        self.prev_errors = np.zeros(len(joint_indices))
        self.prev_velocities = np.zeros(len(joint_indices))
        self.prev_time = 0.0
        
        # 末端执行器控制
        self.eef_target_pos = np.array([0.2, 0.0, 0.2])
        self.eef_target_vel = np.zeros(3)
        self.eef_control_enabled = False
        
    def _init_control_parameters(self):
        """初始化控制参数 - 基于实际yaml配置"""
        
        # 速度跟踪PID参数 (来自 ctrl_param_sim_velocity_tracking.yaml)
        self.velocity_pid_params = {
            'kp': np.array([0.5, 0.3, 0.15, 0.001, 10.0]),      # P增益
            'ki': np.array([0.25, 0.25, 0.15, 0.015, 1.0]),     # I增益  
            'kd': np.array([0.005, 0.005, 0.001, 0.00001, 0.1]), # D增益
            'kff': np.array([1.0, 1.0, 1.0, 1.0, 1.0])         # 前馈增益
        }
        
        # 关节运动学控制增益
        self.joint_kinematic_gains = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        # 末端执行器运动学控制增益
        self.eef_kinematic_gains = np.array([1.0, 1.0, 1.0])  # x, y, z
        
        # 关节限位和努力限制
        self.joint_limits = {
            'joint_1': {'lower': 0.032, 'upper': 6.02, 'effort': 54.36, 'velocity': 2.0},
            'joint_2': {'lower': 0.0174533, 'upper': 3.40339, 'effort': 54.36, 'velocity': 2.0},
            'joint_3': {'lower': 0.0174533, 'upper': 3.40339, 'effort': 47.112, 'velocity': 2.0},
            'joint_4': {'lower': -3.14159, 'upper': 3.14159, 'effort': 33.069, 'velocity': 2.0},
            'joint_5': {'lower': 0.0013, 'upper': 0.0133, 'effort': 28.992, 'velocity': 1.0}
        }
        
        # 积分限幅
        self.integral_limits = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
        
        # 控制模式
        self.use_feedforward = True
        self.use_underwater_compensation = True
        self.use_actuator_model = True
        
    def _init_kinematics(self):
        """初始化运动学参数 - 基于DH参数"""
        
        # DH参数 (来自kinematics.hpp中的常量定义)
        self.theta_c = np.arctan(40 / 145.3)  # THETA_C
        self.d0 = 0.0462
        self.a0 = 0.02
        self.a1 = np.sqrt(40*40 + 145.3*145.3) / 1000  # A1
        self.a2 = 0.02
        self.d3 = -0.18
        
        # DH表格 [d, theta0, a, alpha]
        self.dh_table = np.array([
            [self.d0, np.pi, self.a0, np.pi/2],
            [0.0, -1.3021582029078473, self.a1, np.pi],
            [0.0, -1.3021582029078473, self.a2, -np.pi/2],
            [self.d3, np.pi/2, 0.0, np.pi/2],
            [0.0, -np.pi/2, 0.0, 0.0]
        ])
        
        # 雅可比矩阵缓存
        self.jacobian = np.zeros((3, 4))  # 位置雅可比
        self.jacobian_full = np.zeros((6, 4))  # 完整雅可比
        
    def _init_actuator_model(self):
        """初始化执行器模型 - 基于actuator_model.hpp"""
        
        # 扭矩常数 [Nm/A]
        self.K_t = np.array([0.0134, 0.0134, 0.0134, 0.0271, 0.01])
        
        # 齿轮比
        self.gear_ratios = np.array([1754.4, 1754.4, 1754.4, 340.4, 1.0])
        
        # 电流偏移 [mA]
        self.current_offsets = np.array([200, 250, 290, 45, 0])
        
        # 单位转换常数
        self.mA_to_A = 0.001
        
    def _init_underwater_dynamics(self):
        """初始化水下动力学参数"""
        
        # 水下环境参数
        self.water_density = 1025.0  # kg/m³
        self.gravity = 9.81
        
        # 各链接质量 (来自inertial参数)
        self.link_masses = np.array([0.367, 0.161, 0.38, 0.142, 0.355])
        
        # 浮力质量 (来自hyd_mass参数)
        # self.buoyancy_masses = np.array([0.205588, 0.031936, 0.217564, 0.031936, 0.13373200000000002])
        self.buoyancy_masses = np.array([0.205588, 0.031936, 0.217564, 0.031936, 0.133732])
        # 基于流体力学计算的阻尼参数
        self.linear_damping = np.array([2.8, 2.8, 2.8, 2.8, 40.0])      # 4x空气阻尼
        self.quadratic_damping = np.array([0.026, 0.026, 0.026, 0.026, 0.1])  # 基于40mm直径压差阻力
        
    def compute_forward_kinematics(self, joint_positions):
        """计算正运动学 - 基于DH参数"""
        
        # 更新DH表格中的关节角度
        dh_current = self.dh_table.copy()
        for i in range(min(4, len(joint_positions))):
            dh_current[i, 1] += joint_positions[i]  # theta = theta0 + q
            
        # 计算变换矩阵
        T = np.eye(4)
        for i in range(4):
            d, theta, a, alpha = dh_current[i]
            
            # DH变换矩阵
            T_i = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            
            T = T @ T_i
            
        # 末端执行器位置
        eef_position = T[:3, 3]
        eef_orientation = T[:3, :3]
        
        return eef_position, eef_orientation
    
    def compute_jacobian(self, joint_positions):
        """计算雅可比矩阵"""
        
        # 简化的数值雅可比计算
        epsilon = 1e-6
        eef_pos_0, _ = self.compute_forward_kinematics(joint_positions)
        
        jacobian = np.zeros((3, len(joint_positions)))
        
        for i in range(min(4, len(joint_positions))):  # 只对前4个关节计算
            q_plus = joint_positions.copy()
            q_plus[i] += epsilon
            
            eef_pos_plus, _ = self.compute_forward_kinematics(q_plus)
            jacobian[:, i] = (eef_pos_plus - eef_pos_0) / epsilon
            
        return jacobian
    
    def compute_underwater_forces(self, joint_positions, joint_velocities):
        """计算水下力 - 基于水下动力学"""
        
        if not self.use_underwater_compensation:
            return np.zeros(len(joint_positions))
            
        underwater_torques = np.zeros(len(joint_positions))
        
        for i in range(len(joint_positions)):
            # 浮力补偿 (简化)
            buoyancy_force = self.buoyancy_masses[i] * self.gravity * 0.1  # 简化系数
            
            # 阻尼力
            linear_damping_force = -self.linear_damping[i] * joint_velocities[i]
            quadratic_damping_force = -self.quadratic_damping[i] * joint_velocities[i] * abs(joint_velocities[i])
            
            # 总的水下力矩
            underwater_torques[i] = buoyancy_force + linear_damping_force + quadratic_damping_force
            
        return underwater_torques
    
    def compute_eef_kinematic_control(self, current_joint_positions):
        """末端执行器运动学控制"""
        
        if not self.eef_control_enabled:
            return np.zeros(len(current_joint_positions))
            
        # 获取当前末端执行器位置
        eef_pos, _ = self.compute_forward_kinematics(current_joint_positions)
        
        # 位置误差
        pos_error = self.eef_target_pos - eef_pos
        
        # 期望笛卡尔空间速度
        desired_cartesian_vel = self.eef_target_vel + np.diag(self.eef_kinematic_gains) @ pos_error
        
        # 计算雅可比矩阵
        jacobian = self.compute_jacobian(current_joint_positions)
        
        # 雅可比伪逆求解关节速度
        try:
            JJT = jacobian[:, :4] @ jacobian[:, :4].T  # 只用前4个关节
            JJT_inv = np.linalg.inv(JJT + 1e-6 * np.eye(3))  # 添加阻尼避免奇异
            desired_joint_vel = jacobian[:, :4].T @ JJT_inv @ desired_cartesian_vel
            
            # 扩展到5个关节 (夹爪保持)
            full_desired_vel = np.zeros(len(current_joint_positions))
            full_desired_vel[:4] = desired_joint_vel
            
        except np.linalg.LinAlgError:
            # 奇异情况下使用最小二乘解
            full_desired_vel = np.zeros(len(current_joint_positions))
            
        return full_desired_vel
    
    def apply_joint_control(self, target_positions=None, target_velocities=None, dt=1.0/240.0):
        
        
        # 获取当前关节状态
        current_positions = np.zeros(len(self.joint_indices))
        current_velocities = np.zeros(len(self.joint_indices))
        
        for i, joint_idx in enumerate(self.joint_indices):
            pos, vel, _, _ = p.getJointState(self.robot_id, joint_idx)
            current_positions[i] = pos
            current_velocities[i] = vel
            
        # 1. 末端执行器控制 (如果启用)
        if self.eef_control_enabled and target_velocities is None:
            target_velocities = self.compute_eef_kinematic_control(current_positions)
            
        # 2. 位置控制转速度控制 (如果提供目标位置)
        if target_positions is not None and target_velocities is None:
            # 关节空间运动学控制
            pos_errors = target_positions - current_positions
            target_velocities = np.diag(self.joint_kinematic_gains) @ pos_errors
            
        # 3. 速度控制 (默认为零速度)
        if target_velocities is None:
            target_velocities = np.zeros(len(self.joint_indices))
            
        # 限制目标速度
        for i, joint_name in enumerate(self.joint_names):
            if i < len(target_velocities) and joint_name in self.joint_limits:
                max_vel = self.joint_limits[joint_name]['velocity']
                target_velocities[i] = np.clip(target_velocities[i], -max_vel, max_vel)
        
        # 4. 速度PID控制 + 前馈控制
        control_torques = np.zeros(len(self.joint_indices))
        
        for i in range(len(self.joint_indices)):
            # 速度误差
            vel_error = target_velocities[i] - current_velocities[i]
            
            # 积分项 (带限幅)
            self.integral_errors[i] += vel_error * dt
            self.integral_errors[i] = np.clip(self.integral_errors[i], 
                                            -self.integral_limits[i], 
                                            self.integral_limits[i])
            
            # 微分项
            d_error = (vel_error - self.prev_errors[i]) / dt
            
            # PID控制器输出
            pid_output = (self.velocity_pid_params['kp'][i] * vel_error + 
                         self.velocity_pid_params['ki'][i] * self.integral_errors[i] + 
                         self.velocity_pid_params['kd'][i] * d_error)
            
            # 前馈控制 (如果启用)
            feedforward_output = 0.0
            if self.use_feedforward:
                desired_acceleration = self.velocity_pid_params['kff'][i] * vel_error
                feedforward_output = desired_acceleration * 0.1  # 简化的前馈项
                
            control_torques[i] = pid_output + feedforward_output
            
            # 更新历史误差
            self.prev_errors[i] = vel_error
            
        # 5. 水下动力学补偿
        underwater_compensation = self.compute_underwater_forces(current_positions, current_velocities)
        control_torques += underwater_compensation
            
        # 6. 扭矩限制和应用
        for i, (joint_idx, joint_name) in enumerate(zip(self.joint_indices, self.joint_names)):
            if joint_name in self.joint_limits:
                max_effort = self.joint_limits[joint_name]['effort']
                torque = np.clip(control_torques[i], -max_effort, max_effort)
            else:
                torque = control_torques[i]
                
            # 应用扭矩控制
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.TORQUE_CONTROL,
                force=torque
            )
            
        return control_torques
    
    def set_eef_target(self, position, velocity=None, enable=True):
        """设置末端执行器目标"""
        self.eef_target_pos = np.array(position)
        self.eef_target_vel = np.array(velocity) if velocity is not None else np.zeros(3)
        self.eef_control_enabled = enable
        
    def reset_integrators(self):
        """重置积分器"""
        self.integral_errors.fill(0.0)
        self.prev_errors.fill(0.0)
        
    def enable_features(self, feedforward=None, underwater=None, actuator_model=None):
        """启用/禁用功能"""
        if feedforward is not None:
            self.use_feedforward = feedforward
        if underwater is not None:
            self.use_underwater_compensation = underwater
        if actuator_model is not None:
            self.use_actuator_model = actuator_model
