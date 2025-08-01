import pybullet as p
import pybullet_data
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import yaml
from envs.alpha_controller import AlphaRobotController

class AlphaRobotEnv(gym.Env):
    """Alpha Robot 强化学习环境"""
    
    def __init__(self, render_mode="human", max_steps=1000, dense_reward=True,
                 enable_safety=True, curriculum_learning=False):
        super(AlphaRobotEnv, self).__init__()

        # 🆕 简单的目标管理
        self.target_list = [
            np.array([0.2, 0.1, 0.2]),      # 目标1：正前方
            np.array([0.15, 0.15, 0.25]),   # 目标2：右前方  
            np.array([0.25, -0.15, 0.2]),   # 目标3：左前方
            np.array([0.2, 0.1, 0.2]),      # 目标4：远高位
            np.array([0.18, 0.0, 0.15]),    # 目标5：近低位
        ] 
        self.current_target_index = 0
        self.success_count = 0
        self.episodes_on_target = 0
        self.required_successes = 5  # 连续成功5次才换目标
        self.prev_distance = None
        self.realistic_controller = None  # 初始化控制器
        self.use_realistic_controller = True  # 是否使用真实控制器
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.dense_reward = dense_reward
        self.enable_safety = enable_safety
        self.curriculum_learning = curriculum_learning
        
        # Alpha机械臂参数（基于YAML配置）
        self.MAX_REACH = 0.4  # 最大可达范围 400mm
        self.MIN_REACH = 0.08  # 最小可达范围（避免奇异性）
        self.GRIPPER_RANGE = 0.012  # 夹爪范围 (0.0133 - 0.0013)
        
        # 关节限制（来自alpha_joint_lim_urdf.yaml）
        self.joint_limits = {
            'joint_1': {'lower': 0.032, 'upper': 6.02, 'effort': 54.36, 'velocity': 2.0},
            'joint_2': {'lower': 0.0174533, 'upper': 3.40339, 'effort': 54.36, 'velocity': 2.0},
            'joint_3': {'lower': 0.0174533, 'upper': 3.40339, 'effort': 47.112, 'velocity': 2.0},
            'joint_4': {'lower': -3.14159, 'upper': 3.14159, 'effort': 33.069, 'velocity': 2.0},
            'joint_5': {'lower': 0.0013, 'upper': 0.0133, 'effort': 28.992, 'velocity': 1.0}
        }
        
        # 安全参数
        self.joint_safety_margin = 0.05  # 关节限位安全边距（弧度）
        self.collision_threshold = 0.02  # 碰撞检测阈值（米）
        self.singularity_threshold = 0.05  # 奇异性检测阈值
        self.max_joint_acceleration = 5.0  # 最大关节加速度
        
        # 初始关节位置（安全的中间位置）
        self.initial_joint_positions = [
            3.0,    # joint_1: 基座旋转（中间位置）
            3.0,    # joint_2: 肩部（稍微抬起）
            1.0,    # joint_3: 肘部（弯曲）
            0.0,    # joint_4: 腕部旋转（中间）
            0.01,   # joint_5: 夹爪（微开）
        ]
        
        # 连接PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.8,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.2]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 初始化场景
        self._setup_scene()
        
        # 定义动作空间（5个关节的归一化动作）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(5,), 
            dtype=np.float32
        )
        
        # 定义观察空间
        # [关节位置(5), 关节速度(5), 末端位置(3), 末端姿态(4), 
        #  目标位置(3), 相对位置(3), 雅可比条件数(1), 关节扭矩(5)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            #shape=(29,), 
            shape=(16,), 
            dtype=np.float32
        )
        
        # 性能追踪
        self.episode_rewards = []
        self.success_count = 0
        self.collision_count = 0
        self.singularity_count = 0
        
        # 课程学习参数
        if self.curriculum_learning:
            self.difficulty_level = 0.0  # 0到1，逐渐增加难度
            self.success_threshold = 0.8  # 成功率阈值，用于提升难度
            self.recent_success_rate = 0.0
            
    def _setup_scene(self):
        """设置场景"""
        # 创建地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 设置地面摩擦力
        p.changeDynamics(self.plane_id, -1, 
                        lateralFriction=1.0,
                        spinningFriction=0.1,
                        rollingFriction=0.1)
        
        # 加载机器人
        self.robot_id = self._load_robot()
        #self.table = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        # 获取关节信息
        self._setup_joints()
        
        # 创建目标
        self._create_target()
        
        # 创建工作空间可视化
        if self.render_mode == "human":
            self._create_workspace_visualization()
            
        # 初始化碰撞检测
        if self.enable_safety:
            self._setup_collision_detection()

        if self.use_realistic_controller:
            # 初始化真实控制器
            self.realistic_controller = AlphaRobotController(self.robot_id, self.joint_indices,self.joint_names)
            print("已使用严谨的控制器")
            
    def _load_robot(self):
        
        """加载机器人URDF"""
        robot_path = os.path.join(os.path.dirname(__file__), 
                                 "../alpha_description/urdf/alpha_robot_for_pybullet.urdf")
            
        robot_id = p.loadURDF(
            robot_path,
            basePosition=[0, 0, 0.02],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        )
        
        # 设置关节阻尼
        for i in range(p.getNumJoints(robot_id)):
            # p.changeDynamics(robot_id, i, 
            #                jointDamping=0.1, # 陆地上的阻尼
            #                lateralFriction=0.8)
            
            # 模拟水下行为
            p.changeDynamics(robot_id, i,
                 jointDamping=0.7,  # 关节阻尼
                 lateralFriction=0.5,  # 侧向摩擦
                 linearDamping=0.1,  # 空间阻尼
                 angularDamping=0.1)

                           
        return robot_id
        
    def _setup_joints(self):
        """设置关节信息"""
        self.joint_indices = []
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
        self.joint_info = {}
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            
            if joint_name in self.joint_names:
                self.joint_indices.append(i)
                self.joint_info[joint_name] = {
                    'index': i,
                    'type': joint_info[2],
                    'lower': joint_info[8],
                    'upper': joint_info[9],
                    'max_force': joint_info[10],
                    'max_velocity': joint_info[11]
                }
                
        # 查找末端执行器
        self.end_effector_index = self.joint_indices[-1] if self.joint_indices else 0
        
        # 查找TCP（工具中心点）
        for i in range(p.getNumJoints(self.robot_id)):
            link_info = p.getJointInfo(self.robot_id, i)
            if b'tcp' in link_info[12].lower() or b'ee' in link_info[12].lower():
                self.tcp_index = i
                break
        else:
            self.tcp_index = self.end_effector_index
            
    def _create_target(self):
        """创建目标"""
        # 目标球体
        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.01, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        
        self.target_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.02
        )
        
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.target_visual,
            baseCollisionShapeIndex=self.target_collision,
            basePosition=[0.2, 0.1, 0.2]
        )
        
    def _create_workspace_visualization(self):
        """创建工作空间可视化"""
        # 可达范围球体
        workspace_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.MAX_REACH,
            rgbaColor=[0.2, 0.2, 0.8, 0.1]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=workspace_visual,
            basePosition=[0, 0, 0.1]
        )
        
        # 安全边界
        safety_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.MIN_REACH,
            rgbaColor=[0.8, 0.2, 0.2, 0.1]
        )
        
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=safety_visual,
            basePosition=[0, 0, 0.1]
        )
        
    def _setup_collision_detection(self):
        """设置碰撞检测"""
        # 获取所有链接对，用于自碰撞检测
        self.link_pairs = []
        num_links = p.getNumJoints(self.robot_id)
        
        for i in range(num_links):
            for j in range(i + 2, num_links):  # 跳过相邻链接
                self.link_pairs.append((i, j))
                
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)   
        self.current_step = 0
        self.episode_rewards = []
        self.episodes_on_target += 1      
        # 重置机器人到安全初始位置
        for i, (joint_idx, pos) in enumerate(zip(self.joint_indices, self.initial_joint_positions)):
            p.resetJointState(self.robot_id, joint_idx, pos, targetVelocity=0)
            
        # 🆕 使用当前固定目标
        self.target_position = self.target_list[self.current_target_index].copy()
        # # 设置目标位置
        # if self.curriculum_learning:
        #     self.target_position = self._sample_curriculum_target()
        # else:
        #     self.target_position = self._sample_valid_target()
        
        p.resetBasePositionAndOrientation(
            self.target_id,
            self.target_position,
            [0, 0, 0, 1]
        )
         # 🆕 打印进度（可选）
        if self.episodes_on_target % 50 == 0:
            print(f"目标 {self.current_target_index + 1}/{len(self.target_list)}: {self.target_position}, "
                f"已训练 {self.episodes_on_target} episodes, 连续成功 {self.success_count} 次")
            # 重置历史数据
            self.prev_distance = self._get_distance_to_target()
            self.prev_joint_positions = np.array(self.initial_joint_positions)
        # self.prev_jacobian_cond = self._compute_jacobian_condition()
        
        # 运行几步仿真以稳定
        for _ in range(10):
            p.stepSimulation()
            
        return self._get_observation(), {}
        
    def _sample_valid_target(self):
        """采样有效的目标位置"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # 在工作空间内随机采样
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(np.pi/6, np.pi/2)  # 限制俯仰角
            r = np.random.uniform(self.MIN_REACH + 0.05, self.MAX_REACH - 0.05)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) + 0.05  # 基座高度偏移
            
            # 确保目标在合理高度
            if 0.05 < z < 0.35:
                target = np.array([x, y, z])
                
                # 检查是否可达（简单逆运动学检查）
                if self._is_position_reachable(target):
                    return target
                    
        # 如果找不到，返回默认安全位置
        return np.array([0.2, 0.0, 0.2])
        
    def _sample_curriculum_target(self):
        """课程学习的目标采样"""
        # 根据难度调整目标范围
        min_r = self.MIN_REACH + 0.05
        max_r = min_r + (self.MAX_REACH - min_r) * (0.5 + 0.5 * self.difficulty_level)
        
        # 角度范围也随难度增加
        max_theta = np.pi * (0.5 + 0.5 * self.difficulty_level)
        
        theta = np.random.uniform(-max_theta, max_theta)
        phi = np.random.uniform(np.pi/4, np.pi/2)
        r = np.random.uniform(min_r, max_r)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi) + 0.05
        
        return np.array([x, y, z])
        
    def step(self, action):
        """执行动作"""
        self.current_step += 1
        
        # # 应用安全约束
        # if self.enable_safety:
        #     if self.current_step < self.max_steps // 2:
        #         action = self._apply_safety_constraints_easy(action)
        #     else:
        #         action = self._apply_safety_constraints_hard(action)
        
        # 简单直接的约束
        action = np.clip(action, -1.0, 1.0)  # 确保在动作空间内
        # 将归一化动作转换为关节目标
        # target_positions = self._action_to_joint_positions(action)

        # 将归一化的动作映射到扭矩范围
        torques = self._action_to_torques(action)  # 不是 positions！
        # 应用关节控制
        # self._apply_joint_control(target_positions)

        # 应用关节扭矩控制
        self._apply_torque_control(torques)
        # 步进仿真
        for _ in range(4):
            p.stepSimulation()
            if self.render_mode == "human":
                pass  # GUI自动渲染
                
        # 检测安全违规
        safety_penalty = 0.0
        if self.enable_safety:
            safety_penalty = self._check_safety()
            
        # 获取新状态
        observation = self._get_observation()
        
        # 计算奖励
        reward = self._calculate_reward() - safety_penalty
        
        # 检查终止条件
        success = self._is_success()  # 🆕 只调用一次
        terminated = success
        truncated = self.current_step >= self.max_steps
        
        # 记录性能
        self.episode_rewards.append(reward)
        
        # 信息字典
        info = {
            'distance': self._get_distance_to_target(),
            'success': success,  # 🆕 使用已计算的success
            'episode_reward': sum(self.episode_rewards),
            'safety_penalty': safety_penalty,
        }
        
        # 🆕 目标切换逻辑
        if terminated or truncated:
            if success:
                self.success_count += 1
                print(f"✅ 成功！连续成功 {self.success_count}/{self.required_successes} 次")
                
                # 检查是否需要切换目标
                if self.success_count >= self.required_successes:
                    if self.current_target_index < len(self.target_list) - 1:
                        # 切换到下一个目标
                        self.current_target_index += 1
                        self.success_count = 0
                        self.episodes_on_target = 0
                        
                        new_target = self.target_list[self.current_target_index]
                        print(f"🎯 目标切换！新目标 {self.current_target_index + 1}: {new_target}")
                    else:
                        print(f"🎉 所有目标都掌握了！可以开始随机目标训练")
            else:
                # 失败了，重置连续成功计数
                if self.success_count > 0:
                    print(f"❌ 失败，连续成功计数重置")
                self.success_count = 0
            
            # 更新info
            info['consecutive_successes'] = self.success_count
            info['current_target_index'] = self.current_target_index
            info['episodes_on_target'] = self.episodes_on_target
        
        # 更新历史数据
        self.prev_distance = info['distance']
        self.prev_joint_positions = self._get_joint_positions()
        
        return observation, reward, terminated, truncated, info
        
    def _apply_safety_constraints_hard(self, action):
        """应用安全约束"""
        # 限制动作变化率
        if hasattr(self, 'prev_action'):
            
            max_change = 0.1  # 最大动作变化
            action = np.clip(action, 
                            self.prev_action - max_change,
                            self.prev_action + max_change)
        self.prev_action = action.copy()
        
        return action
        
    def _apply_safety_constraints_easy(self, action):
        """应用安全约束"""
        # 限制动作变化率
        if hasattr(self, 'prev_action'):
            
            max_change = 0.3  # 最大动作变化
            action = np.clip(action, 
                            self.prev_action - max_change,
                            self.prev_action + max_change)
        self.prev_action = action.copy()
        
        return action
    def _apply_torque_control(self, torques):
        """直接使用力矩控制关节"""
        for i, (joint_idx, torque) in enumerate(zip(self.joint_indices, torques)):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                controlMode=p.TORQUE_CONTROL,
                force=torque
            )
    def _action_to_torques(self, action):
        """将[-1,1]的动作映射到扭矩范围"""
        torques = []
        for i, joint_name in enumerate(self.joint_names):
            max_torque = self.joint_limits[joint_name]['effort']
            # 直接映射到扭矩
            torque = action[i] * max_torque
            torques.append(torque)
        return torques

    def _action_to_joint_positions(self, action):
        """将动作转换为关节位置"""
        target_positions = []
        
        for i, joint_name in enumerate(self.joint_names):
            limits = self.joint_limits[joint_name]
            
            # 添加安全边距
            low = limits['lower'] + self.joint_safety_margin
            high = limits['upper'] - self.joint_safety_margin
            
            # 线性映射
            target_pos = low + (action[i] + 1) * (high - low) / 2
            target_positions.append(target_pos)
            
        return target_positions
    def _apply_joint_control(self, target_positions):
        """应用关节控制"""
        # if self.use_realistic_controller and self.realistic_controller is not None:
        #     # 使用真实控制器
        #     return self.realistic_controller.apply_joint_control(target_positions = target_positions)
        # else:
        #     # 使用原始关节控制
        #     return self._apply_original_joint_control(target_positions)
        return self._apply_original_joint_control(target_positions)
    def _apply_original_joint_control(self, target_positions):
        # """应用关节控制"""
        # for i, (joint_idx, target_pos) in enumerate(zip(self.joint_indices, target_positions)):
        #     joint_name = self.joint_names[i]
        #     limits = self.joint_limits[joint_name]
            
        #     # 获取当前关节状态
        #     joint_state = p.getJointState(self.robot_id, joint_idx)
        #     current_pos = joint_state[0]
        #     current_vel = joint_state[1]
            
        #     # 限制速度
        #     max_vel = min(limits['velocity'], 
        #                  abs(target_pos - current_pos) * 10)
            
        #     # PD控制
        #     p.setJointMotorControl2(
        #         self.robot_id,
        #         joint_idx,
        #         p.POSITION_CONTROL,
        #         targetPosition=target_pos,
        #         targetVelocity=0,
        #         force=limits['effort'] * 0.8,  # 使用80%的最大力矩
        #         maxVelocity=max_vel,
        #         positionGain=0.2,
        #         velocityGain=0.1
        #     )
        """应用关节控制：所有5个关节都使用PID控制（基于alpha_controller.py的参数）"""
        
        # 初始化：只在第一次调用时创建积分误差和前次误差缓存
        if not hasattr(self, 'integral_errors'):
            # 为所有5个关节创建PID历史
            self.integral_errors = [0.0] * len(self.joint_indices)
            self.prev_errors = [0.0] * len(self.joint_indices)

        # 控制周期 dt（与仿真步长或 controller 更新率一致）
        dt = 1.0 / 240.0  # 240 Hz

        # 所有5个关节的 PID 参数（来自alpha_controller.py的velocity_pid_params）
        p_gains = [0.5, 0.3, 0.15, 0.001, 10.0]        # P 增益
        i_gains = [0.25, 0.25, 0.15, 0.015, 1.0]       # I 增益
        d_gains = [0.005, 0.005, 0.001, 0.00001, 0.1]  # D 增益
        integral_max = [1.0, 1.0, 1.0, 1.0, 0.5]       # 积分限幅（末端执行器用较小值）

        for i, (joint_idx, target_pos) in enumerate(zip(self.joint_indices, target_positions)):
            # 读取当前状态
            pos, vel, *_ = p.getJointState(self.robot_id, joint_idx)
            
            # 安全检查：确保索引不越界
            if i >= len(p_gains):
                print(f"警告：关节 {i} 超出PID参数范围，使用默认参数")
                kp, ki, kd, imax = 0.1, 0.01, 0.001, 1.0
            else:
                kp = p_gains[i]
                ki = i_gains[i]
                kd = d_gains[i]
                imax = integral_max[i]
            
            # 计算位置误差
            error = target_pos - pos
            
            # 积分累加并限幅
            self.integral_errors[i] = max(-imax, min(self.integral_errors[i] + error * dt, imax))
            
            # 微分项
            d_error = (error - self.prev_errors[i]) / dt
            
            # PID 输出
            torque = kp * error + ki * self.integral_errors[i] + kd * d_error
            
            # 更新历史误差
            self.prev_errors[i] = error
            
            # 限制最大扭矩
            if i < len(self.joint_names) and self.joint_names[i] in self.joint_limits:
                max_effort = self.joint_limits[self.joint_names[i]]['effort']
            else:
                max_effort = 50.0  # 默认值
                
            torque = max(-max_effort, min(torque, max_effort))

            # 发送扭矩控制命令
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.TORQUE_CONTROL,
                force=torque
            )

            # 调试信息（可选，仅末端执行器）
            if i == 4:  # 末端执行器
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 0
                    
                # 每100步打印一次调试信息
                if self._debug_counter % 100 == 0:
                    print(f"末端执行器 - 目标: {target_pos:.4f}, 当前: {pos:.4f}, "
                        f"误差: {error:.4f}, 扭矩: {torque:.2f}")

    def set_eef_target(self, target_position):
        """设置末端执行器目标位置"""
        if self.realistic_controller is not None:
            self.realistic_controller.set_eef_target(target_position, enable=True)
            print(f"末端执行器目标设置为: {target_position}")

    def toggle_control_mode(self, use_eef_control=False):
        """切换控制模式"""
        if self.realistic_controller is not None:
            self.realistic_controller.eef_control_enabled = use_eef_control
            if use_eef_control:
                print("已切换到末端执行器控制模式")
            else:
                print("已切换到关节控制模式")

    def enable_controller_features(self, **kwargs):
        """启用控制器功能"""
        if self.realistic_controller is not None:
            self.realistic_controller.enable_features(**kwargs)

    def _check_safety(self):
        """检查安全约束"""
        penalty = 0.0
        
        # 1. 检查自碰撞
        if self._check_self_collision():
            penalty += 10.0
            self.collision_count += 1
            
        # # 2. 检查奇异性
        # jacobian_cond = self._compute_jacobian_condition()
        # if jacobian_cond > 1 / self.singularity_threshold:
        #     penalty += 5.0 * (jacobian_cond * self.singularity_threshold - 1)
        #     self.singularity_count += 1
            
        # 3. 检查关节限位
        joint_positions = self._get_joint_positions()
        for i, (pos, joint_name) in enumerate(zip(joint_positions, self.joint_names)):
            limits = self.joint_limits[joint_name]
            if pos < limits['lower'] or pos > limits['upper']:
                penalty += 5.0
                
        # 4. 检查速度限制
        joint_velocities = self._get_joint_velocities()
        for i, (vel, joint_name) in enumerate(zip(joint_velocities, self.joint_names)):
            if abs(vel) > self.joint_limits[joint_name]['velocity']:
                penalty += 2.0
                
        return penalty
        
    def _check_self_collision(self):
        """检查自碰撞"""
        for link1, link2 in self.link_pairs:
            contacts = p.getClosestPoints(
                self.robot_id, self.robot_id,
                self.collision_threshold,
                link1, link2
            )
            if contacts:
                return True
        return False
        
    def _compute_jacobian_condition(self):
        """计算雅可比矩阵条件数（奇异性度量）"""
        # 获取雅可比矩阵
        ee_state = p.getLinkState(self.robot_id, self.tcp_index, 
                                 computeForwardKinematics=True)
        ee_pos = ee_state[0]
        
        joint_positions = self._get_joint_positions()
        
        # 使用PyBullet的雅可比计算
        jacobian = p.calculateJacobian(
            self.robot_id,
            self.tcp_index,
            [0, 0, 0],  # 局部坐标
            joint_positions,
            [0] * len(self.joint_indices),  # 零速度
            [0] * len(self.joint_indices)   # 零加速度
        )
        
        # 提取线性部分
        J_linear = np.array(jacobian[0])[:, :len(self.joint_indices)]
        
        # 计算条件数
        try:
            cond = np.linalg.cond(J_linear)
        except:
            cond = 1e6  # 奇异情况
            
        return cond
        
    def _get_observation(self):
        """获取观察状态"""
        # 关节状态
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        joint_torques = self._get_joint_torques()
        
        # 末端执行器状态
        ee_state = p.getLinkState(self.robot_id, self.tcp_index,
                                 computeForwardKinematics=True)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])
        
        # 相对信息
        relative_pos = self.target_position - ee_pos
        distance = self._get_distance_to_target()
        # 雅可比条件数（归一化）
        # jacobian_cond = min(self._compute_jacobian_condition() / 100, 1.0)
        
        # 组合观察
        observation = np.concatenate([
            joint_positions,      # 5
            joint_velocities,     # 5
            ee_pos,              # 3
            #ee_orn,              # 4
            self.target_position, # 3
            #relative_pos,        # 3
            #[distance],         # 1
            #[jacobian_cond],     # 1
            #joint_torques        # 5
        ]).astype(np.float32)
        
        return observation
        
    def _get_joint_positions(self):
        """获取关节位置"""
        positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
        return np.array(positions)
        
    def _get_joint_velocities(self):
        """获取关节速度"""
        velocities = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            velocities.append(joint_state[1])
        return np.array(velocities)
        
    def _get_joint_torques(self):
        """获取关节力矩"""
        torques = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            torques.append(joint_state[3])  # Applied joint motor torque
        return np.array(torques)
        
    def _calculate_reward(self):
        """计算奖励"""
        ee_pos = self._get_end_effector_pos()
        distance = np.linalg.norm(ee_pos - self.target_position)
        
        if self.dense_reward:
            # 1. 距离奖励（指数衰减）
            distance_reward = np.exp(-10 * distance)
            
            if self.prev_distance is not None:
                # 2. 进步奖励
                progress = self.prev_distance - distance
                progress_reward = 50 * progress
            else:
                progress_reward = 0
            # 3. 成功奖励
            if distance < 0.02:
                success_reward = 200
            elif distance < 0.05:
                success_reward = 50
            else:
                success_reward = 0
            
            # 总奖励
            reward = (distance_reward + progress_reward + success_reward)
            
        else:
            # 稀疏奖励
            if distance < 0.02:
                reward = 100
            else:
                reward = -1
                
        return reward

        # """极简奖励函数"""
        # ee_pos = self._get_end_effector_pos()
        # distance = np.linalg.norm(ee_pos - self.target_position)
        
        # # 方法1：简单线性奖励
        # reward = 1.0 - distance  # 距离越近奖励越高
        
        # # 方法2：成功/失败奖励
        # if distance < 0.02:
        #     reward = 10.0  # 成功
        # else:
        #     reward = -0.1  # 小惩罚鼓励尽快完成
        
        # return reward
        
    def _get_end_effector_pos(self):
        """获取末端执行器位置"""
        ee_state = p.getLinkState(self.robot_id, self.tcp_index)
        return np.array(ee_state[0])
        
    def _get_distance_to_target(self):
        """计算到目标的距离"""
        ee_pos = self._get_end_effector_pos()
        return np.linalg.norm(ee_pos - self.target_position)
        
    def _is_success(self):
        """检查是否成功"""
        return self._get_distance_to_target() < 0.02
        
    def _is_position_reachable(self, position):
        """简单的可达性检查"""
        # 检查是否在工作空间内
        distance_from_base = np.linalg.norm(position[:2])
        if distance_from_base < self.MIN_REACH or distance_from_base > self.MAX_REACH:
            return False
            
        # 高度检查
        if position[2] < 0.05 or position[2] > 0.4:
            return False
            
        return True
        
    def update_curriculum(self, success_rate):
        """更新课程学习难度"""
        if self.curriculum_learning:
            self.recent_success_rate = success_rate
            
            # 根据成功率调整难度
            if success_rate > self.success_threshold and self.difficulty_level < 1.0:
                self.difficulty_level = min(1.0, self.difficulty_level + 0.1)
                print(f"Curriculum: Difficulty increased to {self.difficulty_level:.2f}")
            elif success_rate < 0.5 and self.difficulty_level > 0.0:
                self.difficulty_level = max(0.0, self.difficulty_level - 0.05)
                print(f"Curriculum: Difficulty decreased to {self.difficulty_level:.2f}")
                
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # GUI模式自动渲染
            pass
            
    def close(self):
        """关闭环境"""
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
            
    def get_info(self):
        """获取环境信息（用于调试和分析）"""
        return {
            'joint_positions': self._get_joint_positions(),
            'joint_velocities': self._get_joint_velocities(),
            'joint_torques': self._get_joint_torques(),
            'end_effector_pos': self._get_end_effector_pos(),
            'target_position': self.target_position,
            'distance': self._get_distance_to_target(),
            'jacobian_condition': self._compute_jacobian_condition(),
            'success_rate': self.success_count / max(1, self.current_step),
            'collision_rate': self.collision_count / max(1, self.current_step),
            'singularity_rate': self.singularity_count / max(1, self.current_step)
        }