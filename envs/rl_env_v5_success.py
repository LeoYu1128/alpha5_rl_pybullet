import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pybullet as p
import pybullet_data
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
# from envs.test_joint5 import GripperController  # ✅ 注释掉避免导入错误

class AlphaReachEnv(gym.Env):
    """
    水下Alpha机械臂到达任务环境
    任务:在水下环境中控制4个主要关节,让末端执行器到达目标位置
    考虑流体阻力、浮力、水流干扰等水下物理特性
    
    ⚠️ 关键修改:
    1. 添加真实角度<->URDF角度转换
    2. 加载URDF后立即设置到home位置
    3. reset时使用URDF home位置初始化
    """
    
    def __init__(self, render_mode=None, max_steps=500, reward_type='dense',
                 enable_target_drift=None, enable_domain_randomization=None, enable_curriculum=None):
        # ✅ train_v8兼容性: 接受但忽略这些参数 (v5环境不支持这些功能)
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        self.physics_client = None
        self.reward_type = reward_type
        
        # ============ 关键修改1: 真实机械臂的home位置定义 ============
        # 真实机械臂的home位置(编码器读数)
        self.real_home_positions = [
            np.radians(2.34),
            np.radians(87.8),
            np.radians(1.0),
            np.radians(0.1)
        ]
        
        # URDF中的home位置(PyBullet中的角度)
        self.urdf_home_positions = [
            np.radians(2.34),
            np.radians(0),      # URDF中joint_2=0时直立
            np.radians(1.0),
            np.radians(0.1)
        ]
        
        # 角度偏移(真实->URDF需要减去这个值)
        self.angle_offset = np.array([0, np.radians(87.8), 0, 0])
        
        # 夹爪home位置
        self.home_gripper_position = 0.0014  # 1.4mm
        
        # 水下环境参数
        self.water_density = 1000.0
        self.gravity = -9.81
        self.base_height = 0.1
        self.workspace_radius = 0.4
        
        # 真实机械臂质量参数
        self.urdf_total_mass = 1.52
        self.target_actual_mass = 1.36
        self.target_underwater_mass = 0.9
        
        buoyancy_mass = self.target_actual_mass - self.target_underwater_mass
        self.buoyancy_compensation_ratio = buoyancy_mass / self.target_actual_mass
        self.robot_mass_scale = self.target_actual_mass / self.urdf_total_mass
        
        # 流体动力学参数
        self.drag_coefficient = 0.5
        self.added_mass_coefficient = 0.3
        self.buoyancy_enabled = True
        
        # 水流参数
        self.x_current = np.random.uniform(-0.08,0.08)
        self.y_current = np.random.uniform(-0.05,0.05)
        self.z_current = np.random.uniform(-0.03,0.03)
        self.current_velocity = np.array([self.x_current, self.y_current, self.z_current])
        self.current_variation = True
        self.turbulence_strength = 0.01
        
        print("\n" + "="*70)
        print("水下机械臂环境初始化 - 基于真实机械臂配置")
        print("="*70)
        print(f"真实Home位置(编码器): {np.degrees(self.real_home_positions)}")
        print(f"URDF Home位置(内部):  {np.degrees(self.urdf_home_positions)}")
        print(f"角度偏移(Real-URDF):  {np.degrees(self.angle_offset)}")
        print(f"夹爪Home位置:         {self.home_gripper_position*1000:.1f}mm")
        print("="*70 + "\n")

        # 连接物理引擎
        self._connect_physics()
        self._setup_underwater_scene()
        self._analyze_robot()

        # 奖励参数
        self.distance_scale = 1.0
        self.progress_scale = 1.2
        self.control_penalty = 0.01
        self.velocity_penalty = 0.02
        self.time_penalty = 0.005
        self.success_bonus = 5.0
        self.success_threshold = 0.05
        self.previous_distance = None

        # 动作空间: 4个关节 + 1个夹爪
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(5,),
            dtype=np.float32
        )
        
        # 观察空间: 4关节位置 + 4关节速度 + 3末端位置 + 3目标位置 + 1夹爪
        obs_dim = 15
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print(f"动作空间: {self.action_space}")
        print(f"观察空间: {self.observation_space.shape}")
        print(f"水流速度: {self.current_velocity}\n")
    
    def real_to_urdf(self, real_angles):
        """真实机械臂角度 -> URDF角度(用于控制PyBullet)"""
        return np.array(real_angles) - self.angle_offset
    
    def urdf_to_real(self, urdf_angles):
        """URDF角度 -> 真实机械臂角度(用于显示)"""
        return np.array(urdf_angles) + self.angle_offset
    
    def _connect_physics(self):
        """连接PyBullet物理引擎"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
        
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, 
                                      physicsClientId=self.physics_client)
            p.resetDebugVisualizerCamera(2.0, 30, -20, [0, 0, 0.3], 
                                         physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), 
                                  physicsClientId=self.physics_client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
    
    def _setup_underwater_scene(self):
        """设置水下仿真场景"""
        # 加载海底地面
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.1, 0.2, 0.4, 1.0],
                           physicsClientId=self.physics_client)
        
        # ============ 关键修改2: 加载URDF并立即设置home位置 ============
        try:
            from pathlib import Path
            this_dir = Path(__file__).resolve().parent
            proj_root = this_dir.parent
        except:
            this_dir = None
            proj_root = None

        robot_paths = [
            "alpha_robot_for_pybullet.urdf",
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf",
        ]
        
        if proj_root is not None:
            robot_paths.extend([
                str((proj_root / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
                str((proj_root.parent / "alpha_description/urdf/alpha_robot_for_pybullet.urdf").resolve()),
            ])
        
        self.robot_id = None
        for robot_path in robot_paths:
            if os.path.exists(robot_path):
                try:
                    self.robot_id = p.loadURDF(
                        robot_path,
                        basePosition=[0, 0, self.base_height],
                        useFixedBase=True,
                        flags=p.URDF_USE_SELF_COLLISION,
                        physicsClientId=self.physics_client
                    )
                    print(f"✅ 加载URDF成功: {robot_path}")
                    
                    # ⚠️ 立即设置关节到home位置(使用URDF角度)
                    print("   立即设置home位置...")
                    num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
                    for i in range(num_joints):
                        joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                        joint_name = joint_info[1].decode('utf-8')
                        
                        if joint_name == 'joint_1':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[0],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_2':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[1],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_3':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[2],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_4':
                            p.resetJointState(self.robot_id, i, self.urdf_home_positions[3],
                                            physicsClientId=self.physics_client)
                        elif joint_name == 'joint_5':
                            p.resetJointState(self.robot_id, i, self.home_gripper_position,
                                            physicsClientId=self.physics_client)
                    
                    # 让物理引擎稳定一下
                    for _ in range(100):
                        p.stepSimulation(physicsClientId=self.physics_client)
                    
                    print("   ✅ Home位置设置完成")
                    break
                    
                except Exception as e:
                    print(f"   加载{robot_path}失败: {e}")
                    continue
        
        if self.robot_id is None:
            raise FileNotFoundError("找不到Alpha机械臂URDF文件")
        
        self._add_underwater_decorations()

    def _add_underwater_decorations(self):
        """添加水下装饰物"""
        try:
            for i in range(3):
                x = np.random.uniform(-0.8, 0.8)
                y = np.random.uniform(-0.8, 0.8)
                z = np.random.uniform(0.1, 0.5)
                visual_shape = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.05,
                    rgbaColor=[0.1, 0.4, 0.2, 0.8],
                    physicsClientId=self.physics_client
                )
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape,
                                 basePosition=[x, y, z], physicsClientId=self.physics_client)
        except:
            pass
    
    def _analyze_robot(self):
        """分析机器人结构"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        self.joint_info = {}
        self.main_joint_indices = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            self.joint_info[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower': lower_limit,
                'upper': upper_limit
            }
            
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
        
        print(f"主要控制关节: {len(self.main_joint_indices)}个")
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            print(f"  [{i}] {joint['name']}: [{joint['lower']:.2f}, {joint['upper']:.2f}]")
        
        # 找末端执行器
        self.tcp_index = None
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            if 'tcp' in joint_name.lower():
                self.tcp_index = i
                break
        
        if self.tcp_index is None:
            self.tcp_index = num_joints - 1
        
        print(f"末端执行器索引: {self.tcp_index}")
        
        # 初始化夹爪
        try:
            self.gripper = GripperController(self.robot_id)
            print("✅ 夹爪控制器初始化成功")
        except Exception as e:
            print(f"⚠️ 夹爪控制器初始化失败: {e}")
            self.gripper = None
        
        self._setup_underwater_dynamics()
    
    def _setup_underwater_dynamics(self):
        """设置水下动力学特性"""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        self.link_masses = {}
        self.link_indices = []
        
        for i in range(-1, num_joints):
            try:
                dynamics_info = p.getDynamicsInfo(self.robot_id, i, physicsClientId=self.physics_client)
                urdf_mass = dynamics_info[0]
                
                if urdf_mass > 0:
                    real_mass = urdf_mass * self.robot_mass_scale
                    self.link_masses[i] = real_mass
                    self.link_indices.append(i)
                    
                    p.changeDynamics(self.robot_id, i, mass=real_mass,
                                   physicsClientId=self.physics_client)
                
                if i >= 0:
                    p.changeDynamics(self.robot_id, i,
                                   linearDamping=2.0, angularDamping=2.0, jointDamping=0.5,
                                   physicsClientId=self.physics_client)
                else:
                    p.changeDynamics(self.robot_id, i,
                                   linearDamping=1.0, angularDamping=1.0,
                                   physicsClientId=self.physics_client)
            except:
                pass
    
    def _update_current_velocity(self):
        """更新水流速度"""
        if self.current_variation:
            time_factor = self.current_step * 0.01
            base_current = self.current_velocity.copy()
            periodic_variation = np.array([
                0.01 * np.sin(time_factor),
                0.01 * np.cos(time_factor * 1.5),
                0.01 * np.sin(time_factor * 0.5)
            ])
            turbulence = np.random.normal(0, self.turbulence_strength, 3)
            self.current_velocity_actual = base_current + periodic_variation + turbulence
        else:
            self.current_velocity_actual = self.current_velocity.copy()
    
    def _apply_underwater_forces(self):
        """应用水下力"""
        self._update_current_velocity()
        
        # 1. 浮力补偿
        if self.buoyancy_enabled:
            for link_idx in self.link_indices:
                try:
                    mass = self.link_masses.get(link_idx, 0)
                    if mass > 0:
                        gravity_force = mass * abs(self.gravity)
                        buoyancy_force = gravity_force * self.buoyancy_compensation_ratio
                        p.applyExternalForce(
                            self.robot_id, link_idx,
                            forceObj=[0, 0, buoyancy_force],
                            posObj=[0, 0, 0],
                            flags=p.LINK_FRAME,
                            physicsClientId=self.physics_client
                        )
                except:
                    pass
        
        # 2. 流体阻力
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        key_links = [self.tcp_index]
        for i, joint_idx in enumerate(self.main_joint_indices[:3]):
            if joint_idx < num_joints:
                key_links.append(joint_idx)
        
        for link_idx in key_links:
            try:
                link_state = p.getLinkState(self.robot_id, link_idx, computeLinkVelocity=1,
                                           physicsClientId=self.physics_client)
                link_velocity = np.array(link_state[6])
                relative_velocity = link_velocity - self.current_velocity_actual
                speed = np.linalg.norm(relative_velocity)
                
                if speed > 0.001:
                    characteristic_area = 0.008 if link_idx == self.tcp_index else 0.015
                    drag_magnitude = 0.5 * self.water_density * self.drag_coefficient * \
                                    characteristic_area * speed ** 2
                    
                    if speed > 1e-6:
                        drag_direction = -relative_velocity / speed
                        drag_force = drag_direction * drag_magnitude
                    else:
                        drag_force = np.zeros(3)
                    
                    drag_force = np.clip(drag_force, -12.0, 12.0)
                    p.applyExternalForce(self.robot_id, link_idx, forceObj=drag_force.tolist(),
                                       posObj=[0, 0, 0], flags=p.LINK_FRAME,
                                       physicsClientId=self.physics_client)
            except:
                pass
        
        # 3. 水流推力
        if np.linalg.norm(self.current_velocity_actual) > 0.01:
            for link_idx in self.link_indices:
                try:
                    mass = self.link_masses.get(link_idx, 0)
                    if mass > 0:
                        apparent_mass = mass * (1 - self.buoyancy_compensation_ratio)
                        current_force = self.water_density * 0.005 * apparent_mass * self.current_velocity_actual
                        p.applyExternalForce(self.robot_id, link_idx, forceObj=current_force.tolist(),
                                           posObj=[0, 0, 0], flags=p.WORLD_FRAME,
                                           physicsClientId=self.physics_client)
                except:
                    pass
        
        # # 4. 关节阻尼
        # for joint_idx in self.main_joint_indices:
        #     try:
        #         joint_state = p.getJointState(self.robot_id, joint_idx,
        #                                      physicsClientId=self.physics_client)
        #         joint_velocity = joint_state[1]
                
        #         if abs(joint_velocity) > 0.05:
        #             damping_torque = -0.15 * joint_velocity * abs(joint_velocity)
        #         else:
        #             damping_torque = -0.1 * joint_velocity
                
        #         damping_torque = np.clip(damping_torque, -2.0, 2.0)
        #         p.setJointMotorControl2(self.robot_id, joint_idx, p.TORQUE_CONTROL,
        #                                force=damping_torque, physicsClientId=self.physics_client)
        #     except:
        #         pass
    
    def _sample_target_position(self):
        """采样目标位置"""
        safe_radius_min = 0.12
        safe_radius_max = 0.35
        z_min = self.base_height
        z_max = 0.35 + self.base_height
        
        for _ in range(50):
            r = np.random.uniform(safe_radius_min, safe_radius_max)
            theta = np.random.uniform(-np.pi/2, np.pi/2)
            z = np.random.uniform(z_min, z_max)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.array([x, y, z], dtype=np.float32)
        
        return np.array([0.2, 0.0, 0.25], dtype=np.float32)
    
    def _update_target_position(self, dt=1./240.):
        """更新目标位置(受水流影响)"""
        if hasattr(self, 'target_position'):
            self.target_position += self.current_velocity_actual * dt
            x, y, z = self.target_position
            r = math.sqrt(x**2 + y**2)
            if r > self.workspace_radius:
                scale = self.workspace_radius / r
                x *= scale
                y *= scale
            z = np.clip(z, self.base_height, 0.35 + self.base_height)
            self.target_position = np.array([x, y, z], dtype=np.float32)
    
    def _update_target_visual(self):
        """更新目标视觉标记"""
        if hasattr(self, 'target_visual_id') and self.target_visual_id is not None:
            try:
                p.resetBasePositionAndOrientation(
                    self.target_visual_id, self.target_position, [0, 0, 0, 1],
                    physicsClientId=self.physics_client
                )
            except:
                pass
    
    def _get_joint_positions(self):
        """获取主要关节位置"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx,
                                         physicsClientId=self.physics_client)
            positions.append(joint_state[0])
        return np.array(positions, dtype=np.float32)
    
    def _get_joint_velocities(self):
        """获取主要关节速度"""
        velocities = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx,
                                         physicsClientId=self.physics_client)
            velocities.append(joint_state[1])
        return np.array(velocities, dtype=np.float32)
    
    def _get_end_effector_position(self):
        """获取末端执行器位置"""
        link_state = p.getLinkState(self.robot_id, self.tcp_index,
                                    physicsClientId=self.physics_client)
        return np.array(link_state[0], dtype=np.float32)
    
    # def _apply_action(self, action):
    #     """应用动作"""
    #     current_positions = self._get_joint_positions()
        
    #     arm_action = action[:4]
    #     gripper_action = action[4]
        
    #     scaled_action = arm_action * 0.15
    #     target_positions = current_positions + scaled_action
        
    #     for i, joint_idx in enumerate(self.main_joint_indices):
    #         joint = self.joint_info[joint_idx]
    #         target_positions[i] = np.clip(target_positions[i], joint['lower'], joint['upper'])
        
    #     control_torque = 9.0 * 3.0  # 27 Nm
        
    #     for i, joint_idx in enumerate(self.main_joint_indices):
    #         p.setJointMotorControl2(
    #             self.robot_id, joint_idx, p.POSITION_CONTROL,
    #             targetPosition=target_positions[i],
    #             maxVelocity=0.5,
    #             force=control_torque,
    #             physicsClientId=self.physics_client
    #         )
        
    #     if self.gripper is not None:
    #         self.gripper.control(gripper_action)
    
    def _apply_action(self, action):
        """应用动作 - 修复版"""
        current_positions = self._get_joint_positions()
        
        arm_action = action[:4]
        gripper_action = action[4]
        
        # ✅ 修复1: 自适应动作缩放
        ee_pos = self._get_end_effector_position()
        current_distance = np.linalg.norm(ee_pos - self.target_position)
        
        if current_distance > 0.15:
            scale = 0.3      # 远距离用大步长
        elif current_distance > 0.08:
            scale = 0.15     # 中距离用中步长  
        else:
            scale = 0.08     # 近距离用小步长
        
        scaled_action = arm_action * scale
        target_positions = current_positions + scaled_action
        
        # 限制到关节范围
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            target_positions[i] = np.clip(
                target_positions[i], 
                joint['lower'], 
                joint['upper']
            )
        
        # ✅ 修复2: 不同关节使用不同力矩 (关键!)
        control_torques = [600.0, 300.0, 200.0, 160.0]
        
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                maxVelocity=2.0,              # ✅ 修复3: 提高最大速度
                force=control_torques[i],     # ✅ 使用分级力矩
                physicsClientId=self.physics_client
            )
        
        if self.gripper is not None:
            self.gripper.control(gripper_action)

    def _compute_reward(self, action):
        """计算奖励"""
        achieved_goal = self._get_end_effector_position()
        desired_goal = self.target_position
        
        distance = float(np.linalg.norm(achieved_goal - desired_goal))
        prev_distance = self.previous_distance if self.previous_distance is not None else distance
        distance_delta = prev_distance - distance
        
        shaped_distance = -self.distance_scale * distance
        progress_reward = self.progress_scale * distance_delta
        control_cost = self.control_penalty * float(np.linalg.norm(action) ** 2)
        velocity_cost = self.velocity_penalty * float(np.linalg.norm(self._get_joint_velocities()))
        time_cost = self.time_penalty
        
        success = distance < self.success_threshold
        
        total_reward = shaped_distance + progress_reward - control_cost - velocity_cost - time_cost
        success_bonus = self.success_bonus if success else 0.0
        total_reward += success_bonus
        
        self.previous_distance = distance
        
        reward_terms = {
            'reward_distance': shaped_distance,
            'reward_progress': progress_reward,
            'reward_control': -control_cost,
            'reward_velocity': -velocity_cost,
            'reward_time': -time_cost,
            'reward_success': success_bonus
        }
        
        return float(total_reward), bool(success), reward_terms
    
    def _get_observation(self):
        """获取观察"""
        joint_positions = self._get_joint_positions()
        joint_velocities = self._get_joint_velocities()
        ee_position = self._get_end_effector_position()
        
        if self.gripper is not None:
            gripper_state = self.gripper.get_state()
            gripper_pose = gripper_state['normalized']
        else:
            gripper_pose = 0.0
        
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            self.target_position,
            [gripper_pose]
        ]).astype(np.float32)
        
        return observation
    
    def _create_target_visual(self):
        """创建目标视觉标记"""
        if hasattr(self, 'target_visual_id'):
            try:
                p.removeBody(self.target_visual_id, physicsClientId=self.physics_client)
            except:
                pass
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.01,
            rgbaColor=[1, 0.5, 0, 1.0],
            physicsClientId=self.physics_client
        )
        
        self.target_visual_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.target_position,
            physicsClientId=self.physics_client
        )
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        if self.physics_client is None:
            self._connect_physics()
            self._setup_underwater_scene()
            self._analyze_robot()
        
        self.current_step = 0
        
        # ============ 关键修改3: 使用URDF home位置初始化 ============
        # 基于URDF home位置添加小范围扰动
        init_positions = []
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.joint_info[joint_idx]
            home_pos = self.urdf_home_positions[i]  # 使用URDF home位置
            range_width = 0.1  # ±0.1弧度扰动
            
            init_pos = np.random.uniform(
                max(joint['lower'], home_pos - range_width),
                min(joint['upper'], home_pos + range_width)
            )
            init_positions.append(init_pos)
        
        # 设置关节初始位置(URDF角度)
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(
                self.robot_id, joint_idx, init_positions[i],
                physicsClientId=self.physics_client
            )
        
        # 设置夹爪到home位置
        if hasattr(self, 'gripper') and self.gripper is not None:
            # 找到joint_5并设置
            for i in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
                joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                if joint_info[1].decode('utf-8') == 'joint_5':
                    p.resetJointState(self.robot_id, i, self.home_gripper_position,
                                    physicsClientId=self.physics_client)
                    break
            
            # 控制夹爪到home位置
            gripper_home_normalized = (self.home_gripper_position - 0.00137) / (0.0133 - 0.00137)
            self.gripper.control(gripper_home_normalized)
        
        # 生成目标位置
        self.target_position = self._sample_target_position()
        
        # 重置水流
        self.current_velocity_actual = self.current_velocity.copy()
        
        # 创建目标视觉
        self._create_target_visual()
        
        # 稳定仿真
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        self.previous_distance = float(np.linalg.norm(
            self._get_end_effector_position() - self.target_position
        ))
        
        observation = self._get_observation()
        
        # ============ 关键修改4: info中包含角度信息 ============
        # 获取当前URDF角度并转换为真实角度用于调试
        current_urdf_angles = self._get_joint_positions()
        current_real_angles = self.urdf_to_real(current_urdf_angles)
        
        info = {
            'target_position': self.target_position.copy(),
            'initial_distance': self.previous_distance,
            'urdf_home_positions': self.urdf_home_positions.copy(),
            'real_home_positions': self.real_home_positions.copy(),
            'current_urdf_angles': current_urdf_angles.copy(),
            'current_real_angles': current_real_angles.copy()
        }
        
        return observation, info
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        action = np.array(action, dtype=np.float32)
        
        self._apply_action(action)
        
        # 应用水下力并仿真
        for _ in range(4):
            self._apply_underwater_forces()
            p.stepSimulation(physicsClientId=self.physics_client)
        
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        self._update_target_position(dt=4./240.)
        self._update_target_visual()
        
        observation = self._get_observation()
        reward, success, reward_terms = self._compute_reward(action)
        
        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_steps)
        
        ee_pos = self._get_end_effector_position()
        current_distance = float(np.linalg.norm(ee_pos - self.target_position))
        
        info = {
            'success': success,
            'distance': current_distance,
            'is_success': success,
            'current_velocity': self.current_velocity_actual.copy(),
            'underwater': True
        }
        info.update(reward_terms)
        
        return observation, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
            self.physics_client = None
    
    def render(self):
        """渲染"""
        pass

# 测试环境
if __name__ == "__main__":
    print("="*70)
    print("水下Alpha机械臂环境测试 - 基于真实机械臂配置")
    print("="*70)
    
    env = AlphaReachEnv(render_mode="human")
    
    obs, info = env.reset()
    
    print(f"\n初始化信息:")
    print(f"  观察维度: {obs.shape}")
    print(f"  目标位置: {info['target_position']}")
    print(f"  初始距离: {info['initial_distance']:.3f}m")
    print(f"\n角度信息:")
    print(f"  URDF Home: {np.degrees(info['urdf_home_positions'])}")
    print(f"  Real Home: {np.degrees(info['real_home_positions'])}")
    print(f"  当前URDF角度: {np.degrees(info['current_urdf_angles'])}")
    print(f"  当前真实角度: {np.degrees(info['current_real_angles'])}")
    
    print(f"\n开始测试...")
    
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if step % 20 == 0:
            # 获取当前角度并转换
            current_urdf = env._get_joint_positions()
            current_real = env.urdf_to_real(current_urdf)
            
            print(f"\n步数 {step}:")
            print(f"  奖励={reward:.3f}, 距离={info['distance']:.3f}m, 成功={info['success']}")
            print(f"  当前真实角度: {np.degrees(current_real).astype(int)}°")
        
        if done:
            print(f"\n回合结束: 成功={info['success']}, 最终距离={info['distance']:.3f}m")
            obs, info = env.reset()
            print(f"  重置到真实角度: {np.degrees(info['current_real_angles']).astype(int)}°")
    
    env.close()
    print("\n测试完成!")