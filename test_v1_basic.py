import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class AlphaRobotController:
    """
    Alpha机械臂专用控制器
    针对你的URDF优化，只控制4个主要关节
    """
    
    def __init__(self, render_mode="GUI"):
        print("初始化Alpha机械臂控制器...")
        
        # 连接PyBullet
        if render_mode == "GUI":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # 加载环境
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载Alpha机械臂
        self.robot_id = self._load_alpha_robot()
        
        # 定义关节映射
        self._setup_joint_mapping()
        
        print(f"Alpha机械臂控制器初始化完成")
        self._print_robot_info()
    
    def _load_alpha_robot(self):
        """加载Alpha机械臂"""
        robot_paths = [
            "alpha_robot_for_pybullet.urdf",
            "alpha_description/urdf/alpha_robot_for_pybullet.urdf",
            "../alpha_description/urdf/alpha_robot_for_pybullet.urdf"
        ]
        
        for robot_path in robot_paths:
            if os.path.exists(robot_path):
                try:
                    robot_id = p.loadURDF(
                        robot_path, 
                        basePosition=[0, 0, 0.1], 
                        useFixedBase=True
                    )
                    print(f"成功加载Alpha机械臂: {robot_path}")
                    return robot_id
                except Exception as e:
                    print(f"加载失败: {e}")
                    continue
        
        raise FileNotFoundError("找不到Alpha机械臂URDF文件")
    
    def _setup_joint_mapping(self):
        """设置关节映射"""
        num_joints = p.getNumJoints(self.robot_id)
        
        # 存储所有关节信息
        self.all_joints = {}
        self.main_joint_indices = []
        self.gripper_joint_indices = []
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            self.all_joints[i] = {
                'name': joint_name,
                'type': joint_type,
                'lower': lower_limit,
                'upper': upper_limit
            }
            
            # 分类关节
            if joint_name in ['joint_1', 'joint_2', 'joint_3', 'joint_4']:
                self.main_joint_indices.append(i)
            elif 'jaw' in joint_name.lower() or joint_name == 'joint_5':
                self.gripper_joint_indices.append(i)
        
        # 主要关节的描述和限制
        self.main_joint_info = {
            'joint_1': {'desc': '基座旋转', 'range': '±176°'},
            'joint_2': {'desc': '肩关节', 'range': '±100°'},
            'joint_3': {'desc': '肘关节', 'range': '±92°'},
            'joint_4': {'desc': '腕关节', 'range': '±45°'}
        }
    
    def _print_robot_info(self):
        """打印机器人信息"""
        print("\n" + "="*60)
        print("Alpha机械臂关节信息")
        print("="*60)
        
        print("主要控制关节:")
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.all_joints[joint_idx]
            info = self.main_joint_info.get(joint['name'], {'desc': '未知', 'range': '未知'})
            print(f"  [{i}] {joint['name']}: {info['desc']} {info['range']}")
            print(f"      关节索引: {joint_idx}")
            print(f"      限制: [{joint['lower']:.2f}, {joint['upper']:.2f}] 弧度")
        
        print(f"\n抓手关节: {len(self.gripper_joint_indices)}个")
        for joint_idx in self.gripper_joint_indices:
            joint = self.all_joints[joint_idx]
            print(f"  {joint['name']}: [{joint['lower']:.2f}, {joint['upper']:.2f}]")
        
        print(f"\n总关节数: {len(self.all_joints)}")
        print(f"主控制关节数: {len(self.main_joint_indices)}")
    
    def get_main_joint_positions(self):
        """获取主要关节的当前位置"""
        positions = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
        return np.array(positions)
    
    def get_main_joint_velocities(self):
        """获取主要关节的当前速度"""
        velocities = []
        for joint_idx in self.main_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            velocities.append(joint_state[1])
        return np.array(velocities)
    
    def apply_position_control(self, action, step_size=1.0):
        """
        应用位置控制
        action: 4维数组，对应4个主要关节的位置增量（弧度）
        step_size: 步长倍数
        """
        if len(action) != 4:
            raise ValueError(f"action必须是4维，得到{len(action)}维")
        
        # 获取当前位置
        current_pos = self.get_main_joint_positions()
        
        # 计算目标位置
        target_pos = current_pos + np.array(action) * step_size
        
        # 应用关节限制
        for i, joint_idx in enumerate(self.main_joint_indices):
            joint = self.all_joints[joint_idx]
            target_pos[i] = np.clip(target_pos[i], joint['lower'], joint['upper'])
        
        # 执行位置控制
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos[i],
                maxVelocity=1.0,
                force=500
            )
        
        return target_pos
    
    def apply_velocity_control(self, action):
        """
        应用速度控制
        action: 4维数组，对应4个主要关节的角速度（弧度/秒）
        """
        if len(action) != 4:
            raise ValueError(f"action必须是4维，得到{len(action)}维")
        
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.VELOCITY_CONTROL,
                targetVelocity=action[i],
                force=500
            )
    
    def apply_torque_control(self, action):
        """
        应用扭矩控制
        action: 4维数组，对应4个主要关节的扭矩（牛顿米）
        """
        if len(action) != 4:
            raise ValueError(f"action必须是4维，得到{len(action)}维")
        
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.TORQUE_CONTROL,
                force=action[i]
            )
    
    def control_gripper(self, open_close):
        """
        控制抓手
        open_close: 0.0=完全关闭, 1.0=完全打开
        """
        # 简化的抓手控制
        if self.gripper_joint_indices:
            for joint_idx in self.gripper_joint_indices:
                joint = self.all_joints[joint_idx]
                target_pos = joint['lower'] + (joint['upper'] - joint['lower']) * open_close
                
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    maxVelocity=1.0,
                    force=100
                )
    
    def reset_to_home(self):
        """重置到初始位置"""
        home_positions = [0.0, 0.0, 0.0, 0.0]  # 4个主要关节的初始位置
        
        for i, joint_idx in enumerate(self.main_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, home_positions[i])
        
        # 重置抓手
        for joint_idx in self.gripper_joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.0)
    
    def step_simulation(self, steps=1):
        """推进物理仿真"""
        for _ in range(steps):
            p.stepSimulation()
    
    def test_basic_movements(self):
        """测试基本运动"""
        print("\n开始基本运动测试...")
        
        test_actions = [
            ([3, 0, 0, 0], "基座旋转 +28.6°"),
            ([-0.9, 0, 0, 0], "基座旋转 -28.6°"),
            ([0, 0.9, 0, 0], "肩关节 +22.9°"),
            ([0, -0.9, 0, 0], "肩关节 -22.9°"),
            ([0, 0, 0.9, 0], "肘关节 +22.9°"),
            ([0, 0, -0.9, 0], "肘关节 -22.9°"),
            ([0, 0, 0, 0.9], "腕关节 +17.2°"),
            ([0, 0, 0, -0.9], "腕关节 -17.2°"),
            ([0.9, 0.9, 0.9, 0.9], "所有关节同时动作"),
            ([0, 0, 0, 0], "停止")
        ]
        
        for i, (action, description) in enumerate(test_actions):
            print(f"\n[{i+1}/{len(test_actions)}] {description}")
            print(f"动作: {action} (弧度)")
            print(f"角度: {np.degrees(action)} (度)")
            
            # 执行前状态
            pos_before = self.get_main_joint_positions()
            print(f"执行前: {np.degrees(pos_before).tolist()} (度)")
            
            # 执行动作
            target_pos = self.apply_position_control(action)
            
            # 仿真运行
            for _ in range(720):  # 0.5秒
                self.step_simulation()
                time.sleep(1./240.)
            
            # 执行后状态
            pos_after = self.get_main_joint_positions()
            print(f"执行后: {np.degrees(pos_after).tolist()} (度)")
            
            # 检查运动
            movement = np.abs(pos_after - pos_before)
            if np.any(movement > 0.01):  # 约0.6度
                print("✅ 检测到运动")
            else:
                print("❌ 未检测到明显运动")
            
            input("按Enter继续...")
    
    def test_gripper(self):
        """测试抓手"""
        print("\n测试抓手控制...")
        
        actions = [
            (1.0, "打开抓手"),
            (0.0, "关闭抓手"),
            (0.5, "半开抓手")
        ]
        
        for action, description in actions:
            print(f"\n{description}")
            self.control_gripper(action)
            
            for _ in range(60):  # 0.25秒
                self.step_simulation()
                time.sleep(1./240.)
            
            input("按Enter继续...")
    
    def close(self):
        """关闭连接"""
        p.disconnect(self.physics_client)

# 使用示例
if __name__ == "__main__":
    try:
        # 创建控制器
        controller = AlphaRobotController()
        
        # 重置到初始位置
        controller.reset_to_home()
        
        print("\n选择测试类型:")
        print("1. 基本运动测试")
        print("2. 抓手测试") 
        print("3. 完整测试")
        
        choice = input("请选择 (1/2/3): ").strip()
        
        if choice == "1":
            controller.test_basic_movements()
        elif choice == "2":
            controller.test_gripper()
        elif choice == "3":
            controller.test_basic_movements()
            controller.test_gripper()
        else:
            print("无效选择")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        if 'controller' in locals():
            controller.close()