#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import time
import os

class SliderOnlyViewer:
    def __init__(self):
        # 连接PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # 基本设置
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
        # 加载机械臂
        robot_path = "alpha_description/urdf/alpha_robot_for_pybullet.urdf"
        self.robot_id = p.loadURDF(robot_path, [0, 0, 0.4])
        print(f"机械臂加载成功！使用修正版URDF")
        
        # 获取关节信息
        self.setup_joints()
        
        # 创建滑桿
        self.create_sliders()
        
        # 添加预设按钮
        self.create_preset_buttons()
        
        print("\n使用说明:")
        print("1. 用右侧滑桿控制关节")
        print("2. 点击预设按钮快速切换姿态")
        print("3. 鼠标拖拽旋转视角")
        print("4. 修正版URDF应该在零位置正常工作")
        
    def setup_joints(self):
        """设置关节信息"""
        self.num_joints = p.getNumJoints(self.robot_id)
        self.controllable_joints = []
        self.joint_names = {}
        
        print(f"\n发现 {self.num_joints} 个关节:")
        
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            if joint_type in [0, 1]:  # 旋转或直线关节
                self.controllable_joints.append(i)
                self.joint_names[i] = joint_name
                joint_type_name = "旋转" if joint_type == 0 else "直线"
                print(f"  关节 {i}: {joint_name} ({joint_type_name})")
        
        print(f"共 {len(self.controllable_joints)} 个可控关节")
    
    def create_sliders(self):
        """创建滑桿控制"""
        self.sliders = {}
        
        # 更新的关节范围 - 基于官方XACRO限制
        joint_ranges = [
            (-3.054, 3.054, 0.0),      # 关节1 (axis_e): 基座旋转 - 零位置开始
            (-1.745, 1.745, 0.0),      # 关节2 (axis_d): 肩部俯仰 - 零位置开始
            (-1.618, 1.618, 0.0),      # 关节3 (axis_c): 肘部 - 零位置开始 
            (-0.785, 0.785, 0.0),      # 关节4 (axis_b): 手腕旋转 - 零位置开始
            (0.0013, 0.0133, 0.007),   # 关节5 (axis_a): 夹爪 - 中间位置开始
        ]
        
        print(f"\n创建滑桿控制器:")
        
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(joint_ranges):
                lower, upper, initial = joint_ranges[i]
            else:
                lower, upper, initial = -3.14, 3.14, 0.0
            
            joint_name = self.joint_names[joint_idx]
            
            # 创建滑桿
            slider_id = p.addUserDebugParameter(
                paramName=f"J{joint_idx}_{joint_name}",
                rangeMin=lower,
                rangeMax=upper,
                startValue=initial
            )
            
            self.sliders[joint_idx] = slider_id
            print(f"  滑桿 {i+1}: {joint_name} ({lower:.3f} - {upper:.3f}), 初始: {initial}")
    
    def create_preset_buttons(self):
        """创建预设按钮"""
        print(f"\n创建预设按钮:")
        
        # 预设1: 零位置 (修正版应该正常工作)
        self.preset1_button = p.addUserDebugParameter(
            "预设1_零位置", 0, 1, 0
        )
        
        # 预设2: 旧的"能工作"位置 (用于对比)
        self.preset2_button = p.addUserDebugParameter(
            "预设2_旧设置", 0, 1, 0
        )
        
        # 预设3: 安全位置
        self.preset3_button = p.addUserDebugParameter(
            "预设3_安全位置", 0, 1, 0
        )
        
        # 预设4: 测试位置
        self.preset4_button = p.addUserDebugParameter(
            "预设4_测试位置", 0, 1, 0
        )
        
        print("  按钮1: 零位置 [0.0, 0.0, 0.0, 0.0, 0.007] - 应该正常")
        print("  按钮2: 旧设置 [3.0, 3.0, 1.0, 0.0, 0.01] - 用于对比")
        print("  按钮3: 安全位置 [0.0, -0.5, 0.8, 0.0, 0.007]")
        print("  按钮4: 测试位置 [1.0, 0.5, -0.5, 0.2, 0.01]")
        
        # 记录按钮状态
        self.last_preset1 = 0
        self.last_preset2 = 0
        self.last_preset3 = 0
        self.last_preset4 = 0
    
    def check_preset_buttons(self):
        """检查预设按钮是否被点击"""
        # 检查预设1 - 零位置
        current1 = p.readUserDebugParameter(self.preset1_button)
        if current1 > self.last_preset1:
            self.apply_preset([0.0, 0.0, 0.0, 0.0, 0.007], "零位置 (修正版)")
        self.last_preset1 = current1
        
        # 检查预设2 - 旧设置
        current2 = p.readUserDebugParameter(self.preset2_button)
        if current2 > self.last_preset2:
            self.apply_preset([3.0, 3.0, 1.0, 0.0, 0.01], "旧设置 (对比)")
        self.last_preset2 = current2
        
        # 检查预设3 - 安全位置
        current3 = p.readUserDebugParameter(self.preset3_button)
        if current3 > self.last_preset3:
            self.apply_preset([0.0, -0.5, 0.8, 0.0, 0.007], "安全位置")
        self.last_preset3 = current3
        
        # 检查预设4 - 测试位置
        current4 = p.readUserDebugParameter(self.preset4_button)
        if current4 > self.last_preset4:
            self.apply_preset([1.0, 0.5, -0.5, 0.2, 0.01], "测试位置")
        self.last_preset4 = current4
    
    def apply_preset(self, positions, name):
        """应用预设姿态"""
        print(f"\n应用预设: {name}")
        
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(positions):
                target_pos = positions[i]
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    maxVelocity=1.0,
                    force=1000
                )
                print(f"  关节 {joint_idx}: {target_pos}")
        
        # 检查是否有碰撞或异常
        time.sleep(0.5)  # 等待机械臂移动
        self.check_robot_health()
    
    def check_robot_health(self):
        """检查机械臂健康状态"""
        # 检查所有关节位置是否合理
        has_issue = False
        
        for joint_idx in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            position = joint_state[0]
            
            # 检查是否有异常值
            if abs(position) > 10:  # 超过合理范围
                print(f"  警告: 关节 {joint_idx} 位置异常: {position}")
                has_issue = True
        
        if not has_issue:
            print("  ✓ 机械臂状态正常")
        
        return not has_issue
    
    def update_robot(self):
        """根据滑桿值更新机械臂"""
        for joint_idx, slider_id in self.sliders.items():
            target_pos = p.readUserDebugParameter(slider_id)
            
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_pos,
                maxVelocity=2.0,
                force=1000
            )
    
    def print_current_state(self):
        """打印当前状态"""
        positions = []
        print(f"\n当前关节位置:")
        
        for joint_idx in self.controllable_joints:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            position = joint_state[0]
            positions.append(position)
            
            joint_name = self.joint_names[joint_idx]
            if joint_idx == self.controllable_joints[-1]:  # 最后一个是夹爪
                print(f"  {joint_name}: {position:.6f} 米")
            else:
                degrees = position * 180 / 3.14159
                print(f"  {joint_name}: {position:.3f} 弧度 ({degrees:.1f}°)")
        
        print(f"Python数组: {positions}")
        
        # 检查健康状态
        self.check_robot_health()
        
        return positions
    
    def test_all_presets(self):
        """测试所有预设位置"""
        print(f"\n开始测试所有预设位置...")
        
        presets = [
            ([0.0, 0.0, 0.0, 0.0, 0.007], "零位置"),
            ([1.0, 0.5, -0.5, 0.2, 0.01], "测试位置1"),
            ([0.0, -0.5, 0.8, 0.0, 0.007], "安全位置"),
            ([3.0, 3.0, 1.0, 0.0, 0.01], "旧设置")
        ]
        
        for positions, name in presets:
            print(f"\n测试: {name}")
            self.apply_preset(positions, name)
            time.sleep(2)  # 等待稳定
            
        print(f"\n所有预设测试完成")
    
    def run(self):
        """运行主循环"""
        print(f"\n开始运行...")
        print("现在你可以:")
        print("- 调整滑桿观察机械臂变化")
        print("- 点击预设按钮测试不同姿态")
        print("- 重点测试零位置是否正常工作")
        print("- 按 't' 键自动测试所有预设")
        
        # 首先应用零位置进行初始测试
        print(f"\n初始测试: 应用零位置...")
        self.apply_preset([0.0, 0.0, 0.0, 0.0, 0.007], "初始零位置")
        
        last_print = time.time()
        
        try:
            while True:
                # 检查预设按钮
                self.check_preset_buttons()
                
                # 更新机械臂
                self.update_robot()
                
                # 每15秒打印一次当前状态
                if time.time() - last_print > 15:
                    self.print_current_state()
                    last_print = time.time()
                
                # 步进仿真
                p.stepSimulation()
                time.sleep(1/240)
                
        except KeyboardInterrupt:
            print(f"\n程序结束")
            final_positions = self.print_current_state()
            
            # 总结测试结果
            print(f"\n=== 测试总结 ===")
            print(f"最终位置: {final_positions}")
            if all(abs(pos) < 10 for pos in final_positions[:-1]):  # 排除夹爪
                print("✓ 修正版URDF工作正常")
            else:
                print("✗ 仍有问题，需要进一步调试")
                
        finally:
            p.disconnect()

def main():
    print("Alpha机械臂修正版测试器")
    print("=" * 40)
    print("测试目标: 验证修正版URDF的零位置是否正常")
    
    # 检查URDF文件是否存在
    urdf_path = "alpha_description/urdf/alpha_robot_for_pybullet_test.urdf"
    if not os.path.exists(urdf_path):
        print(f"错误: 找不到URDF文件: {urdf_path}")
        print("请确保:")
        print("1. URDF文件路径正确")
        print("2. mesh文件在正确位置")
        return
    
    viewer = SliderOnlyViewer()
    viewer.run()

if __name__ == "__main__":
    main()