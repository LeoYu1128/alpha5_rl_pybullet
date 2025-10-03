import pybullet as p
import pybullet_data
import time
import math

# ============================================================
# 核心：夹爪控制器类
# ============================================================
class GripperController:
    """
    这个类的作用：
    1. 你只需要告诉它"夹爪开到什么程度"（0到1之间的数字）
    2. 它会自动计算并控制3个关节（joint_5, finger1, finger2）
    3. 模拟真实机器人的mimic机制
    """
    
    def __init__(self, robot_id):
        self.robot_id = robot_id
        
        # 步骤1：找到3个关节的索引号
        self.find_joint_indices()
        
        # 步骤2：设置mimic参数（从URDF中抄来的）
        self.setup_mimic_parameters()
        
        print("✅ 夹爪控制器初始化成功")
        print(f"   joint_5 索引: {self.joint_5_index}")
        print(f"   finger1 索引: {self.finger1_index}")
        print(f"   finger2 索引: {self.finger2_index}")
    
    def find_joint_indices(self):
        """找到我们需要控制的3个关节"""
        print("\n🔍 查找关节索引...")
        
        # 遍历机器人的所有关节
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')  # 关节名字
            
            # 找到joint_5
            if joint_name == "joint_5":
                self.joint_5_index = i
                print(f"   找到 joint_5: 索引 {i}")
            
            # 找到finger1
            elif joint_name == "standard_jaws_rs1_130_joint":
                self.finger1_index = i
                print(f"   找到 finger1: 索引 {i}")
            
            # 找到finger2
            elif joint_name == "standard_jaws_rs1_139_joint":
                self.finger2_index = i
                print(f"   找到 finger2: 索引 {i}")
    
    def setup_mimic_parameters(self):
        """设置mimic关系的参数"""
        # 从URDF中抄来的数值
        self.joint_5_min = 0.001   # joint_5的最小值（米）
        self.joint_5_max = 0.013   # joint_5的最大值（米）
        self.multiplier = 30    # mimic的倍数
        
        print("\n📋 Mimic参数:")
        print(f"   joint_5 范围: {self.joint_5_min} ~ {self.joint_5_max} 米")
        print(f"   倍数: {self.multiplier}")
    
    def control(self, normalized_position):
        """
        核心函数：控制夹爪
        
        参数:
            normalized_position: 0到1之间的数字
                - 0.0 = 完全关闭
                - 0.5 = 半开
                - 1.0 = 完全打开
        
        这个函数做3件事:
        1. 计算joint_5应该移动到哪里
        2. 根据mimic关系计算手指应该转到什么角度
        3. 发送命令给PyBullet
        """
        
        # ===== 第1步：计算joint_5的位置 =====
        # 将0~1映射到0.001~0.013
        joint_5_position = self.joint_5_min + normalized_position * (self.joint_5_max - self.joint_5_min)
        
        # ===== 第2步：根据mimic计算手指角度 =====
        # 公式：finger_angle = joint_5_position × multiplier
        finger_angle = joint_5_position * self.multiplier
        
        # ===== 第3步：发送命令到PyBullet =====
        # 控制joint_5
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.joint_5_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_5_position,
            force=10,
            maxVelocity=0.5
        )
        
        # 控制finger1
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.finger1_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=finger_angle,
            force=10,
            maxVelocity=1.0
        )
        
        # 控制finger2（和finger1一样）
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.finger2_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=finger_angle,
            force=10,
            maxVelocity=1.0
        )
    
    def get_state(self):
        """获取夹爪当前状态"""
        # 读取joint_5的状态
        joint_5_state = p.getJointState(self.robot_id, self.joint_5_index)
        current_position = joint_5_state[0]
        
        # 转换成归一化值
        normalized = (current_position - self.joint_5_min) / (self.joint_5_max - self.joint_5_min)
        
        return {
            'normalized': normalized,
            'joint_5_position': current_position,
            'joint_5_velocity': joint_5_state[1]
        }


# ============================================================
# 主程序：演示如何使用
# ============================================================

def main():
    print("=" * 60)
    print("夹爪控制器演示程序")
    print("=" * 60)
    
    # 1. 初始化PyBullet
    print("\n🚀 启动PyBullet...")
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 2. 加载环境
    print("📦 加载地面...")
    plane_id = p.loadURDF("plane.urdf")
    
    print("🤖 加载机器人...")
    # ⚠️ 这里改成你的URDF文件名
    robot_id = p.loadURDF("alpha_description/urdf/alpha_robot_for_pybullet.urdf", [0, 0, 0], useFixedBase=True)
    
    # 3. 创建夹爪控制器
    print("\n" + "=" * 60)
    gripper = GripperController(robot_id)
    print("=" * 60)
    
    # 4. 测试：让夹爪循环开合
    print("\n" + "=" * 60)
    print("🧪 测试1：基本开合")
    print("=" * 60)
    
    for cycle in range(3):
        print(f"\n--- 循环 {cycle + 1} ---")
        
        # 打开夹爪
        print("  📖 打开夹爪到100%...")
        gripper.control(1.0)  # 1.0 = 完全打开
        
        # 等待运动完成（1秒 = 240步）
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 检查状态
        state = gripper.get_state()
        print(f"     当前位置: {state['normalized']:.3f} (归一化)")
        print(f"     joint_5: {state['joint_5_position']*1000:.2f} mm")
        
        time.sleep(0.5)
        
        # 关闭夹爪
        print("  📕 关闭夹爪到0%...")
        gripper.control(0.0)  # 0.0 = 完全关闭
        
        # 等待运动完成
        for _ in range(240):
            p.stepSimulation()
            time.sleep(1./240.)
        
        # 检查状态
        state = gripper.get_state()
        print(f"     当前位置: {state['normalized']:.3f} (归一化)")
        print(f"     joint_5: {state['joint_5_position']*1000:.2f} mm")
        
        time.sleep(0.5)
    
    # 5. 测试：精确控制
    print("\n" + "=" * 60)
    print("🧪 测试2：精确位置控制")
    print("=" * 60)
    
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    for pos in positions:
        print(f"\n  设置位置: {pos:.2f}")
        gripper.control(pos)
        
        # 等待
        for _ in range(120):
            p.stepSimulation()
            time.sleep(1./240.)
        
        state = gripper.get_state()
        print(f"     实际位置: {state['normalized']:.3f}")
    
    # 6. 测试：平滑运动
    print("\n" + "=" * 60)
    print("🧪 测试3：平滑连续运动")
    print("按 Ctrl+C 停止")
    print("=" * 60)
    
    try:
        t = 0
        while True:
            # 用正弦波生成平滑的开合运动
            # sin(t)的值在-1到1之间，我们转换到0到1
            position = 0.5 + 0.5 * math.sin(t)
            
            gripper.control(position)
            
            # 每2秒打印一次状态
            if int(t * 100) % 200 == 0:
                state = gripper.get_state()
                print(f"  位置: {state['normalized']:.3f}")
            
            p.stepSimulation()
            time.sleep(1./240.)
            t += 1./240.
            
    except KeyboardInterrupt:
        print("\n\n✅ 测试完成！")
    
    # 7. 清理
    p.disconnect()


if __name__ == "__main__":
    main()