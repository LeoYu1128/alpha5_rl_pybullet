"""
深度关节诊断工具
专门分析为什么joint_1, joint_2, joint_5无响应
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os

class DeepJointDiagnostic:
    """深度关节诊断"""
    
    def __init__(self):
        # 连接PyBullet
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # 加载地面
        self.plane_id = p.loadURDF("plane.urdf")
        
    def load_and_diagnose(self, urdf_path):
        """加载并深度诊断URDF"""
        print("🔍 深度关节诊断开始...")
        print("=" * 60)
        
        try:
            # 加载机器人
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0.1],
                useFixedBase=True
            )
            print(f"✅ URDF加载成功")
            
        except Exception as e:
            print(f"❌ URDF加载失败: {e}")
            return False
            
        # 深度分析每个关节
        self.analyze_all_joints()
        self.test_problematic_joints()
        self.check_joint_constraints()
        self.analyze_urdf_structure()
        
        return True
        
    def analyze_all_joints(self):
        """分析所有关节的详细信息"""
        print("\n🔧 详细关节分析:")
        print("-" * 50)
        
        num_joints = p.getNumJoints(self.robot_id)
        self.joint_data = {}
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            
            self.joint_data[i] = {
                'name': joint_name,
                'type': info[2],
                'axis': info[13],
                'parent_index': info[16],
                'limits': (info[8], info[9]),
                'max_force': info[10],
                'max_velocity': info[11],
                'damping': info[6],
                'friction': info[7]
            }
            
            # 详细输出
            print(f"\n关节 {i}: {joint_name}")
            print(f"  类型: {info[2]} ({self._get_joint_type_name(info[2])})")
            print(f"  轴向: {info[13]}")
            print(f"  父关节: {info[16]}")
            print(f"  限制: [{info[8]:.6f}, {info[9]:.6f}]")
            print(f"  最大力: {info[10]:.6f}")
            print(f"  最大速度: {info[11]:.6f}")
            print(f"  阻尼: {info[6]:.6f}")
            print(f"  摩擦: {info[7]:.6f}")
            
            # 🚨 检查可疑设置
            if info[2] == 0:  # REVOLUTE
                if abs(info[9] - info[8]) < 0.01:
                    print(f"  ⚠️  关节范围太小: {abs(info[9] - info[8]):.6f}")
                if info[11] > 1000:
                    print(f"  ⚠️  速度限制异常: {info[11]}")
                if info[10] < 0.1:
                    print(f"  ⚠️  最大力太小: {info[10]}")
                    
    def test_problematic_joints(self):
        """专门测试问题关节：1, 2, 5"""
        print("\n🧪 问题关节专项测试:")
        print("-" * 50)
        
        problem_joints = ['joint_1', 'joint_2', 'joint_5']
        
        for joint_name in problem_joints:
            # 找到关节索引
            joint_idx = self._find_joint_by_name(joint_name)
            if joint_idx is None:
                print(f"❌ 找不到关节: {joint_name}")
                continue
                
            print(f"\n🎯 测试关节: {joint_name} (索引 {joint_idx})")
            self._test_single_joint_deeply(joint_idx, joint_name)
            
    def _test_single_joint_deeply(self, joint_idx, joint_name):
        """深度测试单个关节"""
        
        # 1. 获取初始状态
        initial_state = p.getJointState(self.robot_id, joint_idx)
        initial_pos = initial_state[0]
        initial_vel = initial_state[1]
        
        print(f"  初始位置: {initial_pos:.6f}")
        print(f"  初始速度: {initial_vel:.6f}")
        
        # 2. 测试不同控制模式
        self._test_position_control(joint_idx, joint_name)
        self._test_velocity_control(joint_idx, joint_name)
        self._test_torque_control(joint_idx, joint_name)
        
        # 3. 测试关节约束
        self._test_joint_constraints(joint_idx, joint_name)
        
    def _test_position_control(self, joint_idx, joint_name):
        """测试位置控制"""
        print(f"    📍 测试位置控制...")
        
        joint_info = self.joint_data[joint_idx]
        lower, upper = joint_info['limits']
        
        # 安全的测试位置（25%和75%位置）
        test_positions = [
            lower + (upper - lower) * 0.25,
            lower + (upper - lower) * 0.75
        ]
        
        for test_pos in test_positions:
            print(f"      目标位置: {test_pos:.4f}")
            
            # 应用位置控制
            p.setJointMotorControl2(
                self.robot_id, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=test_pos,
                force=joint_info['max_force'] * 0.5
            )
            
            # 运行仿真
            for _ in range(120):  # 0.5秒
                p.stepSimulation()
                
            # 检查结果
            final_state = p.getJointState(self.robot_id, joint_idx)
            final_pos = final_state[0]
            movement = abs(final_pos - test_pos)
            
            print(f"      最终位置: {final_pos:.4f}")
            print(f"      误差: {movement:.4f}")
            
            if movement > 0.1:
                print(f"      ❌ 位置控制失败")
            else:
                print(f"      ✅ 位置控制成功")
                
    def _test_velocity_control(self, joint_idx, joint_name):
        """测试速度控制"""
        print(f"    🏃 测试速度控制...")
        
        joint_info = self.joint_data[joint_idx]
        test_velocity = 0.1  # 慢速测试
        
        initial_state = p.getJointState(self.robot_id, joint_idx)
        initial_pos = initial_state[0]
        
        # 应用速度控制
        p.setJointMotorControl2(
            self.robot_id, joint_idx,
            p.VELOCITY_CONTROL,
            targetVelocity=test_velocity,
            force=joint_info['max_force'] * 0.3
        )
        
        # 运行仿真
        for _ in range(60):  # 0.25秒
            p.stepSimulation()
            
        # 检查结果
        final_state = p.getJointState(self.robot_id, joint_idx)
        final_pos = final_state[0]
        actual_movement = abs(final_pos - initial_pos)
        
        print(f"      预期移动: ~{test_velocity * 0.25:.4f}")
        print(f"      实际移动: {actual_movement:.4f}")
        
        if actual_movement < 0.001:
            print(f"      ❌ 速度控制失败")
        else:
            print(f"      ✅ 速度控制成功")
            
    def _test_torque_control(self, joint_idx, joint_name):
        """测试力矩控制"""
        print(f"    💪 测试力矩控制...")
        
        joint_info = self.joint_data[joint_idx]
        test_torque = joint_info['max_force'] * 0.1  # 10%力矩
        
        initial_state = p.getJointState(self.robot_id, joint_idx)
        initial_pos = initial_state[0]
        
        # 应用力矩控制
        p.setJointMotorControl2(
            self.robot_id, joint_idx,
            p.TORQUE_CONTROL,
            force=test_torque
        )
        
        # 运行仿真
        for _ in range(60):  # 0.25秒
            p.stepSimulation()
            
        # 检查结果
        final_state = p.getJointState(self.robot_id, joint_idx)
        final_pos = final_state[0]
        actual_movement = abs(final_pos - initial_pos)
        
        print(f"      应用力矩: {test_torque:.4f}")
        print(f"      实际移动: {actual_movement:.4f}")
        
        if actual_movement < 0.001:
            print(f"      ❌ 力矩控制失败")
        else:
            print(f"      ✅ 力矩控制成功")
            
    def _test_joint_constraints(self, joint_idx, joint_name):
        """测试关节约束"""
        print(f"    🔒 测试关节约束...")
        
        # 检查关节链
        parent_chain = self._get_parent_chain(joint_idx)
        print(f"      父关节链: {parent_chain}")
        
        # 检查质量
        mass = p.getDynamicsInfo(self.robot_id, joint_idx)[0]
        print(f"      链接质量: {mass:.6f}")
        
        if mass < 0.001:
            print(f"      ⚠️  质量过小可能导致控制问题")
            
    def check_joint_constraints(self):
        """检查关节约束和依赖"""
        print("\n🔗 关节约束分析:")
        print("-" * 50)
        
        # 检查关节树结构
        self._analyze_joint_tree()
        
        # 检查固定关节
        self._check_fixed_joints()
        
    def _analyze_joint_tree(self):
        """分析关节树结构"""
        print("关节树结构:")
        
        for i, data in self.joint_data.items():
            indent = "  " * self._get_joint_depth(i)
            parent_idx = data['parent_index']
            
            if parent_idx == -1:
                parent_name = "BASE"
            else:
                parent_name = self.joint_data.get(parent_idx, {}).get('name', f'Joint{parent_idx}')
                
            print(f"{indent}{data['name']} ← {parent_name}")
            
    def _check_fixed_joints(self):
        """检查固定关节"""
        print("\n固定关节:")
        
        for i, data in self.joint_data.items():
            if data['type'] == 4:  # FIXED
                print(f"  {data['name']}: 固定关节")
                
    def analyze_urdf_structure(self):
        """分析URDF结构问题"""
        print("\n📋 URDF结构分析:")
        print("-" * 50)
        
        # 检查问题模式
        self._check_common_issues()
        
    def _check_common_issues(self):
        """检查常见问题"""
        print("常见问题检查:")
        
        # 1. 检查关节范围
        for i, data in self.joint_data.items():
            if data['type'] == 0:  # REVOLUTE
                range_size = abs(data['limits'][1] - data['limits'][0])
                if range_size < 0.01:
                    print(f"  ⚠️  {data['name']}: 关节范围过小 ({range_size:.6f})")
                    
        # 2. 检查速度限制
        for i, data in self.joint_data.items():
            if data['max_velocity'] > 1000:
                print(f"  ⚠️  {data['name']}: 速度限制异常 ({data['max_velocity']})")
                
        # 3. 检查力矩限制
        for i, data in self.joint_data.items():
            if data['max_force'] < 0.1:
                print(f"  ⚠️  {data['name']}: 最大力过小 ({data['max_force']})")
                
    def _get_joint_type_name(self, joint_type):
        """获取关节类型名称"""
        type_map = {0: "REVOLUTE", 1: "PRISMATIC", 2: "SPHERICAL", 3: "PLANAR", 4: "FIXED"}
        return type_map.get(joint_type, f"UNKNOWN({joint_type})")
        
    def _find_joint_by_name(self, joint_name):
        """通过名称找关节索引"""
        for i, data in self.joint_data.items():
            if data['name'] == joint_name:
                return i
        return None
        
    def _get_parent_chain(self, joint_idx):
        """获取父关节链"""
        chain = []
        current = joint_idx
        
        while current != -1 and len(chain) < 10:  # 防止无限循环
            data = self.joint_data.get(current)
            if data:
                chain.append(data['name'])
                current = data['parent_index']
            else:
                break
                
        return " → ".join(reversed(chain))
        
    def _get_joint_depth(self, joint_idx):
        """获取关节深度"""
        depth = 0
        current = joint_idx
        
        while current != -1 and depth < 10:
            data = self.joint_data.get(current)
            if data:
                current = data['parent_index']
                depth += 1
            else:
                break
                
        return depth
        
    def close(self):
        """关闭诊断工具"""
        p.disconnect()


def main():
    """主函数"""
    print("🔍 Alpha机器人深度关节诊断")
    print("=" * 50)
    
    diagnostic = DeepJointDiagnostic()
    
    urdf_path = "../alpha_description/urdf/alpha_robot_for_pybullet.urdf"
        
    if os.path.exists(urdf_path):
        diagnostic.load_and_diagnose(urdf_path)
        input("\n按Enter键关闭...")
    else:
        print(f"❌ URDF文件未找到: {urdf_path}")
        
    diagnostic.close()


if __name__ == "__main__":
    main()