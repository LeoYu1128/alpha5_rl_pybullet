"""
深度信号传递诊断工具 - 检查从action到关节运动的完整链路
"""

import sys
sys.path.append('.')
sys.path.append('./envs')

import numpy as np
import pybullet as p
from envs.rl_env_v13 import AlphaReachEnv
import time

class SignalFlowDiagnostic:
    """信号流诊断器"""
    
    def __init__(self):
        self.env = None
        self.test_results = {}
        
    def run_full_diagnostic(self):
        """运行完整诊断"""
        print("\n" + "="*80)
        print("🔬 深度信号传递诊断")
        print("="*80)
        
        # 1. 创建环境
        print("\n1️⃣ 创建环境...")
        self.env = AlphaReachEnv(render_mode="human")
        obs = self.env.reset()
        print(f"   ✅ 环境创建成功")
        print(f"   观察空间: {obs.shape}")
        print(f"   动作空间: {self.env.action_space.shape}")
        
        # 稳定物理
        print("\n   等待物理稳定...")
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.env.physics_client)
        
        # 2. 检查关节索引映射
        print("\n2️⃣ 检查关节索引映射...")
        self.check_joint_mapping()
        
        # 3. 检查初始状态读取
        print("\n3️⃣ 检查初始状态读取...")
        self.check_state_reading()
        
        # 4. 测试单个action的完整流程
        print("\n4️⃣ 测试action信号流...")
        self.test_action_signal_flow()
        
        # 5. 测试step函数
        print("\n5️⃣ 测试step函数完整性...")
        self.test_step_function()
        
        # 6. 测试连续控制
        print("\n6️⃣ 测试连续控制...")
        self.test_continuous_control()
        
        # 7. 对比手动控制
        print("\n7️⃣ 对比手动控制...")
        self.test_manual_control()
        
        # 8. 总结
        print("\n" + "="*80)
        print("📊 诊断总结")
        print("="*80)
        self.print_summary()
        
    def check_joint_mapping(self):
        """检查关节索引映射"""
        print(f"\n   主关节索引: {self.env.main_joint_indices}")
        
        for i, joint_idx in enumerate(self.env.main_joint_indices):
            joint_info = p.getJointInfo(self.env.robot_id, joint_idx, 
                                       physicsClientId=self.env.physics_client)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            print(f"   关节 {i}: 索引={joint_idx}, 名称={joint_name}, 类型={joint_type}")
            
            # 检查是否是旋转关节
            if joint_type != 0:  # 0 = REVOLUTE
                print(f"      ⚠️ 警告: 关节类型不是REVOLUTE!")
                self.test_results['joint_mapping'] = False
                return
        
        self.test_results['joint_mapping'] = True
        print(f"   ✅ 关节映射正确")
    
    def check_state_reading(self):
        """检查状态读取"""
        print(f"\n   读取当前关节状态...")
        
        # 方法1: 环境的方法
        joint_pos_env = self.env._get_joint_positions()
        print(f"   环境方法: {np.degrees(joint_pos_env)}")
        
        # 方法2: 直接PyBullet
        joint_pos_direct = []
        for joint_idx in self.env.main_joint_indices:
            state = p.getJointState(self.env.robot_id, joint_idx,
                                   physicsClientId=self.env.physics_client)
            joint_pos_direct.append(state[0])
        joint_pos_direct = np.array(joint_pos_direct)
        print(f"   直接读取: {np.degrees(joint_pos_direct)}")
        
        # 检查是否一致
        if np.allclose(joint_pos_env, joint_pos_direct, atol=1e-6):
            print(f"   ✅ 状态读取一致")
            self.test_results['state_reading'] = True
        else:
            print(f"   ❌ 状态读取不一致!")
            self.test_results['state_reading'] = False
    
    def test_action_signal_flow(self):
        """测试action的完整信号流"""
        print(f"\n   测试: action [1.0, 0, 0, 0] (只动第一个关节)")
        
        # 记录初始状态
        initial_pos = self.env._get_joint_positions()
        print(f"   初始位置: {np.degrees(initial_pos)}")
        
        # 构造action
        action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Step 1: action缩放
        scaled_action = action * 0.15
        print(f"\n   Step 1 - 缩放action:")
        print(f"      原始action: {action}")
        print(f"      缩放后: {scaled_action}")
        print(f"      期望变化: {np.degrees(scaled_action)} 度")
        
        # Step 2: 计算目标位置
        target_positions = initial_pos + scaled_action
        print(f"\n   Step 2 - 计算目标位置:")
        print(f"      目标位置: {np.degrees(target_positions)}")
        
        # Step 3: 限位裁剪
        clipped_targets = []
        for i, joint_idx in enumerate(self.env.main_joint_indices):
            joint = self.env.joint_info[joint_idx]
            clipped = np.clip(target_positions[i], joint['lower'], joint['upper'])
            clipped_targets.append(clipped)
            if clipped != target_positions[i]:
                print(f"      关节 {i}: 被裁剪 {target_positions[i]} -> {clipped}")
        clipped_targets = np.array(clipped_targets)
        
        if np.allclose(clipped_targets, target_positions):
            print(f"      ✅ 无裁剪")
        
        # Step 4: 发送电机指令
        print(f"\n   Step 3 - 发送电机指令:")
        control_torque = 10  # 当前代码中的值
        
        for i, joint_idx in enumerate(self.env.main_joint_indices):
            print(f"      关节 {i}: setJointMotorControl2(")
            print(f"         targetPosition={clipped_targets[i]:.4f} ({np.degrees(clipped_targets[i]):.1f}°),")
            print(f"         force={control_torque},")
            print(f"         maxVelocity=1.0)")
            
            p.setJointMotorControl2(
                self.env.robot_id, joint_idx, p.POSITION_CONTROL,
                targetPosition=clipped_targets[i],
                maxVelocity=1.0,
                force=control_torque,
                physicsClientId=self.env.physics_client
            )
        
        # Step 5: 仿真并观察
        print(f"\n   Step 4 - 仿真4步...")
        for step in range(4):
            self.env._apply_underwater_forces()
            p.stepSimulation(physicsClientId=self.env.physics_client)
            
            # 每步后读取位置
            current_pos = self.env._get_joint_positions()
            change = current_pos - initial_pos
            print(f"      Step {step+1}: 变化 = {np.degrees(change)}")
        
        # 最终结果
        final_pos = self.env._get_joint_positions()
        total_change = final_pos - initial_pos
        
        print(f"\n   📊 结果:")
        print(f"      期望变化: {np.degrees(scaled_action)}")
        print(f"      实际变化: {np.degrees(total_change)}")
        print(f"      达成率: {np.abs(total_change / scaled_action) * 100}%")
        
        # 检查是否有显著移动
        if np.linalg.norm(total_change) > 0.01:  # 约0.57度
            print(f"   ✅ 有显著移动")
            self.test_results['action_flow'] = True
        else:
            print(f"   ❌ 几乎没有移动!")
            self.test_results['action_flow'] = False
        
        # 重置到初始状态
        for i, joint_idx in enumerate(self.env.main_joint_indices):
            p.resetJointState(self.env.robot_id, joint_idx, initial_pos[i],
                            physicsClientId=self.env.physics_client)
    
    def test_step_function(self):
        """测试step函数"""
        print(f"\n   调用 env.step([1.0, 0, 0, 0])...")
        
        initial_pos = self.env._get_joint_positions()
        action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        obs, reward, done, info = self.env.step(action)
        
        final_pos = self.env._get_joint_positions()
        change = final_pos - initial_pos
        
        print(f"   返回值:")
        print(f"      observation shape: {obs.shape}")
        print(f"      reward: {reward:.4f}")
        print(f"      done: {done}")
        print(f"      info: {info.keys()}")
        
        print(f"\n   关节变化: {np.degrees(change)}")
        
        if np.linalg.norm(change) > 0.01:
            print(f"   ✅ step函数有效")
            self.test_results['step_function'] = True
        else:
            print(f"   ❌ step函数无效!")
            self.test_results['step_function'] = False
    
    def test_continuous_control(self):
        """测试连续控制"""
        print(f"\n   测试: 连续10步相同action")
        
        initial_pos = self.env._get_joint_positions()
        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        
        positions_history = [initial_pos.copy()]
        
        for i in range(10):
            obs, reward, done, info = self.env.step(action)
            current_pos = self.env._get_joint_positions()
            positions_history.append(current_pos.copy())
            
            if i < 3 or i == 9:
                change = current_pos - initial_pos
                print(f"      Step {i+1}: 累计变化 = {np.degrees(change[0]):.2f}° (关节0)")
        
        total_change = positions_history[-1] - positions_history[0]
        
        # 检查是否持续移动
        monotonic = True
        for i in range(1, len(positions_history)):
            if positions_history[i][0] < positions_history[i-1][0]:
                monotonic = False
                break
        
        print(f"\n   总变化: {np.degrees(total_change[0]):.2f}°")
        
        if monotonic and np.abs(total_change[0]) > 0.1:  # 约5.7度
            print(f"   ✅ 连续控制正常（单调递增）")
            self.test_results['continuous_control'] = True
        else:
            print(f"   ❌ 连续控制异常!")
            if not monotonic:
                print(f"      原因: 位置不是单调的")
            else:
                print(f"      原因: 移动幅度太小")
            self.test_results['continuous_control'] = False
        
        # 重置
        self.env.reset()
    
    def test_manual_control(self):
        """对比手动控制"""
        print(f"\n   对比测试: 使用force=9.0 vs force=500.0")
        
        initial_pos = self.env._get_joint_positions()
        target_change = np.radians(10)  # 目标移动10度
        
        # 测试1: force=9.0
        print(f"\n   测试 force=9.0:")
        test_pos_1 = initial_pos[0] + target_change
        
        # 限制到关节限位
        joint_info = self.env.joint_info[self.env.main_joint_indices[0]]
        test_pos_1 = np.clip(test_pos_1, joint_info['lower'], joint_info['upper'])
        
        p.resetJointState(self.env.robot_id, self.env.main_joint_indices[0], 
                         initial_pos[0], physicsClientId=self.env.physics_client)
        
        p.setJointMotorControl2(
            self.env.robot_id, self.env.main_joint_indices[0], 
            p.POSITION_CONTROL,
            targetPosition=test_pos_1,
            maxVelocity=1.0,
            force=9.0,
            physicsClientId=self.env.physics_client
        )
        
        for _ in range(240):  # 1秒
            self.env._apply_underwater_forces()
            p.stepSimulation(physicsClientId=self.env.physics_client)
        
        state_1 = p.getJointState(self.env.robot_id, self.env.main_joint_indices[0],
                                 physicsClientId=self.env.physics_client)
        change_1 = state_1[0] - initial_pos[0]
        print(f"      实际变化: {np.degrees(change_1):.2f}°")
        
        # 测试2: force=500.0
        print(f"\n   测试 force=500.0:")
        p.resetJointState(self.env.robot_id, self.env.main_joint_indices[0], 
                         initial_pos[0], physicsClientId=self.env.physics_client)
        
        p.setJointMotorControl2(
            self.env.robot_id, self.env.main_joint_indices[0], 
            p.POSITION_CONTROL,
            targetPosition=test_pos_1,
            maxVelocity=1.0,
            force=500.0,
            physicsClientId=self.env.physics_client
        )
        
        for _ in range(240):  # 1秒
            self.env._apply_underwater_forces()
            p.stepSimulation(physicsClientId=self.env.physics_client)
        
        state_2 = p.getJointState(self.env.robot_id, self.env.main_joint_indices[0],
                                 physicsClientId=self.env.physics_client)
        change_2 = state_2[0] - initial_pos[0]
        print(f"      实际变化: {np.degrees(change_2):.2f}°")
        
        print(f"\n   对比:")
        print(f"      force=9.0:   {np.degrees(change_1):.2f}° ({abs(change_1/target_change)*100:.1f}%达成)")
        print(f"      force=500.0: {np.degrees(change_2):.2f}° ({abs(change_2/target_change)*100:.1f}%达成)")
        
        if abs(change_1) > abs(change_2) * 2:
            print(f"   ⚠️ force=9.0的效果是force=500.0的{abs(change_1/change_2):.1f}倍!")
            self.test_results['force_comparison'] = 'small_better'
        else:
            self.test_results['force_comparison'] = 'similar'
    
    def print_summary(self):
        """打印总结"""
        all_pass = all(self.test_results.values())
        
        print(f"\n测试结果:")
        for test_name, result in self.test_results.items():
            status = "✅" if result else "❌"
            print(f"   {status} {test_name}: {result}")
        
        if not all_pass:
            print(f"\n🔴 发现问题:")
            
            if not self.test_results.get('joint_mapping', True):
                print(f"   • 关节索引映射错误")
            
            if not self.test_results.get('state_reading', True):
                print(f"   • 状态读取不一致")
            
            if not self.test_results.get('action_flow', True):
                print(f"   • Action信号流中断，关节几乎不移动")
            
            if not self.test_results.get('step_function', True):
                print(f"   • step函数返回值有问题")
            
            if not self.test_results.get('continuous_control', True):
                print(f"   • 连续控制失效")
            
            if self.test_results.get('force_comparison') == 'small_better':
                print(f"   • force参数太大，需要从500降到10左右")
        else:
            print(f"\n✅ 所有基础信号流正常")
            
            if self.test_results.get('force_comparison') == 'small_better':
                print(f"   但建议: 降低force参数可以获得更好的控制")
        
        print(f"\n按Enter关闭...")
        input()
        self.env.close()


if __name__ == "__main__":
    diagnostic = SignalFlowDiagnostic()
    diagnostic.run_full_diagnostic()