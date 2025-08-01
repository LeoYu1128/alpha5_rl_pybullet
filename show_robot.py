import time
from envs.alpha_env import AlphaRobotEnv


def diagnose_gripper_issue(env):
    """全面诊断夹爪问题"""
    import pybullet as p
    import numpy as np
    
    print("=" * 60)
    print("开始夹爪问题诊断...")
    print("=" * 60)
    
    # 1. 检查关节总数和信息
    print("\n1. 关节信息检查:")
    num_joints = p.getNumJoints(env.robot_id)
    print(f"机器人总关节数: {num_joints}")
    
    all_joints = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(env.robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        joint_lower = joint_info[8] 
        joint_upper = joint_info[9]
        
        type_names = {
            p.JOINT_REVOLUTE: "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC", 
            p.JOINT_FIXED: "FIXED"
        }
        
        all_joints.append({
            'index': i,
            'name': joint_name,
            'type': type_names.get(joint_type, f"UNKNOWN({joint_type})"),
            'lower': joint_lower,
            'upper': joint_upper
        })
        
        print(f"  {i:2d}: {joint_name:35s} | {type_names.get(joint_type, 'UNKNOWN'):10s} | Range: [{joint_lower:8.4f}, {joint_upper:8.4f}]")
    
    # 2. 检查环境设置的关节索引
    print(f"\n2. 环境关节设置检查:")
    print(f"joint_indices: {env.joint_indices}")
    print(f"joint_names: {env.joint_names}")
    print(f"initial_positions长度: {len(env.initial_joint_positions)}")
    print(f"initial_positions: {env.initial_joint_positions}")
    
    # 3. 查找夹爪相关关节
    print(f"\n3. 夹爪关节识别:")
    gripper_joints = []
    joint_5_index = None
    
    for joint in all_joints:
        if 'joint_5' in joint['name']:
            joint_5_index = joint['index']
            print(f"  找到joint_5: 索引{joint['index']}, 类型{joint['type']}, 范围[{joint['lower']:.4f}, {joint['upper']:.4f}]")
        
        if 'jaws' in joint['name'].lower() or 'gripper' in joint['name'].lower():
            gripper_joints.append(joint)
            print(f"  找到夹爪关节: {joint['name']} (索引{joint['index']}, 类型{joint['type']})")
    
    if joint_5_index is None:
        print("  ❌ 错误：未找到joint_5!")
        return None, []
    
    if not gripper_joints:
        print("  ❌ 警告：未找到夹爪关节!")
    
    # 4. 检查当前关节状态
    print(f"\n4. 当前关节状态:")
    for i, joint_idx in enumerate(env.joint_indices):
        joint_state = p.getJointState(env.robot_id, joint_idx)
        current_pos = joint_state[0]
        current_vel = joint_state[1]
        print(f"  {env.joint_names[i]}: 位置={current_pos:.6f}, 速度={current_vel:.6f}")
    
    # 5. 测试joint_5控制
    print(f"\n5. 测试joint_5控制:")
    test_positions = [0.002, 0.005, 0.008, 0.011]
    
    for test_pos in test_positions:
        print(f"\n  测试位置: {test_pos:.4f}m")
        
        # 方法1: resetJointState (强制设置)
        p.resetJointState(env.robot_id, joint_5_index, test_pos, targetVelocity=0)
        
        # 运行几步仿真
        for _ in range(20):
            p.stepSimulation()
        
        # 检查joint_5实际位置
        joint_5_state = p.getJointState(env.robot_id, joint_5_index)
        actual_pos = joint_5_state[0]
        print(f"    joint_5实际位置: {actual_pos:.6f}m (目标: {test_pos:.6f}m)")
        
        # 检查所有夹爪关节
        for gripper_joint in gripper_joints:
            gripper_state = p.getJointState(env.robot_id, gripper_joint['index'])
            gripper_pos = gripper_state[0]
            gripper_vel = gripper_state[1]
            
            # 计算理论值 (如果是mimic关节)
            if actual_pos != 0:
                apparent_multiplier = gripper_pos / actual_pos
            else:
                apparent_multiplier = 0
                
            print(f"    {gripper_joint['name']}: 位置={gripper_pos:.6f}, 速度={gripper_vel:.6f}, 倍数={apparent_multiplier:.1f}")
        
        # 等待观察
        time.sleep(0.8)  # 给足够时间观察
    
    return joint_5_index, gripper_joints

def manual_gripper_control_test(env, joint_5_index, gripper_joints):
    """手动控制夹爪测试"""
    import pybullet as p
    
    print(f"\n6. 手动控制测试:")
    
    test_openness = [0.0, 0.5, 1.0, 0.3]
    
    for openness in test_openness:
        print(f"\n  设置夹爪开度: {openness:.1f}")
        
        # 计算joint_5位置
        joint_5_pos = 0.0013 + openness * (0.0133 - 0.0013)
        print(f"  对应joint_5位置: {joint_5_pos:.6f}m")
        
        # 控制joint_5
        p.setJointMotorControl2(
            env.robot_id,
            joint_5_index,
            p.POSITION_CONTROL,
            targetPosition=joint_5_pos,
            force=50,
            maxVelocity=0.1
        )
        
        # 如果夹爪关节存在，也手动控制（以防mimic不工作）
        for gripper_joint in gripper_joints:
            if 'jaws' in gripper_joint['name'].lower():
                # 假设51倍关系
                gripper_angle = joint_5_pos * 51
                gripper_angle = min(gripper_angle, 0.5)  # 限制最大角度
                
                p.setJointMotorControl2(
                    env.robot_id,
                    gripper_joint['index'],
                    p.POSITION_CONTROL,
                    targetPosition=gripper_angle,
                    force=20
                )
                print(f"    手动设置{gripper_joint['name']}到: {gripper_angle:.4f}rad")
        
        # 等待运动完成
        for step in range(50):
            p.stepSimulation()
        
        # 检查最终状态
        joint_5_state = p.getJointState(env.robot_id, joint_5_index)
        final_pos = joint_5_state[0]
        print(f"    joint_5最终位置: {final_pos:.6f}m")
        
        for gripper_joint in gripper_joints:
            gripper_state = p.getJointState(env.robot_id, gripper_joint['index'])
            print(f"    {gripper_joint['name']}: {gripper_state[0]:.6f}")
        
        # 暂停观察
        time.sleep(1.5)

def test_gripper_debug(env):
    """主测试函数"""
    # 运行诊断
    joint_5_index, gripper_joints = diagnose_gripper_issue(env)
    
    if joint_5_index is not None:
        # 手动控制测试
        manual_gripper_control_test(env, joint_5_index, gripper_joints)
    
    print("\n" + "="*60)
    print("诊断完成！请查看上面的输出信息。")
    print("="*60)

# 主程序
if __name__ == "__main__":
    try:
        # 创建一个带GUI的环境
        env = AlphaRobotEnv(render_mode="human")
        
        # reset环境，确保机械臂初始化到正确位置
        obs, info = env.reset()
        
        print("环境初始化完成，开始夹爪诊断...")
        time.sleep(1)  # 让环境稳定一下
        
        # 运行诊断
        test_gripper_debug(env)
        
        print("\n诊断完成！现在你可以观察结果...")
        
        # 保持窗口开启一段时间让你观察
        print("窗口将保持开启30秒供观察...")
        for i in range(300):  # 30秒
            env.render()
            time.sleep(0.1)
            
            # 每5秒显示一次倒计时
            if i % 50 == 0:
                remaining = (300 - i) // 10
                print(f"剩余时间: {remaining}秒")
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保关闭环境
        try:
            env.close()
            print("环境已关闭")
        except:
            pass
