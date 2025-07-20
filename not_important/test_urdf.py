# #!/usr/bin/env python3
# """
# 测试URDF文件加载
# 在开始强化学习训练之前，先确保URDF文件能够正确加载
# """

# import pybullet as p
# import pybullet_data
# import os
# import time

# def find_urdf_file():
#     """查找URDF文件"""
#     current_dir = os.getcwd()
#     print(f"当前目录: {current_dir}")
    
#     # 可能的URDF文件路径
#     possible_paths = [
#         # "./alpha_description/urdf/alpha_robot.urdf",
#         # "../alpha_description/urdf/alpha_robot.urdf",
#         # "alpha_description/urdf/alpha_robot.urdf",
#         # "./alpha_robot.urdf",
#         # "alpha_robot.urdf"

#         "./alpha_description/urdf/alpha_robot_test.urdf",
#         "../alpha_description/urdf/alpha_robot_test.urdf",
#         "alpha_description/urdf/alpha_robot_test.urdf",
#         "./alpha_robot_test.urdf",
#         "alpha_robot_test.urdf"

#         # "./alpha_description/urdf/alpha_robot_origin.urdf",
#         # "../alpha_description/urdf/alpha_robot_origin.urdf",
#         # "alpha_description/urdf/alpha_robot_origin.urdf",
#         # "./alpha_robot_origin.urdf",
#         # "alpha_robot_origin.urdf"
#     ]
    
#     print("\n🔍 搜索URDF文件...")
#     for path in possible_paths:
#         abs_path = os.path.abspath(path)
#         print(f"检查: {abs_path}")
#         if os.path.exists(abs_path):
#             print(f"✅ 找到URDF文件: {abs_path}")
#             return abs_path
    
#     print("❌ 没有找到URDF文件")
#     return None

# def find_meshes_dir():
#     """查找meshes目录"""
#     possible_mesh_paths = [
#         "./alpha_description/meshes",
#         "../alpha_description/meshes", 
#         "alpha_description/meshes",
#         "./meshes",
#         "meshes"
#     ]
    
#     print("\n🔍 搜索meshes目录...")
#     for path in possible_mesh_paths:
#         abs_path = os.path.abspath(path)
#         print(f"检查: {abs_path}")
#         if os.path.exists(abs_path):
#             print(f"✅ 找到meshes目录: {abs_path}")
#             return abs_path
    
#     print("⚠️  没有找到meshes目录")
#     return None

# def test_urdf_loading():
#     """测试URDF文件加载"""
#     print("=" * 60)
#     print("🤖 Alpha Robot URDF 加载测试")
#     print("=" * 60)
    
#     # 查找文件
#     urdf_path = find_urdf_file()
#     if not urdf_path:
#         print("\n❌ 无法找到URDF文件，请确保文件结构正确")
#         return False
    
#     meshes_path = find_meshes_dir()
    
#     try:
#         # 启动PyBullet
#         print("\n🚀 启动PyBullet...")
#         physics_client = p.connect(p.GUI)
#         p.setGravity(0, 0, -9.8)
        
#         # 设置搜索路径
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         if meshes_path:
#             p.setAdditionalSearchPath(meshes_path)
#         p.setAdditionalSearchPath(os.path.dirname(urdf_path))
        
#         print(f"📄 URDF路径: {urdf_path}")
#         if meshes_path:
#             print(f"🎨 Meshes路径: {meshes_path}")
        
#         # 创建地面
#         print("\n🌍 创建地面...")
#         plane_id = p.createMultiBody(
#             baseMass=0,
#             baseCollisionShapeIndex=p.createCollisionShape(
#                 p.GEOM_PLANE, planeNormal=[0, 0, 1]
#             ),
#             basePosition=[0, 0, 0]
#         )
        
#         # 加载机器人
#         print("🤖 加载机器人...")
#         robot_id = p.loadURDF(
#             urdf_path,
#             basePosition=[0, 0, 0.1],
#             useFixedBase=True,
#             flags=p.URDF_USE_INERTIA_FROM_FILE
#         )
#         initial_positions = [
#             2.5,    # joint 1 (alpha_axis_e)
#             2.6,    # joint 2 (alpha_axis_d) 
#             1.0,    # joint 3 (alpha_axis_c)
#             0.0,    # joint 4 (alpha_axis_b)
#             0.005   # joint 5 (alpha_axis_a)
#         ]

#         # 应用初始位置
#         for i, pos in enumerate(initial_positions):
#             p.resetJointState(robot_id, i, pos)
#         print(f"✅ 机器人加载成功! ID: {robot_id}")
        
#         # 获取关节信息
#         num_joints = p.getNumJoints(robot_id)
#         print(f"📊 关节数量: {num_joints}")
        
#         movable_joints = []
#         print("\n📋 关节信息:")
#         for i in range(num_joints):
#             joint_info = p.getJointInfo(robot_id, i)
#             joint_name = joint_info[1].decode('utf-8')
#             joint_type = joint_info[2]
            
#             type_names = {
#                 p.JOINT_REVOLUTE: "REVOLUTE",
#                 p.JOINT_PRISMATIC: "PRISMATIC", 
#                 p.JOINT_FIXED: "FIXED",
#                 p.JOINT_PLANAR: "PLANAR",
#                 p.JOINT_SPHERICAL: "SPHERICAL"
#             }
            
#             type_name = type_names.get(joint_type, f"UNKNOWN({joint_type})")
#             print(f"  关节 {i}: {joint_name} ({type_name})")
            
#             if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
#                 movable_joints.append(i)
#                 lower_limit = joint_info[8]
#                 upper_limit = joint_info[9]
#                 print(f"    可移动，范围: [{lower_limit:.3f}, {upper_limit:.3f}]")
        
#         print(f"\n🎮 可移动关节: {len(movable_joints)} 个")
        
#         if len(movable_joints) == 0:
#             print("⚠️  警告: 没有找到可移动关节，可能影响训练")
        
#         # 设置相机
#         p.resetDebugVisualizerCamera(
#             cameraDistance=1.5,
#             cameraYaw=45,
#             cameraPitch=-30,
#             cameraTargetPosition=[0, 0, 0.3]
#         )
        
#         # 创建目标球体用于测试
#         target_visual = p.createVisualShape(
#             p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.8]
#         )
#         target_id = p.createMultiBody(
#             baseMass=0,
#             baseVisualShapeIndex=target_visual,
#             basePosition=[0.3, 0.3, 0.3]
#         )
        
#         print("\n🎯 红色目标球已创建")
#         print("✅ URDF加载测试成功!")
#         print("🎮 你可以在GUI中查看机器人")
#         print("⏹️  按Ctrl+C停止测试")
        
#         # 简单的动作测试
#         print("\n🔄 开始简单动作测试...")
#         step = 0
#         while True:
#             # # 简单的正弦波动作
#             # if movable_joints:
#             #     import math
#             #     for i, joint_id in enumerate(movable_joints):
#             #         joint_info = p.getJointInfo(robot_id, joint_id)
#             #         lower_limit = joint_info[8]
#             #         upper_limit = joint_info[9]
                    
#             #         # 生成正弦波位置
#             #         mid_pos = (lower_limit + upper_limit) / 2
#             #         amplitude = (upper_limit - lower_limit) / 4
#             #         target_pos = mid_pos + amplitude * math.sin(step * 0.01 + i)
                    
#             #         p.setJointMotorControl2(
#             #             robot_id,
#             #             joint_id,
#             #             p.POSITION_CONTROL,
#             #             targetPosition=target_pos,
#             #             force=50
#             #         )
            
#             p.stepSimulation()
#             time.sleep(1./240.)
#             step += 1
            
#             if step % 240 == 0:
#                 print(f"运行步数: {step}")
        
#     except KeyboardInterrupt:
#         print("\n⏹️  测试被用户中断")
#     except Exception as e:
#         print(f"\n❌ 测试失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
#     finally:
#         p.disconnect()
#         print("🔌 PyBullet已断开")
    
#     return True

# if __name__ == "__main__":
#     success = test_urdf_loading()
#     if success:
#         print("\n🎉 测试成功! 现在可以开始强化学习训练了")
#         print("运行: python simple_train_script.py")
#     else:
#         print("\n❌ 测试失败，请检查URDF文件和路径设置")

#!/usr/bin/env python3
"""
简单的Alpha机械臂URDF测试
测试机械臂能否正常显示，无碰撞，无奇异性
"""

import pybullet as p
import pybullet_data
import time
import math
import os

def test_alpha_robot():
    """测试Alpha机械臂"""
    
    # 启动PyBullet GUI
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 创建地面
    p.loadURDF("plane.urdf")
    
    # 查找URDF文件
    urdf_paths = [
        "./alpha_description/urdf/alpha_robot_test.urdf",
        "alpha_robot_test.urdf",
        "./alpha_robot_test.urdf"
    ]
    
    urdf_path = None
    for path in urdf_paths:
        if os.path.exists(path):
            urdf_path = path
            break
    
    if not urdf_path:
        print("❌ 找不到URDF文件")
        return
    
    # 加载机器人
    robot_id = p.loadURDF(
        urdf_path,
        basePosition=[0, 0, 0.1],
        useFixedBase=True
    )
    
    # ⭐ 关键：设置合理的初始关节角度（避免扭麻花）
    # 基于alpha_5_macro.urdf.xacro中的建议值
    safe_initial_positions = [
        1.5,    # joint 1 - 适中位置
        1.0,    # joint 2 - 抬起手臂
        1.5,    # joint 3 - 避免奇异性
        0.0,    # joint 4 - 零位
        0.005   # joint 5 - 夹爪稍微打开
    ]
    
    # 获取可移动关节
    num_joints = p.getNumJoints(robot_id)
    movable_joints = []
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            movable_joints.append(i)
    
    print(f"🤖 机械臂加载成功！可移动关节数: {len(movable_joints)}")
    
    # 应用安全的初始位置
    for i, joint_id in enumerate(movable_joints[:len(safe_initial_positions)]):
        p.resetJointState(robot_id, joint_id, safe_initial_positions[i])
        print(f"关节 {joint_id}: {safe_initial_positions[i]:.3f}")
    
    # 设置相机视角
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45,
        cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.3]
    )
    
    # 创建一个目标点
    target_visual = p.createVisualShape(
        p.GEOM_SPHERE, 
        radius=0.03, 
        rgbaColor=[1, 0, 0, 0.7]
    )
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=[0.3, 0.2, 0.4]
    )
    
    print("✅ 机械臂已就位，开始简单动作测试...")
    print("🎯 红球是目标位置")
    print("⏹️  按Ctrl+C退出")
    
    # 简单的循环动作测试
    step = 0
    try:
        while True:
            # 非常温和的摆动，避免碰撞
            if len(movable_joints) >= 2:
                # 只动前两个关节，幅度很小
                offset1 = 0.3 * math.sin(step * 0.005)  # 很慢的摆动
                offset2 = 0.2 * math.cos(step * 0.003)  # 更慢的摆动
                
                # 在安全范围内小幅摆动
                target1 = safe_initial_positions[0] + offset1
                target2 = safe_initial_positions[1] + offset2
                
                # 限制在安全范围内
                target1 = max(0.5, min(2.5, target1))
                target2 = max(0.5, min(1.8, target2))
                
                p.setJointMotorControl2(
                    robot_id, movable_joints[0],
                    p.POSITION_CONTROL,
                    targetPosition=target1,
                    force=30
                )
                
                p.setJointMotorControl2(
                    robot_id, movable_joints[1], 
                    p.POSITION_CONTROL,
                    targetPosition=target2,
                    force=30
                )
            
            p.stepSimulation()
            time.sleep(1./60.)  # 60 FPS
            step += 1
            
            # 每5秒输出一次状态
            if step % 300 == 0:
                print(f"运行正常 - 步数: {step}")
    
    except KeyboardInterrupt:
        print("\n✅ 测试完成！")
    
    finally:
        p.disconnect()

if __name__ == "__main__":
    print("🚀 启动Alpha机械臂简单测试...")
    test_alpha_robot()
    print("🎉 测试结束！")