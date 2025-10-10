import pybullet as p
import pybullet_data
import time
import numpy as np
import os

# ✅ 修改成你的URDF文件名
URDF_PATH = "alpha_robot_for_pybullet.urdf"

def main():
    # 连接 PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)

    # 设置路径、重力、时间步和子步数
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1./240., numSubSteps=10)  # ✅ 关键：子步数10
    # p.setTimeStep(1./240.)  # 不需要重复

    # 加载地面
    plane_id = p.loadURDF("plane.urdf")

    # 加载机械臂
    robot_id = p.loadURDF(URDF_PATH, [0, 0, 0.2], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)

    # 摄像机角度
    p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=30, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.3])

    num_joints = p.getNumJoints(robot_id)
    print(f"✅ Loaded robot with {num_joints} joints.")

    # 获取每个关节的限制范围，只保留前4个有效关节，去掉末端执行器
    joint_limits = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]

        # 跳过固定关节和末端执行器，只取前4个关节
        if lower_limit < upper_limit and "ee" not in joint_name.lower() and "gripper" not in joint_name.lower():
            joint_limits.append((i, lower_limit, upper_limit))
            print(f"  🔧 Joint {i} ({joint_name}): range = [{lower_limit:.2f}, {upper_limit:.2f}]")
        if len(joint_limits) >= 4:  # 只测试前4个关节
            break

    print("\n🤖 Random motion simulation started. Press Ctrl+C to stop.\n")

    try:
        while True:
            # 随机生成关节目标角度（在限制范围内）
            for joint_id, lower, upper in joint_limits:
                target_angle = np.random.uniform(lower, upper)
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_id,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=9
                )

            # 运行一小段时间
            for _ in range(200):
                p.stepSimulation()  # ✅ 子步已在setPhysicsEngineParameter里生效

                # 检查机械臂和地面的碰撞
                contacts = p.getContactPoints(bodyA=robot_id, bodyB=plane_id)
                if contacts:
                    print("⚠️ Detected contact points:")
                    for c in contacts:
                        print(f"  - Link {c[3]} collided at position {np.round(c[6], 4)} with normal {np.round(c[7], 4)}")
                time.sleep(1./240.)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
    finally:
        if p.isConnected():
            p.disconnect()
        print("✅ Clean exit from PyBullet.")

if __name__ == "__main__":
    if not os.path.exists(URDF_PATH):
        print(f"❌ 找不到URDF文件: {URDF_PATH}")
    else:
        main()
