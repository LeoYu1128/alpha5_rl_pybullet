import pybullet as p
import pybullet_data
import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

class AlphaRobotEnv(gym.Env):
    """Alpha Robot 强化学习环境"""
    
    def __init__(self, render_mode=None, max_steps=1000):
        super(AlphaRobotEnv, self).__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # 初始化目标位置（修复：在初始化时就设置）
        self.target_position = np.array([0.2, 0.1, 0.2])
        
        # 设置路径 - 修复路径解析
        current_dir = os.getcwd()  # 获取当前工作目录
        self.urdf_path = os.path.join(current_dir, "alpha_description/urdf/alpha_robot.urdf")
        self.mesh_path = os.path.join(current_dir, "alpha_description/meshes")
        
        # 打印路径用于调试
        print(f"Current directory: {current_dir}")
        print(f"URDF path: {self.urdf_path}")
        print(f"Mesh path: {self.mesh_path}")
        
        # 连接PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(self.mesh_path)
        p.setAdditionalSearchPath(os.path.dirname(self.urdf_path))
        
        # 初始化机器人
        self._setup_robot()
        
        # 定义动作和观察空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.movable_joints),), 
            dtype=np.float32
        )
        
        # 观察空间：关节位置、速度、目标位置
        obs_dim = len(self.movable_joints) * 2 + 3 + 3  # 关节位置 + 关节速度 + 末端位置 + 目标位置
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        print(f"✅ 环境初始化完成")
        print(f"   观察空间维度: {obs_dim}")
        print(f"   动作空间维度: {len(self.movable_joints)}")
        
    def _setup_robot(self):
        """设置机器人和环境"""
        # 创建地面
        self.plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_PLANE, planeNormal=[0, 0, 1]
            ),
            basePosition=[0, 0, 0]
        )
        
        # 加载机器人
        try:
            self.robot_id = p.loadURDF(
                self.urdf_path,
                basePosition=[0, 0, 0.1],
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
            )
            print(f"✅ 成功加载机器人 URDF: {self.urdf_path}")
        except Exception as e:
            print(f"❌ 加载机器人失败: {e}")
            # 如果加载失败，使用一个简单的方块作为替代
            print("使用简单方块作为替代机器人...")
            self.robot_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]
                ),
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0, 1, 0, 1]
                ),
                basePosition=[0, 0, 0.1]
            )
        
        # 获取可移动关节
        num_joints = p.getNumJoints(self.robot_id)
        self.movable_joints = []
        self.joint_limits = []
        
        print(f"机器人关节数: {num_joints}")
        
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode('utf-8')
            
            print(f"关节 {i}: {joint_name}, 类型: {joint_type}")
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.movable_joints.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                self.joint_limits.append((lower_limit, upper_limit))
                print(f"  可移动关节: {joint_name}, 范围: [{lower_limit:.3f}, {upper_limit:.3f}]")
        
        # # 如果没有可移动关节，创建一些虚拟关节
        # if len(self.movable_joints) == 0:
        #     print("警告: 没有找到可移动关节，使用虚拟关节")
        #     self.movable_joints = [0, 1, 2]  # 虚拟关节ID
        #     self.joint_limits = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
        
        print(f"可移动关节数: {len(self.movable_joints)}")
        
        # 创建目标可视化
        self.target_visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8]
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.target_visual,
            basePosition=self.target_position
        )
        
    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # 重置机器人关节到随机位置
        for i, joint_id in enumerate(self.movable_joints):
            if i < len(self.joint_limits):
                lower, upper = self.joint_limits[i]
                random_pos = np.random.uniform(lower, upper)
                try:
                    p.resetJointState(self.robot_id, joint_id, random_pos)
                except:
                    # 如果是虚拟关节，跳过
                    pass
        
        # 设置新的随机目标位置
        self.target_position = self.sample_valid_target()
        
        # 更新目标可视化
        try:
            p.resetBasePositionAndOrientation(
                self.target_id, 
                self.target_position, 
                [0, 0, 0, 1]
            )
        except:
            pass
        
        return self._get_observation(), {}
    
    def sample_valid_target(self,max_reach=0.4, min_radius=0.1, z_range=(0.1, 0.3)):
        while True:
            x = np.random.uniform(-max_reach, max_reach)
            y = np.random.uniform(-max_reach, max_reach)
            z = np.random.uniform(*z_range)
            r = np.linalg.norm([x, y, z])
            if min_radius < r <= max_reach:
                return np.array([x, y, z])
            
    def step(self, action):
        """执行动作"""
        self.current_step += 1
        
        # 应用动作到关节
        for i, joint_id in enumerate(self.movable_joints):
            if i < len(self.joint_limits) and i < len(action):
                lower, upper = self.joint_limits[i]
                # 将动作从[-1,1]映射到关节限制范围
                target_pos = lower + (action[i] + 1) * (upper - lower) / 2
                
                try:
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint_id,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=100
                    )
                except:
                    # 如果是虚拟关节，跳过
                    pass
        
        # 运行仿真
        p.stepSimulation()
        
        # 获取观察、奖励、终止条件
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        return observation, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """获取观察状态"""
        joint_positions = []
        joint_velocities = []
        
        for i, joint_id in enumerate(self.movable_joints):
            if i < len(self.joint_limits):
                try:
                    joint_state = p.getJointState(self.robot_id, joint_id)
                    joint_positions.append(joint_state[0])
                    joint_velocities.append(joint_state[1])
                    # joint_state[0] → 关节位置  
                    # joint_state[1] → 关节速度  
                    # joint_state[2] → 反作用力  
                    # joint_state[3] → 实际施加的力矩
                except:
                    # 如果是虚拟关节，使用随机值
                    joint_positions.append(0.0)
                    joint_velocities.append(0.0)
        
        # 获取末端执行器位置（假设最后一个link是末端执行器）
        num_links = p.getNumJoints(self.robot_id)
        if num_links > 0:
            try:
                end_effector_state = p.getLinkState(self.robot_id, num_links - 1)
                end_effector_pos = end_effector_state[0]
            except:
                end_effector_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        else:
            end_effector_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        
        # 确保目标位置已初始化
        if not hasattr(self, 'target_position') or self.target_position is None:
            self.target_position = np.array([0.2, 0.1, 0.2])
        
        # 组合观察 - 确保维度一致
        obs = np.concatenate([
            joint_positions,        # 关节位置
            joint_velocities,       # 关节速度
            end_effector_pos,       # 末端执行器位置 (3维)
            self.target_position    # 目标位置 (3维)
        ]).astype(np.float32)
        
        return obs
    def _calculate_reward(self):
        """计算密集奖励 (Dense Reward)"""
        # 获取末端执行器位置
        num_links = p.getNumJoints(self.robot_id)
        if num_links > 0:
            try:
                end_effector_state = p.getLinkState(self.robot_id, num_links - 1)
                end_effector_pos = np.array(end_effector_state[0])
            except:
                end_effector_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        else:
            end_effector_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        
        # 确保目标位置已初始化
        if not hasattr(self, 'target_position') or self.target_position is None:
            self.target_position = np.array([0.2, 0.1, 0.2])
        
        # 计算到目标的距离
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        
        # 1. 距离奖励 - 使用指数衰减函数，提供连续的密集反馈
        max_distance = 0.4  # 机械臂的最大可达范围
        distance_reward = 20.0 * np.exp(-3.0 * distance / max_distance)
        
        # 2. 距离改进奖励 - 奖励向目标移动的行为
        if hasattr(self, 'prev_distance'):
            distance_improvement = self.prev_distance - distance
            improvement_reward = 100.0 * distance_improvement  # 在小范围内放大改进奖励
        else:
            improvement_reward = 0.0
        self.prev_distance = distance
        
        # 3. 目标到达奖励 - 高斯函数形式，在目标附近提供强烈奖励
        sigma = 0.02  # 控制奖励的集中程度，适应小范围
        target_reward = 200.0 * np.exp(-0.5 * (distance / sigma) ** 2)
        
        # 4. 动作平滑奖励 - 惩罚剧烈运动
        joint_velocities = []
        for i, joint_id in enumerate(self.movable_joints):
            if i < len(self.joint_limits):
                try:
                    joint_state = p.getJointState(self.robot_id, joint_id)
                    joint_velocities.append(joint_state[1])
                except:
                    joint_velocities.append(0.0)
        
        if joint_velocities:
            # 使用L2正则化的平滑惩罚
            velocity_penalty = -0.5 * np.sum(np.square(joint_velocities))
        else:
            velocity_penalty = 0.0
        
        # 5. 方向奖励 - 奖励朝向目标的运动方向
        if hasattr(self, 'prev_end_effector_pos'):
            movement_vec = end_effector_pos - self.prev_end_effector_pos
            target_direction = self.target_position - end_effector_pos
            
            if np.linalg.norm(movement_vec) > 1e-6 and np.linalg.norm(target_direction) > 1e-6:
                # 计算运动方向与目标方向的余弦相似度
                cos_similarity = np.dot(movement_vec, target_direction) / (
                    np.linalg.norm(movement_vec) * np.linalg.norm(target_direction))
                direction_reward = 10.0 * cos_similarity  # 增强方向奖励
            else:
                direction_reward = 0.0
        else:
            direction_reward = 0.0
        self.prev_end_effector_pos = end_effector_pos.copy()
        
        # 6. 稳定性奖励 - 奖励在目标附近保持稳定
        if distance < 0.03:  # 调整为更小的稳定区域
            if joint_velocities:
                stability_reward = 5.0 * np.exp(-np.sum(np.square(joint_velocities)))
            else:
                stability_reward = 5.0
        else:
            stability_reward = 0.0
        
        # 7. 精度奖励 - 在非常接近目标时给予额外奖励
        if distance < 0.01:
            precision_reward = 50.0
        elif distance < 0.02:
            precision_reward = 20.0
        else:
            precision_reward = 0.0
        
        # 8. 时间效率奖励 - 轻微惩罚时间步长，鼓励快速到达
        time_penalty = -0.02
        
        # 组合所有奖励分量
        total_reward = (distance_reward + 
                    improvement_reward + 
                    target_reward + 
                    velocity_penalty + 
                    direction_reward + 
                    stability_reward + 
                    precision_reward +
                    time_penalty)
        
        # 可选：添加调试信息
        if hasattr(self, 'debug_reward') and self.debug_reward:
            print(f"Distance: {distance:.4f}, Distance Reward: {distance_reward:.4f}")
            print(f"Improvement Reward: {improvement_reward:.4f}")
            print(f"Target Reward: {target_reward:.4f}")
            print(f"Velocity Penalty: {velocity_penalty:.4f}")
            print(f"Direction Reward: {direction_reward:.4f}")
            print(f"Stability Reward: {stability_reward:.4f}")
            print(f"Precision Reward: {precision_reward:.4f}")
            print(f"Total Reward: {total_reward:.4f}")
        
        return total_reward
    
    def _is_terminated(self):
        """检查是否终止"""
        # 获取末端执行器位置
        num_links = p.getNumJoints(self.robot_id)
        if num_links > 0:
            try:
                end_effector_state = p.getLinkState(self.robot_id, num_links - 1)
                end_effector_pos = np.array(end_effector_state[0])
            except:
                end_effector_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        else:
            end_effector_pos = np.array(p.getBasePositionAndOrientation(self.robot_id)[0])
        
        # 确保目标位置已初始化
        if not hasattr(self, 'target_position') or self.target_position is None:
            self.target_position = np.array([0.2, 0.1, 0.2])
        
        # 如果接近目标，任务完成
        distance = np.linalg.norm(end_effector_pos - self.target_position)
        return distance < 0.02
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            # GUI模式已经在显示
            pass
        
    def close(self):
        """关闭环境"""
        try:
            p.disconnect()
        except:
            pass

# 用于连续动作空间的Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(ActorCritic, self).__init__()
        
        print(f"🧠 创建ActorCritic网络:")
        print(f"   输入维度: {state_size}")
        print(f"   输出维度: {action_size}")
        print(f"   隐藏层维度: {hidden_size}")
        
        # 共享层
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor网络（输出动作均值）
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_std = nn.Linear(hidden_size, action_size)
        
        # Critic网络（输出状态值）
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # 打印输入维度用于调试
        # print(f"ActorCritic输入形状: {x.shape}")
        
        shared_out = self.shared(x)
        
        # Actor输出
        action_mean = torch.tanh(self.actor_mean(shared_out))
        action_std = F.softplus(self.actor_std(shared_out)) + 1e-3  # 修复：使用 F.softplus
        
        # Critic输出
        state_value = self.critic(shared_out)
        
        return action_mean, action_std, state_value


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, env, state_size, action_size, lr=3e-4):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 PPO训练器配置:")
        print(f"   设备: {self.device}")
        print(f"   状态空间大小: {state_size}")
        print(f"   动作空间大小: {action_size}")
        print(f"   学习率: {lr}")
        
        # 创建网络
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 超参数
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01
        self.value_coeff = 0.5
        
        # 存储经验
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        
    def collect_experience(self, num_steps):
        """收集经验"""
        try:
            state, _ = self.env.reset()
        except:
            # 如果重置失败，使用默认状态
            state = np.zeros(self.env.observation_space.shape[0])
        
        for step in range(num_steps):
            # 打印第一步的状态维度用于调试
            if step == 0:
                print(f"   第一步状态维度: {state.shape}")
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                try:
                    action_mean, action_std, _ = self.policy(state_tensor)
                except Exception as e:
                    print(f"   网络前向传播错误: {e}")
                    print(f"   状态张量形状: {state_tensor.shape}")
                    print(f"   期望输入维度: {self.env.observation_space.shape[0]}")
                    raise e
                
            # 从策略中采样动作
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(dim=-1)
            
            # 执行动作
            try:
                next_state, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
                done = terminated or truncated
            except Exception as e:
                print(f"步骤执行错误: {e}")
                # 使用默认值
                next_state = np.zeros_like(state)
                reward = 0.0
                done = True
            
            # 存储经验
            self.states.append(state)
            self.actions.append(action.cpu().numpy()[0])
            self.rewards.append(reward)
            self.dones.append(done)
            self.log_probs.append(log_prob.cpu().numpy()[0])
            
            state = next_state
            
            if done:
                try:
                    state, _ = self.env.reset()
                except:
                    state = np.zeros(self.env.observation_space.shape[0])
    
    def train(self, num_epochs=10):
        """训练策略"""
        if len(self.states) == 0:
            return
        
        # 转换为tensor
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # 计算优势和回报
        advantages, returns = self._compute_advantages()
        
        for _ in range(num_epochs):
            # 前向传播
            action_mean, action_std, state_values = self.policy(states)
            
            # 计算新的log概率
            action_dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = action_dist.log_prob(actions).sum(dim=-1)
            
            # 计算策略比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = nn.MSELoss()(state_values.squeeze(), returns)
            
            # 熵损失
            entropy = action_dist.entropy().mean()
            
            # 总损失
            total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
            
            # 更新
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # 清空经验
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
    
    def _compute_advantages(self):
        """计算优势和回报"""
        states = torch.FloatTensor(self.states).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        with torch.no_grad():
            _, _, values = self.policy(states)
            values = values.squeeze()
        
        advantages = []
        returns = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

def main():
    """主训练函数"""
    # 创建环境
    env = AlphaRobotEnv(render_mode="human")  # 设置为"human"可以看到训练过程
    
    # 获取状态和动作空间大小
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # 创建训练器
    trainer = PPOTrainer(env, state_size, action_size)
    
    # 训练参数
    num_iterations = 1000
    steps_per_iteration = 2048
    
    rewards_history = []
    
    print("Starting training...")
    
    for iteration in range(num_iterations):
        # 收集经验
        trainer.collect_experience(steps_per_iteration)
        
        # 训练
        trainer.train()
        
        # 测试当前策略
        if iteration % 10 == 0:
            test_reward = test_policy(env, trainer.policy)
            rewards_history.append(test_reward)
            print(f"Iteration {iteration}, Test Reward: {test_reward:.2f}")
            
            # 保存模型
            torch.save(trainer.policy.state_dict(), f'alpha_robot_model_{iteration}.pth')
    
    env.close()
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.title('Alpha Robot Training Progress')
    plt.xlabel('Iteration (x10)')
    plt.ylabel('Test Reward')
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()


def test_policy(env, policy, num_episodes=3):
    """测试策略"""
    total_reward = 0
    
    for _ in range(num_episodes):
        try:
            state, _ = env.reset()
        except:
            state = np.zeros(env.observation_space.shape[0])
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                action_mean, _, _ = policy(state_tensor)
            
            action = action_mean.cpu().numpy()[0]
            
            try:
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            except:
                done = True
                reward = 0
            
            steps += 1
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


if __name__ == "__main__":
    main()