"""
Alpha Underwater Reach Environment Configuration for Isaac Lab
针对RTX 3070优化的配置
"""

import math
from dataclasses import MISSING
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Scene definition
##

@configclass
class AlphaReachSceneCfg(InteractiveSceneCfg):
    """场景配置 - 水下环境"""

    # 地面 (海底)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(
            color=(0.1, 0.2, 0.4),  # 深蓝色海底
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.3,
                restitution=0.0,
            ),
        ),
    )

    # Alpha机器人 (水下配置)
    robot: ArticulationCfg = MISSING

    # 目标标记 (橙色球体)
    target = AssetBaseCfg(
        prim_path="/World/target",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, 0.25)),
    )

    # 照明 (模拟水下光照)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=800.0,
            color=(0.7, 0.8, 1.0),  # 偏蓝色调
        ),
    )


##
# MDP settings - 观察、奖励、终止条件
##

@configclass
class ObservationsCfg:
    """观察空间配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观察"""
        
        # 关节位置 (4D)
        joint_pos = ObsTerm(func=lambda env: env.scene.robot.data.joint_pos[:, :4])
        
        # 关节速度 (4D)
        joint_vel = ObsTerm(func=lambda env: env.scene.robot.data.joint_vel[:, :4])
        
        # 末端执行器位置 (3D)
        ee_position = ObsTerm(
            func=lambda env: env.scene.robot.data.body_pos_w[:, env.cfg.ee_body_id, :3]
        )
        
        # 目标位置 (3D)
        target_position = ObsTerm(func=lambda env: env.target_position)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """奖励函数配置"""
    
    # 距离奖励 (主要)
    distance_to_target = RewTerm(
        func=lambda env: -torch.norm(
            env.scene.robot.data.body_pos_w[:, env.cfg.ee_body_id, :3] - env.target_position,
            dim=-1
        ),
        weight=1.0,
    )
    
    # 接近目标奖励
    reaching_progress = RewTerm(
        func=lambda env: env.prev_distance - torch.norm(
            env.scene.robot.data.body_pos_w[:, env.cfg.ee_body_id, :3] - env.target_position,
            dim=-1
        ),
        weight=1.4,
    )
    
    # 控制代价
    action_penalty = RewTerm(
        func=lambda env: -torch.sum(env.actions ** 2, dim=-1),
        weight=0.01,
    )
    
    # 速度惩罚 (水下要平滑)
    velocity_penalty = RewTerm(
        func=lambda env: -torch.sum(env.scene.robot.data.joint_vel[:, :4] ** 2, dim=-1),
        weight=0.02,
    )
    
    # 成功奖励
    success_bonus = RewTerm(
        func=lambda env: (torch.norm(
            env.scene.robot.data.body_pos_w[:, env.cfg.ee_body_id, :3] - env.target_position,
            dim=-1
        ) < env.cfg.success_threshold).float(),
        weight=10.0,
    )


@configclass
class TerminationsCfg:
    """终止条件配置"""
    
    # 时间限制
    time_out = DoneTerm(func=lambda env: env.episode_length_buf >= env.max_episode_length, time_out=True)
    
    # 成功达到目标
    success = DoneTerm(
        func=lambda env: torch.norm(
            env.scene.robot.data.body_pos_w[:, env.cfg.ee_body_id, :3] - env.target_position,
            dim=-1
        ) < env.cfg.success_threshold
    )


@configclass
class EventCfg:
    """事件配置 - 域随机化"""
    
    # 重置时随机化机器人姿态
    reset_robot_joints = EventTerm(
        func=lambda env, env_ids: env.scene.robot.set_joint_position_target(
            env.scene.robot.data.default_joint_pos[env_ids] + 
            torch.randn_like(env.scene.robot.data.default_joint_pos[env_ids]) * 0.1,
            env_ids=env_ids
        ),
        mode="reset",
    )
    
    # 重置时随机化目标位置
    reset_target_position = EventTerm(
        func=lambda env, env_ids: env._sample_target_positions(env_ids),
        mode="reset",
    )


##
# 环境配置
##

@configclass
class AlphaReachEnvCfg(ManagerBasedRLEnvCfg):
    """Alpha机器人到达任务环境配置"""
    
    # 场景设置
    scene: AlphaReachSceneCfg = AlphaReachSceneCfg(num_envs=4096, env_spacing=2.0)
    
    # 观察、奖励、终止
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    # 环境参数
    episode_length_s = 10.0  # 10秒episode
    decimation = 4  # 控制频率降采样 (240Hz → 60Hz)
    
    # 任务参数
    success_threshold: float = 0.03  # 3cm成功阈值
    workspace_radius: float = 0.4  # 40cm工作空间
    ee_body_id: int = -1  # 末端执行器body ID (在环境初始化时设置)
    
    # 水下物理参数
    water_density: float = 1000.0
    drag_coefficient: float = 0.5
    buoyancy_compensation: float = 0.338  # 33.8%浮力补偿
    
    # 课程学习
    enable_curriculum: bool = True
    curriculum_levels: list = None  # 在__post_init__中设置
    
    # 域随机化
    enable_domain_randomization: bool = True
    
    def __post_init__(self):
        """配置后处理"""
        self.sim.dt = 1.0 / 240.0  # 240Hz物理步
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        
        # 课程学习配置
        if self.curriculum_levels is None:
            self.curriculum_levels = [
                {"workspace_radius": 0.4, "success_threshold": 0.08, "drift_strength": 0.0253},
                {"workspace_radius": 0.4, "success_threshold": 0.06, "drift_strength": 0.0355},
                {"workspace_radius": 0.4, "success_threshold": 0.05, "drift_strength": 0.0456},
                {"workspace_radius": 0.4, "success_threshold": 0.04, "drift_strength": 0.0558},
                {"workspace_radius": 0.4, "success_threshold": 0.03, "drift_strength": 0.0659},
            ]


##
# RTX 3070 优化配置
##

@configclass
class AlphaReachEnvCfg_RTX3070(AlphaReachEnvCfg):
    """针对RTX 3070优化的配置"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # RTX 3070: 8GB VRAM, 5888 CUDA cores
        # 推荐并行环境数: 2048-4096
        self.scene.num_envs = 2048
        
        # 优化内存使用
        self.sim.physx.gpu_max_rigid_contact_count = 2**22  # 降低以节省内存
        self.sim.physx.gpu_max_rigid_patch_count = 2**22
        
        # 提高性能
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**21
        self.sim.physx.gpu_collision_stack_size = 2**26
        
        print(f"✓ RTX 3070 优化配置: {self.scene.num_envs} 并行环境")