# Underwater Alpha Robotic Arm - Deep Reinforcement Learning Training System

A comprehensive reinforcement learning training framework for underwater robotic arm reaching tasks using PyBullet simulation. This system supports multiple state-of-the-art RL algorithms with progressive curriculum learning, domain randomization, and publication-quality evaluation tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Training Modes](#training-modes)
- [Configuration](#configuration)
- [Algorithm Comparison](#algorithm-comparison)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a reinforcement learning system for training an underwater Alpha robotic arm to reach target positions. The environment simulates realistic underwater physics including:

- **Fluid drag** - Resistance forces on the end effector
- **Buoyancy** - Upward force counteracting gravity
- **Water currents** - Time-varying flow disturbances
- **Turbulence** - Random perturbations

The system supports three modern RL algorithms:
- **SAC** (Soft Actor-Critic)
- **TQC** (Truncated Quantile Critics)
- **CrossQ** (Cross Q-Learning)

## ✨ Features

- 🎓 **Curriculum Learning** - Progressive difficulty increase across 6 levels
- 🎲 **Domain Randomization** - Randomized physics parameters for robust policies
- 🎯 **Target Drift** - Moving targets that simulate real-world conditions
- 📊 **Comprehensive Monitoring** - Real-time training metrics and visualization
- 🎬 **Automatic GIF Generation** - Visual demonstrations of trained policies
- 📈 **Publication-Quality Plots** - Ready-to-use figures for research papers
- 🔬 **Algorithm Comparison** - Side-by-side performance analysis

## 📦 Requirements

### Dependencies

```bash
# Core dependencies
python >= 3.8
numpy
torch
gymnasium
pybullet
stable-baselines3
sb3-contrib

# Visualization
matplotlib
seaborn
pandas
pillow

# Optional
scipy
```

### Hardware

- **GPU**: CUDA-compatible GPU recommended for faster training
- **RAM**: Minimum 8GB, 16GB+ recommended for parallel environments
- **Storage**: ~2GB for models and logs per experiment

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd underwater-alpha-arm
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install torch numpy gymnasium pybullet
pip install stable-baselines3 sb3-contrib
pip install matplotlib seaborn pandas pillow scipy
```

4. **Verify URDF file**

Ensure the Alpha robotic arm URDF file is located at:
```
alpha_description/urdf/alpha_robot_for_pybullet.urdf
```

## 📁 Project Structure

```
underwater-alpha-arm/
├── train_v8.py                    # Main training script
├── rl_env_v7.py                   # Gymnasium environment (place in envs/)
├── compare_algorithms_enhanced.py  # Algorithm comparison system
├── run_comparison.py              # One-click comparison launcher
├── monitor_callbacks.py           # Training monitoring callbacks
├── curriculum_callback.py         # Curriculum learning callback (required)
├── envs/
│   └── rl_env_v7.py              # Environment module
├── alpha_description/
│   └── urdf/
│       └── alpha_robot_for_pybullet.urdf
├── experiments/                   # Training outputs (auto-generated)
└── comparison_results/            # Comparison outputs (auto-generated)
```

## 🚀 Quick Start

### Basic Training

```bash
# Train with default settings (SAC, 500K steps, stage4)
python train_v8.py

# Train with specific algorithm
python train_v8.py --algorithm TQC --timesteps 500000

# Train specific stage
python train_v8.py --stage stage1 --timesteps 300000
```

### Quick Algorithm Comparison

```bash
# Interactive launcher
python run_comparison.py

# Direct comparison
python compare_algorithms_enhanced.py --timesteps 500000 --stage stage2
```

## 🎮 Training Modes

### 1. Single Algorithm Training (`--mode train`)

Train a single algorithm with full visualization:

```bash
python train_v8.py --mode train \
    --algorithm SAC \
    --timesteps 500000 \
    --num_envs 8 \
    --stage stage4 \
    --seed 42
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--algorithm` | SAC | RL algorithm: SAC, TQC, CrossQ |
| `--timesteps` | 500000 | Total training steps |
| `--num_envs` | 8 | Number of parallel environments |
| `--stage` | stage4 | Training stage (stage1-4) |
| `--seed` | 42 | Random seed for reproducibility |
| `--auto_visualize` | True | Generate GIFs after training |

### 2. Model Testing (`--mode test`)

Test a trained model with visualization:

```bash
python train_v8.py --mode test \
    --model ./experiments/SAC_stage4_500000steps_*/models/SAC_final \
    --algorithm SAC \
    --episodes 10
```

### 3. Algorithm Comparison (`--mode compare`)

Compare all three algorithms:

```bash
python train_v8.py --mode compare --timesteps 500000
```

## ⚙️ Configuration

### Training Stages

The system uses progressive curriculum learning with 4 stages:

| Stage | Description | Features | Recommended Steps |
|-------|-------------|----------|-------------------|
| `stage1` | Basic | Static target, all features enabled | 500K |
| `stage2` | Domain Randomization | Randomized physics parameters | 500K |
| `stage3` | Target Drift | Moving targets with sensor noise | 500K |
| `stage4` | Full Curriculum | Progressive difficulty, final version | 1M |

### Curriculum Levels (within each stage)

The curriculum automatically advances through 6 difficulty levels:

| Level | Drift Strength | Success Threshold | Episodes to Advance |
|-------|----------------|-------------------|---------------------|
| 0 | 0.0253 m/s | 4 cm | 50 |
| 1 | 0.0355 m/s | 4 cm | 50 |
| 2 | 0.0456 m/s | 4 cm | 50 |
| 3 | 0.0558 m/s | 4 cm | 50 |
| 4 | 0.0659 m/s | 4 cm | 50 |
| 5 | 0.0760 m/s | 4 cm | ∞ (final) |

Advancement requires >70% success rate over the episode window.

### Environment Parameters

Key environment parameters in `rl_env_v7.py`:

```python
# Physics
water_density = 1000.0      # kg/m³
drag_coefficient = 0.5      # Fluid drag
buoyancy_compensation = 0.34 # 34% gravity offset

# Sensor Noise
position_noise_std = 0.002   # Joint position noise
velocity_noise_std = 0.008   # Joint velocity noise
ee_position_noise_std = 0.003 # End-effector noise (0.3cm)

# Rewards
success_threshold = 0.03     # 3cm for success
success_bonus = 10.0         # Bonus for reaching target
```

### Algorithm Hyperparameters

Default configurations in `train_v8.py`:

**SAC:**
```python
learning_rate = 3e-4
buffer_size = 500000
batch_size = 512
gamma = 0.98
tau = 0.005
net_arch = [256, 256, 256]
```

**TQC:**
```python
learning_rate = 3e-4
buffer_size = 500000
batch_size = 256
n_quantiles = 25
top_quantiles_to_drop = 2
```

**CrossQ:**
```python
learning_rate = 3e-4
buffer_size = 300000
batch_size = 256
n_critics = 2
```

## 📊 Algorithm Comparison

### Using the Interactive Launcher

```bash
python run_comparison.py
```

Options:
1. **Quick Test** (10K steps, ~30 min) - For code verification
2. **Standard** (200K steps, ~5 hours) - For paper drafts
3. **Full** (600K steps, ~10 hours) - For final results
4. **Custom** - Configure your own settings

### Using the Comparison Script Directly

```bash
python compare_algorithms_enhanced.py \
    --algorithms SAC TQC CrossQ \
    --timesteps 500000 \
    --stage stage4 \
    --num_envs 1 \
    --seed 42 \
    --save_dir comparison_results
```

### Generated Outputs

The comparison system generates:

1. **Learning Curves** (`1_learning_curves.png`)
   - Episode returns over training
   - Curriculum stage markers
   - Full training + final 20% view

2. **Success Rate Curves** (`2_success_rate_curves.png`)
   - Success rate progression during training

3. **Final Performance** (`3_final_performance.png`)
   - Success rate comparison
   - Precision (final distance)
   - Episode returns
   - Training stability (CV)

4. **Sample Efficiency** (`4_sample_efficiency.png`)
   - Steps to convergence comparison

5. **Statistical Comparison** (`5_statistical_comparison.png`)
   - Box plots of rewards and distances
   - Distribution analysis

6. **Radar Chart** (`6_comprehensive_radar.png`)
   - Multi-dimensional performance comparison

7. **Text Report** (`comparison_report.txt`)
   - Detailed statistics and rankings

## 📂 Output Files

### Experiment Directory Structure

```
experiments/SAC_stage4_500000steps_20241121_120000/
├── config.json              # Experiment configuration
├── training_summary.json    # Final results summary
├── models/
│   ├── SAC_final.zip       # Final trained model
│   ├── SAC_vecnormalize.pkl # Environment normalization
│   ├── SAC_best/           # Best checkpoint
│   └── checkpoints/        # Periodic checkpoints
├── logs/
│   ├── *.monitor.csv       # Episode statistics
│   ├── tensorboard/        # TensorBoard logs
│   ├── curriculum_history.json # Curriculum progression
│   ├── metrics/            # Periodic metric snapshots
│   └── training_report.json # Training statistics
├── plots/
│   ├── training_curves.png  # Training progression
│   └── final_training_results.png
└── videos/
    ├── episode_1.gif       # Demo episode 1
    ├── episode_2.gif       # Demo episode 2
    └── episode_3.gif       # Demo episode 3
```

### Key Output Files

| File | Description |
|------|-------------|
| `*_final.zip` | Trained model weights |
| `*_vecnormalize.pkl` | Observation/reward normalization stats |
| `training_summary.json` | Final metrics (reward, time, etc.) |
| `curriculum_history.json` | Curriculum stage transitions |
| `*.gif` | Demonstration videos of trained policy |

## 🔍 Monitoring Training

### Console Output

Training progress is printed every 10 episodes:
```
Ep  100 | R:   -2.5 | SR:  15.0% | Dist:  8.2cm
Ep  110 | R:   -1.8 | SR:  22.0% | Dist:  6.5cm
```

### TensorBoard

```bash
tensorboard --logdir experiments/*/logs/tensorboard
```

### Curriculum Advancement

Curriculum transitions are logged:
```
============================================================
[Curriculum Advance] Level 0 -> 1
  Success rate: 75.0% (last 50 episodes)
  New workspace radius: 0.40m
  New success threshold: 0.040m
============================================================
```

## ❓ Troubleshooting

### Common Issues

**1. URDF file not found**
```
FileNotFoundError: Cannot find Alpha robotic arm URDF file
```
Solution: Ensure URDF is at `alpha_description/urdf/alpha_robot_for_pybullet.urdf`

**2. OpenMP conflict**
```
OMP: Error #15: Initializing libiomp5md.dll
```
Solution: Already handled by `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`

**3. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce `buffer_size` or `batch_size`, or use `device='cpu'`

**4. curriculum_callback not found**
```
ModuleNotFoundError: No module named 'curriculum_callback'
```
Solution: Create the curriculum callback file (see below)

### Creating curriculum_callback.py

If missing, create `curriculum_callback.py`:

```python
import os
import json
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumMonitorCallback(BaseCallback):
    def __init__(self, log_dir, verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.transitions = []
        self.episodes_per_stage = {}
        self.current_stage = None
        self.episode_count = 0
        self.success_count = 0
        
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                if info.get('is_success', False):
                    self.success_count += 1
                    
                # Check for stage change
                env = self.training_env.envs[0]
                if hasattr(env, 'env'):
                    env = env.env
                stage = getattr(env, 'curriculum_stage', 0)
                
                if self.current_stage is None:
                    self.current_stage = stage
                elif stage != self.current_stage:
                    self.transitions.append({
                        'timestep': self.num_timesteps,
                        'episode': self.episode_count,
                        'old_stage': self.current_stage,
                        'new_stage': stage
                    })
                    self.current_stage = stage
        return True
    
    def _on_training_end(self):
        history = {
            'total_episodes': self.episode_count,
            'total_successes': self.success_count,
            'transitions': self.transitions,
            'final_stage': self.current_stage
        }
        path = os.path.join(self.log_dir, 'curriculum_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{underwater_alpha_arm,
  title = {Underwater Alpha Robotic Arm RL Training System},
  year = {2024},
  description = {Deep reinforcement learning for underwater robotic manipulation}
}
```

## 📄 License

This project is provided for research and educational purposes.

---

For questions or issues, please open an issue on the repository.
