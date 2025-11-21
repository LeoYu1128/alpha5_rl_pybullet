"""
只重新生成图表，不重新训练
从已有的实验数据中读取并生成publication-quality图表
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime

# ============================================================================
# 🎯 配置：填入你已有的实验目录
# ============================================================================
# 方法1：自动查找最新的实验
AUTO_FIND = True

# 方法2：手动指定（如果AUTO_FIND=False）
EXPERIMENTS = {
    'SAC': r'experiments\SAC_stage4_600000steps_20251119_102248',
    'TQC': r'experiments\TQC_stage4_600000steps_20251119_012603',
    'CrossQ': r'experiments\CrossQ_stage4_600000steps_20251119_050909'
}

OUTPUT_DIR = 'comparison_results_regenerated'
# ============================================================================

def find_latest_experiments():
    """自动查找最新的实验目录"""
    experiments = {}
    for algo in ['SAC', 'TQC', 'CrossQ']:
        pattern = f'experiments/{algo}_stage4_600000steps_*'
        dirs = glob.glob(pattern)
        if dirs:
            # 按时间排序，取最新的
            latest = max(dirs, key=os.path.getmtime)
            experiments[algo] = latest
            print(f"✅ Found {algo}: {latest}")
        else:
            print(f"⚠️  {algo}: No experiments found")
    return experiments

def load_learning_curve(exp_dir, algo_name):
    """从monitor.csv加载学习曲线"""
    log_dir = os.path.join(exp_dir, 'logs')
    monitor_files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
    
    if not monitor_files:
        print(f"   ⚠️  No monitor files found in {log_dir}")
        return None
    
    all_data = []
    for f in monitor_files:
        try:
            df = pd.read_csv(f, comment='#')
            if 'r' in df.columns and 'l' in df.columns:
                # Clean data
                df['r'] = df['r'].astype(str).str.replace('--', '-', regex=False)
                df['r'] = pd.to_numeric(df['r'], errors='coerce')
                df['l'] = pd.to_numeric(df['l'], errors='coerce')
                df = df.dropna()
                if not df.empty:
                    all_data.append(df)
        except Exception as e:
            print(f"   ⚠️  Error reading {f}: {e}")
            continue
    
    if not all_data:
        print(f"   ⚠️  No valid data for {algo_name}")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    cumulative_timesteps = combined['l'].cumsum()
    
    # Smooth rewards
    window = min(50, len(combined))
    smoothed = combined['r'].rolling(window=window, min_periods=1).mean()
    
    print(f"   ✓ Loaded {len(combined)} episodes")
    print(f"     Timesteps: [{cumulative_timesteps.iloc[0]:.0f}, {cumulative_timesteps.iloc[-1]:.0f}]")
    print(f"     Rewards: [{combined['r'].min():.1f}, {combined['r'].max():.1f}], mean={combined['r'].mean():.1f}")
    
    return {
        'timesteps': cumulative_timesteps.values,
        'rewards': combined['r'].values,
        'smoothed': smoothed.values
    }

def load_curriculum_history(exp_dir, algo_name):
    """加载curriculum历史"""
    curriculum_file = os.path.join(exp_dir, 'logs', 'curriculum_history.json')
    if os.path.exists(curriculum_file):
        try:
            with open(curriculum_file, 'r') as f:
                data = json.load(f)
            print(f"   ✓ Curriculum: {data['total_stage_transitions']} transitions, final stage: {data['final_stage']}")
            return data
        except Exception as e:
            print(f"   ⚠️  Error loading curriculum: {e}")
    else:
        print(f"   ⚠️  No curriculum_history.json found")
    return None

def plot_learning_curves_with_curriculum(all_data, all_curriculum, output_dir):
    """绘制带curriculum的学习曲线"""
    print("\n" + "="*60)
    print("📊 Generating Learning Curves with Curriculum Stages")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'lines.linewidth': 2.0,
    })
    
    colors = {'SAC': '#1f77b4', 'TQC': '#d62728', 'CrossQ': '#ff7f0e'}
    
    if not all_data:
        print("❌ No data to plot!")
        return
    
    # 绘制所有算法的曲线
    all_rewards = []
    algo_curricula = {}  # 存储每个算法的curriculum数据
    
    for algo, lc in all_data.items():
        color = colors.get(algo, '#000000')
        timesteps = lc['timesteps']
        rewards = lc['rewards']
        smoothed = lc['smoothed']
        
        # 过滤初始异常值：忽略前1%的数据用于Y轴计算
        start_idx = int(len(smoothed) * 0.01)
        filtered_smoothed = smoothed[start_idx:]
        all_rewards.extend(filtered_smoothed)
        
        print(f"\n📈 Plotting {algo}:")
        print(f"   Points: {len(timesteps)}")
        print(f"   Reward range (full): [{smoothed.min():.1f}, {smoothed.max():.1f}]")
        print(f"   Reward range (filtered): [{filtered_smoothed.min():.1f}, {filtered_smoothed.max():.1f}]")
        
        # 左图：只画平滑曲线（去掉毛茸茸的原始曲线）
        # ax1.plot(timesteps, rewards, alpha=0.2, color=color, linewidth=0.8)  # ← 注释掉这行
        line, = ax1.plot(timesteps, smoothed, label=algo, color=color, linewidth=2.5)
        
        # 保存curriculum数据供后续标注
        if algo in all_curriculum:
            curriculum = all_curriculum[algo]
            valid_trans = [t for t in curriculum['transitions'] if t['old_stage'] is not None]
            if valid_trans:
                algo_curricula[algo] = valid_trans
                print(f"   Found {len(valid_trans)} stage transitions")
        
        # 右图：最后20%
        cutoff = int(len(timesteps) * 0.8)
        if cutoff < len(timesteps):
            ax2.plot(timesteps[cutoff:], smoothed[cutoff:], 
                    label=algo, color=color, linewidth=2.5)
    
    # 设置Y轴范围
    if all_rewards:
        y_min = np.percentile(all_rewards, 5)
        y_max = np.percentile(all_rewards, 95)
        y_range = y_max - y_min if y_max != y_min else 100
        ax1.set_ylim(y_min - y_range*0.15, y_max + y_range*0.25)  # 顶部留更多空间
        print(f"\n📊 Y-axis range: [{y_min:.1f}, {y_max:.1f}] (5th-95th percentile)")
    
    # 🆕 添加curriculum标注 - 用不同高度分开不同算法
    y_lim = ax1.get_ylim()
    label_heights = {
        'CrossQ': y_lim[1] * 0.98,  # 最上面
        'TQC': y_lim[1] * 0.92,      # 中间
        'SAC': y_lim[1] * 0.86       # 稍低
    }
    
    for algo in ['CrossQ', 'TQC', 'SAC']:
        if algo not in algo_curricula:
            continue
        
        color = colors.get(algo, '#000000')
        transitions = algo_curricula[algo]
        y_pos = label_heights[algo]
        
        print(f"\n🎓 Adding stage markers for {algo} at height {y_pos:.1f}")
        
        for trans in transitions:
            timestep = trans['timestep']
            new_stage = trans['new_stage']
            
            # 画细虚线
            ax1.axvline(x=timestep, color=color, linestyle=':', 
                       alpha=0.3, linewidth=1.0, zorder=1)
            
            # 在对应高度添加小标签
            ax1.text(timestep, y_pos, f'{new_stage}', 
                    fontsize=7, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.2', 
                            facecolor=color, 
                            edgecolor='white',
                            linewidth=1.5,
                            alpha=0.9),
                    zorder=10)
    
    # 设置图表样式
    ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
    ax1.set_title('Learning Curves with Curriculum Stage Transitions', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 添加说明文字
    ax1.text(0.98, 0.02, 'Stage numbers at top (colored by algorithm)', 
            transform=ax1.transAxes, fontsize=9,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8,
                     edgecolor='gray', linewidth=1))
    
    ax2.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
    ax2.set_title('Learning Curves (Final 20%)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    save_path = os.path.join(output_dir, 'plots', 'learning_curves_with_curriculum.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved: {save_path}")

def generate_curriculum_report(all_curriculum, output_dir):
    """生成curriculum文本报告"""
    report_path = os.path.join(output_dir, 'curriculum_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Curriculum Learning Progression Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for algo in ['SAC', 'TQC', 'CrossQ']:
            if algo not in all_curriculum:
                f.write(f"{algo}: No curriculum data\n\n")
                continue
            
            curriculum = all_curriculum[algo]
            
            f.write(f"{algo}:\n")
            f.write("-"*80 + "\n")
            f.write(f"  Total Episodes: {curriculum['total_episodes']}\n")
            f.write(f"  Total Successes: {curriculum['total_successes']}\n")
            f.write(f"  Overall Success Rate: {curriculum['overall_success_rate']*100:.1f}%\n")
            f.write(f"  Total Stage Transitions: {curriculum['total_stage_transitions']}\n")
            f.write(f"  Final Stage: {curriculum['final_stage']}\n\n")
            
            # Episodes per stage
            episodes_per_stage = curriculum['episodes_per_stage']
            f.write("  Episodes per Stage:\n")
            for stage in sorted(episodes_per_stage.keys(), key=lambda x: int(x)):
                count = episodes_per_stage[stage]
                f.write(f"    Stage {stage}: {count} episodes\n")
            
            # Transition timeline
            valid_trans = [t for t in curriculum['transitions'] if t['old_stage'] is not None]
            if valid_trans:
                f.write("\n  Stage Transition Timeline:\n")
                for i, trans in enumerate(valid_trans, 1):
                    f.write(f"\n    Transition {i}: Stage {trans['old_stage']} → {trans['new_stage']}\n")
                    f.write(f"      Timestep: {trans['timestep']:,}\n")
                    f.write(f"      Episode: {trans['episode']}\n")
                    f.write(f"      Episodes in previous stage: {trans['episodes_in_previous_stage']}\n")
                    
                    config = trans.get('stage_config', {})
                    if config:
                        f.write(f"      Drift strength: {config.get('drift_strength', 0):.3f} m/s\n")
                        f.write(f"      Success threshold: {config.get('success_threshold', 0)*100:.1f} cm\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"✅ Curriculum report saved: {report_path}")

def main():
    print("\n" + "="*80)
    print("🎨 Regenerating Plots from Existing Training Data")
    print("   (No retraining required)")
    print("="*80 + "\n")
    
    # 查找或使用指定的实验目录
    if AUTO_FIND:
        print("🔍 Auto-finding latest experiments...")
        experiments = find_latest_experiments()
    else:
        experiments = EXPERIMENTS
        print("📁 Using specified experiment directories:")
        for algo, path in experiments.items():
            print(f"   {algo}: {path}")
    
    if not experiments:
        print("\n❌ No experiments found!")
        print("   Please set AUTO_FIND=False and manually specify EXPERIMENTS paths")
        return
    
    print("\n" + "-"*60)
    
    # 加载所有数据
    all_data = {}
    all_curriculum = {}
    
    for algo, exp_dir in experiments.items():
        print(f"\n📂 Loading {algo} from: {exp_dir}")
        
        if not os.path.exists(exp_dir):
            print(f"   ⚠️  Directory not found!")
            continue
        
        # 加载学习曲线
        lc = load_learning_curve(exp_dir, algo)
        if lc:
            all_data[algo] = lc
        
        # 加载curriculum历史
        curriculum = load_curriculum_history(exp_dir, algo)
        if curriculum:
            all_curriculum[algo] = curriculum
    
    if not all_data:
        print("\n❌ No training data found!")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    
    # 生成图表
    plot_learning_curves_with_curriculum(all_data, all_curriculum, OUTPUT_DIR)
    
    # 生成报告
    if all_curriculum:
        generate_curriculum_report(all_curriculum, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("✅ Done! No training was performed.")
    print(f"📊 Check results in: {OUTPUT_DIR}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()