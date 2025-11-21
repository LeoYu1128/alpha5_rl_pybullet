#!/usr/bin/env python3
"""
One-Click Launcher for Modern RL Algorithm Comparison
Generates publication-ready comparison data and visualizations
"""

import sys
import os

def print_banner():
    print("\n" + "="*80)
    print("🚀 Modern RL Algorithm Comparison System")
    print("   For Underwater Robotic Arm Research")
    print("="*80)
    print("\n📊 This experiment will compare:")
    print("   • SAC (Soft Actor-Critic)")
    print("   • TQC (Truncated Quantile Critics)")
    print("   • CrossQ (Cross Q-Learning)")
    print("\n📈 Generated publication-quality outputs:")
    print("   • Success rate curves over training")
    print("   • Episode return learning curves")
    print("   • Sample efficiency comparison")
    print("   • Statistical analysis with boxplots")
    print("   • Comprehensive performance radar chart")
    print("   • Demonstration videos (GIFs)")
    print("   • Detailed text report")
    print("   • Raw data (JSON format)")
    print("\n" + "="*80 + "\n")

def get_user_choice():
    """Get user's experiment configuration choice"""
    print("Please select experiment configuration:\n")
    print("1. 🏃 Quick Test (50K steps, ~30 minutes)")
    print("   For: Code verification, quick preview")
    print("")
    print("2. 📊 Standard Experiment (500K steps, ~5 hours)")
    print("   For: Paper draft, regular comparison")
    print("")
    print("3. 🎯 Full Experiment (1M steps, ~10 hours)")
    print("   For: Final paper version, best results")
    print("")
    print("4. 🔧 Custom Configuration")
    print("")
    
    while True:
        choice = input("Enter option (1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return choice
        print("❌ Invalid choice, please try again")

def get_custom_config():
    """Get custom configuration"""
    print("\nCustom Configuration:")
    
    # Training steps
    while True:
        timesteps_str = input("Training steps (recommended: 50000, 500000, 1000000): ").strip()
        try:
            timesteps = int(timesteps_str)
            if timesteps > 0:
                break
            print("❌ Steps must be greater than 0")
        except ValueError:
            print("❌ Please enter a valid number")
    
    # Training stage
    print("\nAvailable training stages:")
    print("  stage1: Basic version - static target")
    print("  stage2: Domain randomization - more robust")
    print("  stage3: Target drift - more realistic")
    print("  stage4: Curriculum learning - final version")
    
    while True:
        stage = input("Select stage (stage1-stage4, or Enter for default): ").strip()
        if stage == '':
            stage = None
            break
        if stage in ['stage1', 'stage2', 'stage3', 'stage4']:
            break
        print("❌ Invalid stage, please try again")
    
    # Algorithm selection
    print("\nAvailable algorithms: SAC, TQC, CrossQ")
    algorithms_str = input("Select algorithms (space-separated, or Enter for all): ").strip()
    if algorithms_str == '':
        algorithms = ['SAC', 'TQC', 'CrossQ']
    else:
        algorithms = algorithms_str.upper().split()
        valid_algos = ['SAC', 'TQC', 'CROSSQ']
        algorithms = [a for a in algorithms if a in valid_algos]
        if not algorithms:
            print("⚠️  No valid algorithms, using all by default")
            algorithms = ['SAC', 'TQC', 'CrossQ']
    
    return timesteps, stage, algorithms

def run_comparison(timesteps, stage, algorithms):
    """Run comparison experiment"""
    import subprocess
    
    cmd = [
        sys.executable,
        'compare_algorithms_enhanced.py',
        '--timesteps', str(timesteps),
        '--algorithms', *algorithms
    ]
    
    if stage is not None:
        cmd.extend(['--stage', stage])
    
    print("\n" + "="*80)
    print("🚀 Starting experiment...")
    print(f"Command: {' '.join(cmd)}")
    print("="*80 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        return False

def main():
    """Main function"""
    print_banner()
    
    choice = get_user_choice()
    
    # Configuration mapping
    configs = {
        '1': (10000, 'stage1', ['SAC', 'TQC', 'CrossQ']),
        '2': (200000, 'stage2', ['SAC', 'TQC', 'CrossQ']),
        '3': (600000, 'stage4', ['SAC', 'TQC', 'CrossQ']),
    }
    
    if choice in ['1', '2', '3']:
        timesteps, stage, algorithms = configs[choice]
        print(f"\n✅ Selected configuration:")
        print(f"   Training steps: {timesteps:,}")
        print(f"   Training stage: {stage}")
        print(f"   Algorithms: {', '.join(algorithms)}")
        
        # Estimate time
        time_estimates = {
            '1': "approximately 30 minutes",
            '2': "approximately 5 hours",
            '3': "approximately 10 hours"
        }
        print(f"   Estimated time: {time_estimates[choice]}")
        
        confirm = input("\nConfirm to start experiment? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Cancelled")
            return
    else:
        timesteps, stage, algorithms = get_custom_config()
        print(f"\n✅ Custom configuration:")
        print(f"   Training steps: {timesteps:,}")
        print(f"   Training stage: {stage if stage else 'default'}")
        print(f"   Algorithms: {', '.join(algorithms)}")
    
    # Run experiment
    success = run_comparison(timesteps, stage, algorithms)
    
    if success:
        print("\n" + "="*80)
        print("🎉 Experiment completed successfully!")
        print("="*80)
        print("\n📁 Result file locations:")
        print("   • Text report: comparison_results/comparison_*/comparison_report.txt")
        print("   • Plots directory: comparison_results/comparison_*/plots/")
        print("   • Videos directory: comparison_results/comparison_*/videos/")
        print("   • Raw data: comparison_results/comparison_*/data/")
        print("\n💡 Next steps:")
        print("   1. Read comparison_report.txt for detailed results")
        print("   2. Use plots for paper figures")
        print("   3. Check videos for qualitative analysis")
        print("\n" + "="*80 + "\n")
    else:
        print("\n" + "="*80)
        print("❌ Experiment incomplete")
        print("="*80)
        print("\nPlease check:")
        print("   1. All dependencies are correctly installed")
        print("   2. Training script (train_v8.py) is available")
        print("   3. Environment file (rl_env_v7.py) is available")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
        sys.exit(1)