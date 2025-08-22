#!/usr/bin/env python3
"""
檢查數據集結構腳本
"""
import os

def check_dataset():
    base_path = r"D:\thesis\alpha5_rl_pybullet\trash_ICRA19\trash_ICRA19\dataset"
    
    print("檢查數據集結構...")
    print(f"基礎路徑: {base_path}")
    
    if not os.path.exists(base_path):
        print("❌ 基礎路徑不存在!")
        return False
    
    # 檢查子目錄
    subdirs = ['train', 'val', 'test']
    for subdir in subdirs:
        full_path = os.path.join(base_path, subdir)
        if os.path.exists(full_path):
            # 檢查圖片和標註文件
            all_files = os.listdir(full_path)
            images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            labels = [f for f in all_files if f.endswith('.txt')]
            
            print(f"✓ {subdir}: {len(images)} 圖片, {len(labels)} 標註文件")
            
            # 檢查是否有配對的文件
            if len(images) > 0:
                sample_img = images[0]
                sample_label = sample_img.rsplit('.', 1)[0] + '.txt'
                if sample_label in labels:
                    print(f"  ✓ 標註文件格式正確 (例: {sample_img} <-> {sample_label})")
                else:
                    print(f"  ⚠ 可能的標註問題 (例: {sample_img} 沒有對應的 {sample_label})")
            
            # 檢查標註文件內容示例
            if len(labels) > 0:
                sample_label_path = os.path.join(full_path, labels[0])
                try:
                    with open(sample_label_path, 'r') as f:
                        first_line = f.readline().strip()
                        print(f"  標註示例: {first_line}")
                except:
                    print(f"  ⚠ 無法讀取標註文件: {labels[0]}")
        else:
            print(f"❌ {subdir} 目錄不存在: {full_path}")
    
    return True

if __name__ == "__main__":
    check_dataset()