#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高級YOLOv11s垃圾檢測訓練腳本
作者: Leo Yu
項目: 海洋垃圾檢測 - Trash_ICRA19數據集
GPU: RTX 3070
"""

import os
import sys
import time
import yaml
import torch
import logging
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import matplotlib.pyplot as plt
import pandas as pd

class TrashDetectionTrainer:
    def __init__(self, config_path="trash_dataset.yaml"):
        """初始化訓練器"""
        self.config_path = config_path
        self.start_time = None
        self.model = None
        self.results = None
        
        # 設置日誌
        self.setup_logging()
        
        # 檢查環境
        self.check_environment()
        
        # 加載配置
        self.load_config()
        
    def setup_logging(self):
        """設置日誌記錄"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*80)
        self.logger.info("YOLOv11s 海洋垃圾檢測訓練開始")
        self.logger.info("="*80)
        
    def check_environment(self):
        """檢查訓練環境"""
        self.logger.info("🔍 檢查訓練環境...")
        
        # 檢查GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"✅ GPU: {gpu_name}")
            self.logger.info(f"✅ GPU內存: {gpu_memory:.1f} GB")
            self.logger.info(f"✅ CUDA版本: {torch.version.cuda}")
            self.device = 'cuda'
        else:
            self.logger.warning("⚠️  未檢測到GPU，將使用CPU訓練")
            self.device = 'cpu'
            
        # 檢查Python和PyTorch版本
        self.logger.info(f"✅ Python版本: {sys.version.split()[0]}")
        self.logger.info(f"✅ PyTorch版本: {torch.__version__}")
        
    def load_config(self):
        """加載數據集配置"""
        self.logger.info("📋 加載數據集配置...")
        
        if not os.path.exists(self.config_path):
            self.logger.error(f"❌ 配置文件不存在: {self.config_path}")
            sys.exit(1)
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.logger.info(f"✅ 數據集路徑: {self.config['path']}")
        self.logger.info(f"✅ 類別數量: {self.config['nc']}")
        self.logger.info(f"✅ 類別名稱: {list(self.config['names'].values())}")
        
        # 驗證數據集路徑
        if not os.path.exists(self.config['path']):
            self.logger.error(f"❌ 數據集路徑不存在: {self.config['path']}")
            sys.exit(1)
            
    def setup_training_params(self):
        """設置訓練參數"""
        self.logger.info("⚙️  設置訓練參數...")
        
        # 根據GPU調整批次大小
        if self.device == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 6:
                batch_size = 8
                self.logger.warning("⚠️  GPU內存較小，調整批次大小為8")
            elif gpu_memory < 8:
                batch_size = 12
            else:
                batch_size = 16
        else:
            batch_size = 4
            
        self.training_params = {
            'data': self.config_path,
            'epochs': 100,
            'imgsz': 640,
            'batch': batch_size,
            'device': self.device,
            'project': 'trash_detection',
            'name': f'yolov11s_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'patience': 30,
            'save': True,
            'save_period': 10,
            'plots': True,
            'val': True,
            'cache': False,  # 不緩存到內存，節省內存
            'workers': 4,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        self.logger.info("訓練參數:")
        for key, value in self.training_params.items():
            self.logger.info(f"  {key}: {value}")
            
    def load_model(self):
        """加載預訓練模型"""
        self.logger.info("🤖 加載YOLOv11s預訓練模型...")
        
        try:
            self.model = YOLO('yolo11s.pt')
            self.logger.info("✅ 模型加載成功")
            
            # 顯示模型信息
            self.logger.info(f"✅ 模型參數量: {sum(p.numel() for p in self.model.model.parameters()):,}")
            
        except Exception as e:
            self.logger.error(f"❌ 模型加載失敗: {e}")
            sys.exit(1)
            
    def train(self):
        """開始訓練"""
        self.logger.info("🚀 開始訓練...")
        self.start_time = time.time()
        
        try:
            # 開始訓練
            self.results = self.model.train(**self.training_params)
            
            # 計算訓練時間
            training_time = time.time() - self.start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            
            self.logger.info("🎉 訓練完成!")
            self.logger.info(f"⏰ 總訓練時間: {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.logger.info(f"📁 結果保存目錄: {self.results.save_dir}")
            
        except KeyboardInterrupt:
            self.logger.warning("⚠️  訓練被用戶中斷")
            sys.exit(0)
        except Exception as e:
            self.logger.error(f"❌ 訓練過程中出錯: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    def validate_model(self):
        """驗證最佳模型"""
        self.logger.info("📊 驗證最佳模型...")
        
        try:
            best_model_path = self.results.save_dir / 'weights' / 'best.pt'
            if best_model_path.exists():
                model = YOLO(str(best_model_path))
                val_results = model.val(data=self.config_path)
                
                self.logger.info("驗證結果:")
                self.logger.info(f"  mAP50: {val_results.box.map50:.4f}")
                self.logger.info(f"  mAP50-95: {val_results.box.map:.4f}")
                
                # 各類別AP
                for i, class_name in self.config['names'].items():
                    if i < len(val_results.box.ap_class_index):
                        ap = val_results.box.ap[i].mean()
                        self.logger.info(f"  {class_name} AP: {ap:.4f}")
                        
        except Exception as e:
            self.logger.error(f"❌ 驗證過程中出錯: {e}")
            
    def generate_report(self):
        """生成訓練報告"""
        self.logger.info("📋 生成訓練報告...")
        
        try:
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = report_dir / f"training_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("YOLOv11s 海洋垃圾檢測訓練報告\n")
                f.write("="*50 + "\n\n")
                f.write(f"訓練時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"數據集: Trash_ICRA19\n")
                f.write(f"模型: YOLOv11s\n")
                f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
                f.write(f"結果目錄: {self.results.save_dir}\n\n")
                
                f.write("訓練參數:\n")
                for key, value in self.training_params.items():
                    f.write(f"  {key}: {value}\n")
                    
            self.logger.info(f"✅ 報告已保存: {report_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 生成報告時出錯: {e}")
            
    def run(self):
        """運行完整的訓練流程"""
        try:
            self.setup_training_params()
            self.load_model()
            self.train()
            self.validate_model()
            self.generate_report()
            
            self.logger.info("="*80)
            self.logger.info("🎊 訓練流程全部完成!")
            self.logger.info(f"📁 查看結果: {self.results.save_dir}")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ 訓練流程失敗: {e}")
            sys.exit(1)

def main():
    """主函數"""
    print("🌊 YOLOv11s 海洋垃圾檢測訓練系統")
    print("作者: Leo Yu | GPU: RTX 3070")
    print("-" * 50)
    
    # 檢查配置文件
    config_files = ["trash_dataset_simple.yaml", "trash_dataset.yaml"]
    config_file = None
    
    for file in config_files:
        if os.path.exists(file):
            config_file = file
            break
            
    if not config_file:
        print("❌ 錯誤: 找不到數據集配置文件")
        print("請確保存在以下任一文件:")
        for file in config_files:
            print(f"  - {file}")
        sys.exit(1)
        
    # 創建訓練器並運行
    trainer = TrashDetectionTrainer(config_file)
    trainer.run()

if __name__ == "__main__":
    main()