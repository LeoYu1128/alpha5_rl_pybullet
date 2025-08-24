#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced YOLOv11s Trash Detection Training Script
Author: Leo Yu
Project: Marine Trash Detection - Trash_ICRA19 Dataset
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
       """Initialize trainer"""
       self.config_path = config_path
       self.start_time = None
       self.model = None
       self.results = None
       
       # Setup logging
       self.setup_logging()
       
       # Check environment
       self.check_environment()
       
       # Load configuration
       self.load_config()
       
   def setup_logging(self):
       """Setup logging configuration"""
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
       self.logger.info("YOLOv11s Marine Trash Detection Training Started")
       self.logger.info("="*80)
       
   def check_environment(self):
       """Check training environment"""
       self.logger.info("🔍 Checking training environment...")
       
       # Check GPU
       if torch.cuda.is_available():
           gpu_name = torch.cuda.get_device_name(0)
           gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
           self.logger.info(f"✅ GPU: {gpu_name}")
           self.logger.info(f"✅ GPU Memory: {gpu_memory:.1f} GB")
           self.logger.info(f"✅ CUDA Version: {torch.version.cuda}")
           self.device = 'cuda'
       else:
           self.logger.warning("⚠️  No GPU detected, will use CPU for training")
           self.device = 'cpu'
           
       # Check Python and PyTorch versions
       self.logger.info(f"✅ Python Version: {sys.version.split()[0]}")
       self.logger.info(f"✅ PyTorch Version: {torch.__version__}")
       
   def load_config(self):
       """Load dataset configuration"""
       self.logger.info("📋 Loading dataset configuration...")
       
       if not os.path.exists(self.config_path):
           self.logger.error(f"❌ Configuration file does not exist: {self.config_path}")
           sys.exit(1)
           
       with open(self.config_path, 'r', encoding='utf-8') as f:
           self.config = yaml.safe_load(f)
           
       self.logger.info(f"✅ Dataset Path: {self.config['path']}")
       self.logger.info(f"✅ Number of Classes: {self.config['nc']}")
       self.logger.info(f"✅ Class Names: {list(self.config['names'].values())}")
       
       # Validate dataset path
       if not os.path.exists(self.config['path']):
           self.logger.error(f"❌ Dataset path does not exist: {self.config['path']}")
           sys.exit(1)
           
   def setup_training_params(self):
       """Setup training parameters"""
       self.logger.info("⚙️  Setting up training parameters...")
       
       # Adjust batch size based on GPU
       if self.device == 'cuda':
           gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
           if gpu_memory < 6:
               batch_size = 8
               self.logger.warning("⚠️  Low GPU memory, adjusting batch size to 8")
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
           'cache': False,  # Don't cache to memory to save memory
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
       
       self.logger.info("Training Parameters:")
       for key, value in self.training_params.items():
           self.logger.info(f"  {key}: {value}")
           
   def load_model(self):
       """Load pretrained model"""
       self.logger.info("🤖 Loading YOLOv11s pretrained model...")
       
       try:
           self.model = YOLO('yolo11s.pt')
           self.logger.info("✅ Model loaded successfully")
           
           # Display model information
           self.logger.info(f"✅ Model Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
           
       except Exception as e:
           self.logger.error(f"❌ Model loading failed: {e}")
           sys.exit(1)
           
   def train(self):
       """Start training"""
       self.logger.info("🚀 Starting training...")
       self.start_time = time.time()
       
       try:
           # Start training
           self.results = self.model.train(**self.training_params)
           
           # Calculate training time
           training_time = time.time() - self.start_time
           hours = int(training_time // 3600)
           minutes = int((training_time % 3600) // 60)
           seconds = int(training_time % 60)
           
           self.logger.info("🎉 Training completed!")
           self.logger.info(f"⏰ Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
           self.logger.info(f"📁 Results saved to: {self.results.save_dir}")
           
       except KeyboardInterrupt:
           self.logger.warning("⚠️  Training interrupted by user")
           sys.exit(0)
       except Exception as e:
           self.logger.error(f"❌ Error during training: {e}")
           import traceback
           traceback.print_exc()
           sys.exit(1)
           
   def validate_model(self):
       """Validate the best model"""
       self.logger.info("📊 Validating best model...")
       
       try:
           best_model_path = self.results.save_dir / 'weights' / 'best.pt'
           if best_model_path.exists():
               model = YOLO(str(best_model_path))
               val_results = model.val(data=self.config_path)
               
               self.logger.info("Validation Results:")
               self.logger.info(f"  mAP50: {val_results.box.map50:.4f}")
               self.logger.info(f"  mAP50-95: {val_results.box.map:.4f}")
               
               # Per-class AP
               for i, class_name in self.config['names'].items():
                   if i < len(val_results.box.ap_class_index):
                       ap = val_results.box.ap[i].mean()
                       self.logger.info(f"  {class_name} AP: {ap:.4f}")
                       
       except Exception as e:
           self.logger.error(f"❌ Error during validation: {e}")
           
   def generate_report(self):
       """Generate training report"""
       self.logger.info("📋 Generating training report...")
       
       try:
           report_dir = Path("reports")
           report_dir.mkdir(exist_ok=True)
           
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           report_file = report_dir / f"training_report_{timestamp}.txt"
           
           with open(report_file, 'w', encoding='utf-8') as f:
               f.write("YOLOv11s Marine Trash Detection Training Report\n")
               f.write("="*50 + "\n\n")
               f.write(f"Training Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
               f.write(f"Dataset: Trash_ICRA19\n")
               f.write(f"Model: YOLOv11s\n")
               f.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
               f.write(f"Results Directory: {self.results.save_dir}\n\n")
               
               f.write("Training Parameters:\n")
               for key, value in self.training_params.items():
                   f.write(f"  {key}: {value}\n")
                   
           self.logger.info(f"✅ Report saved to: {report_file}")
           
       except Exception as e:
           self.logger.error(f"❌ Error generating report: {e}")
           
   def run(self):
       """Run complete training pipeline"""
       try:
           self.setup_training_params()
           self.load_model()
           self.train()
           self.validate_model()
           self.generate_report()
           
           self.logger.info("="*80)
           self.logger.info("🎊 Training pipeline completed successfully!")
           self.logger.info(f"📁 Check results at: {self.results.save_dir}")
           self.logger.info("="*80)
           
       except Exception as e:
           self.logger.error(f"❌ Training pipeline failed: {e}")
           sys.exit(1)

def main():
   """Main function"""
   print("🌊 YOLOv11s Marine Trash Detection Training System")
   print("Author: Leo Yu | GPU: RTX 3070")
   print("-" * 50)
   
   # Check configuration files
   config_files = ["trash_dataset_simple.yaml", "trash_dataset.yaml"]
   config_file = None
   
   for file in config_files:
       if os.path.exists(file):
           config_file = file
           break
           
   if not config_file:
       print("❌ Error: Cannot find dataset configuration file")
       print("Please ensure one of the following files exists:")
       for file in config_files:
           print(f"  - {file}")
       sys.exit(1)
       
   # Create trainer and run
   trainer = TrashDetectionTrainer(config_file)
   trainer.run()

if __name__ == "__main__":
   main()