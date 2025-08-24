#!/usr/bin/env python3
"""
Real-time marine trash detection system
Supports camera, IP camera, video files
"""
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from collections import deque

class RealTimeTrashDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        """Initialize real-time detector"""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        
        # Class configuration
        self.class_names = {
            0: 'trash', 
            1: 'biological', 
            2: 'rov'
        }
        
        self.class_colors = {
            0: (0, 255, 0),      # trash - green
            1: (255, 165, 0),    # biological - orange
            2: (0, 0, 255),      # rov - red
        }
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.detection_counts = {name: 0 for name in self.class_names.values()}
        
    def load_model(self):
        """Load model"""
        if not Path(self.model_path).exists():
            print(f"❌ Model file does not exist: {self.model_path}")
            return False
            
        print(f"🤖 Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("✅ Model loaded successfully")
        return True
        
    def process_frame(self, frame):
        """Process single frame"""
        start_time = time.time()
        
        # YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        # Calculate FPS
        inference_time = (time.time() - start_time) * 1000
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        
        # Draw detection results
        annotated_frame = self.draw_detections(frame, results[0])
        
        # Add performance info
        annotated_frame = self.draw_info(annotated_frame, inference_time)
        
        return annotated_frame
        
    def draw_detections(self, frame, result):
        """Draw detection results"""
        frame_copy = frame.copy()
        
        if result.boxes is not None:
            for box in result.boxes:
                # Extract information
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.class_names.get(cls, f'class_{cls}')
                
                # Update statistics
                self.detection_counts[class_name] += 1
                
                # Get color
                color = self.class_colors.get(cls, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Prepare label
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(frame_copy, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            color, -1)
                
                # Draw label text
                cv2.putText(frame_copy, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        return frame_copy
        
    def draw_info(self, frame, inference_time):
        """Draw performance and statistics information"""
        h, w = frame.shape[:2]
        
        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        # Info panel background
        info_height = 120
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw information
        y_offset = 30
        line_height = 20
        
        # FPS info
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Inference time
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += line_height
        
        # Detection statistics
        cv2.putText(frame, "Detections:", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        for class_name, count in self.detection_counts.items():
            color = self.class_colors.get(
                next(k for k, v in self.class_names.items() if v == class_name), 
                (255, 255, 255)
            )
            cv2.putText(frame, f"  {class_name}: {count}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += line_height
        
        return frame
        
    def run_camera(self, camera_id=0):
        """Run camera detection"""
        print(f"📹 Starting camera {camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
            
        # Set camera parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🚀 Real-time detection started (press 'q' to exit, 'r' to reset statistics, 's' to screenshot)")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read camera frame")
                break
                
            frame_count += 1
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('🌊 Marine Trash Real-time Detection', processed_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset statistics
                self.detection_counts = {name: 0 for name in self.class_names.values()}
                print("📊 Statistics reset")
            elif key == ord('s'):
                # Screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"trash_detection_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"📸 Screenshot saved: {filename}")
                
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 Detection statistics:")
        for class_name, count in self.detection_counts.items():
            print(f"  {class_name}: {count} detections")
        
    def run_ip_camera(self, ip_url):
        """Run IP camera detection"""
        print(f"📡 Connecting to IP camera: {ip_url}")
        
        cap = cv2.VideoCapture(ip_url)
        if not cap.isOpened():
            print(f"❌ Cannot connect to IP camera: {ip_url}")
            return
            
        print("🚀 IP camera detection started (press 'q' to exit)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Connection lost, attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(ip_url)
                continue
                
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('🌊 Marine Trash Detection (IP Camera)', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def find_latest_model():
    """Automatically find the latest model"""
    trash_detection_dir = Path("trash_detection")
    
    if not trash_detection_dir.exists():
        return None
        
    run_dirs = [d for d in trash_detection_dir.iterdir() 
               if d.is_dir() and d.name.startswith('yolov11s_')]
    
    if not run_dirs:
        return None
        
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_run / "weights" / "best.pt"
    
    return str(model_path) if model_path.exists() else None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-time marine trash detection')
    parser.add_argument('--model', type=str, help='Model path')
    parser.add_argument('--source', type=str, default='0', help='Input source (0=camera, IP address, or video file)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("🌊 YOLOv11s Real-time Marine Trash Detection System")
    print("="*50)
    
    # Get model path
    model_path = args.model or find_latest_model()
    
    if not model_path:
        print("❌ No model file found")
        return
        
    print(f"🤖 Using model: {model_path}")
    print(f"🎯 Confidence threshold: {args.conf}")
    
    # Create detector
    detector = RealTimeTrashDetector(model_path, args.conf)
    
    if not detector.load_model():
        return
    
    # Parse input source
    source = args.source
    
    if source.isdigit():
        # Camera ID
        detector.run_camera(int(source))
    elif source.startswith(('rtsp://', 'http://', 'https://')):
        # IP camera
        detector.run_ip_camera(source)
    else:
        print(f"❌ Unsupported input source: {source}")
        print("Supported formats: 0 (camera ID), rtsp://..., http://...")

if __name__ == "__main__":
    main()