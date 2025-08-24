#!/usr/bin/env python3
"""
Test video using trained YOLOv11s model
"""
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
import time

class VideoTester:
    def __init__(self, model_path, video_path):
        """Initialize video tester"""
        self.model_path = model_path
        self.video_path = video_path
        self.model = None
        self.class_names = {0: 'trash', 1: 'biological', 2: 'rov'}
        
    def load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file does not exist: {self.model_path}")
            return False
            
        print(f"🤖 Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("✅ Model loaded successfully")
        return True
        
    def test_video_simple(self, output_path="output_video.mp4", conf_threshold=0.5):
        """Simple video test - using YOLO built-in functionality"""
        if not self.load_model():
            return
            
        print(f"🎬 Processing video: {self.video_path}")
        print(f"📊 Confidence threshold: {conf_threshold}")
        
        # Use YOLO's predict function to process video
        results = self.model.predict(
            source=self.video_path,
            conf=conf_threshold,
            iou=0.7,
            save=True,
            project="video_results",
            name="trash_detection_video"
        )
        
        print("✅ Video processing completed")
        print(f"📁 Results saved to: video_results/trash_detection_video/")
        
    def test_video_custom(self, output_path="custom_output.mp4", conf_threshold=0.3):
        """Custom video test - more control"""
        if not self.load_model():
            return
            
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {self.video_path}")
            return
            
        # Get video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height}, {fps}FPS, {total_frames} frames")
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        # Statistics
        detection_stats = {'trash': 0, 'biological': 0, 'rov': 0}
        
        print("🚀 Processing started...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # YOLO detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Process detection results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates and info
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = self.class_names.get(cls, f'class_{cls}')
                        
                        # Statistics
                        detection_stats[class_name] += 1
                        
                        # Draw bounding box
                        color = self.get_color(cls)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Progress display
            if frame_count % 30 == 0:  # Show every 30 frames
                progress = frame_count / total_frames * 100
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) | FPS: {fps_current:.1f}")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Show statistics
        elapsed_time = time.time() - start_time
        print("\n" + "="*50)
        print("🎉 Video processing completed!")
        print(f"⏱️  Processing time: {elapsed_time:.2f} seconds")
        print(f"🎬 Processed frames: {frame_count}")
        print(f"⚡ Average FPS: {frame_count/elapsed_time:.2f}")
        print(f"💾 Output file: {output_path}")
        print("\n📊 Detection statistics:")
        for class_name, count in detection_stats.items():
            print(f"  {class_name}: {count} detections")
        print("="*50)
        
    def get_color(self, class_id):
        """Return color based on class ID"""
        colors = {
            0: (0, 255, 0),    # trash - green
            1: (255, 0, 0),    # biological - blue  
            2: (0, 0, 255),    # rov - red
        }
        return colors.get(class_id, (255, 255, 255))
        
    def test_realtime(self, conf_threshold=0.5):
        """Real-time display test"""
        if not self.load_model():
            return
            
        cap = cv2.VideoCapture(self.video_path)
        
        print("🎬 Real-time playback (press 'q' to exit)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # YOLO detection
            results = self.model(frame, conf=conf_threshold, verbose=False)
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Display
            cv2.imshow('Trash Detection', annotated_frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("🌊 YOLOv11s Marine Trash Detection - Video Test")
    print("="*50)
    
    # Automatically find the latest model
    model_path = None
    trash_detection_dir = Path("trash_detection")
    
    if trash_detection_dir.exists():
        run_dirs = [d for d in trash_detection_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('yolov11s_')]
        if run_dirs:
            latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
            model_path = latest_run / "weights" / "best.pt"
    
    if not model_path or not model_path.exists():
        print("❌ No trained model found")
        print("Please ensure training is completed or manually specify model path")
        return
    
    print(f"🤖 Using model: {model_path}")
    
    # Ask user for video path
    video_path = input("📹 Please enter video file path: ").strip().strip('"')
    
    if not os.path.exists(video_path):
        print(f"❌ Video file does not exist: {video_path}")
        return
    
    # Create tester
    tester = VideoTester(str(model_path), video_path)
    
    # Select test mode
    print("\nSelect test mode:")
    print("1. Simple test (using YOLO built-in functionality)")
    print("2. Custom test (more control and statistics)")
    print("3. Real-time playback test")
    
    choice = input("Please enter choice (1-3): ").strip()
    
    if choice == "1":
        tester.test_video_simple()
    elif choice == "2":
        output_path = input("Output filename (default: custom_output.mp4): ").strip() or "custom_output.mp4"
        tester.test_video_custom(output_path)
    elif choice == "3":
        tester.test_realtime()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()