from ultralytics import YOLO
from pathlib import Path

def test_bottle():
    """Test bottle detection"""
    print("🌊 YOLOv11s Marine Trash Detection - Image Test")
    print("="*50)
    
    # Automatically find the latest model (same logic as other files)
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
    
    # Load your trained model
    model = YOLO(model_path)
    
    # Test bottle.jpg
    results = model('bottle.jpg')
    
    # Show results
    results[0].show()
    
    # Save results
    results[0].save('bottle_detection_result.jpg')
    
    # Print detection results
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            print(f"\n📊 Detected {len(boxes)} objects:")
            for i, box in enumerate(boxes):
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                class_name = model.names[cls]  # Get class name
                print(f"  {i+1}. {class_name}: {conf:.2f}")
        else:
            print("\n📊 No objects detected")

if __name__ == "__main__":
    test_bottle()