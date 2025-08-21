#!/usr/bin/env python3
"""
eGPU 測試代碼
測試GPU是否可用，並進行簡單的性能測試
"""

import time
import sys

def test_gpu_basic():
    """基本GPU檢測"""
    print("=== 基本GPU檢測 ===")
    
    # 檢測NVIDIA GPU
    try:
        import nvidia_ml_py3 as nvml
        nvml.nvmlInit()
        device_count = nvml.nvmlDeviceGetCount()
        print(f"找到 {device_count} 個NVIDIA GPU:")
        
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"  GPU {i}: {name}")
            print(f"    記憶體: {memory_info.total // 1024**2} MB")
            print(f"    可用: {memory_info.free // 1024**2} MB")
    except:
        print("未找到NVIDIA GPU或nvidia-ml-py3未安裝")

def test_pytorch_gpu():
    """測試PyTorch GPU支持"""
    print("\n=== PyTorch GPU 測試 ===")
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU數量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory
                print(f"  GPU {i}: {gpu_name}")
                print(f"    記憶體: {gpu_memory // 1024**2} MB")
                
            # 簡單的GPU計算測試
            print("\n進行GPU計算測試...")
            device = torch.device('cuda')
            
            # 創建大矩陣進行乘法運算
            size = 5000
            print(f"測試 {size}x{size} 矩陣乘法...")
            
            # CPU測試
            start_time = time.time()
            a_cpu = torch.randn(size, size)
            b_cpu = torch.randn(size, size)
            c_cpu = torch.mm(a_cpu, b_cpu)
            cpu_time = time.time() - start_time
            print(f"CPU時間: {cpu_time:.2f} 秒")
            
            # GPU測試
            start_time = time.time()
            a_gpu = torch.randn(size, size, device=device)
            b_gpu = torch.randn(size, size, device=device)
            torch.cuda.synchronize()  # 確保GPU計算完成
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            print(f"GPU時間: {gpu_time:.2f} 秒")
            print(f"GPU加速倍數: {cpu_time/gpu_time:.1f}x")
            
        else:
            print("CUDA不可用，請檢查GPU驅動和PyTorch安裝")
            
    except ImportError:
        print("PyTorch未安裝，請執行: pip install torch")

def test_tensorflow_gpu():
    """測試TensorFlow GPU支持"""
    print("\n=== TensorFlow GPU 測試 ===")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow版本: {tf.__version__}")
        
        # 檢測GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"找到 {len(gpus)} 個GPU:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
        
        if gpus:
            # 簡單的TensorFlow GPU測試
            print("\n進行TensorFlow GPU計算測試...")
            
            with tf.device('/CPU:0'):
                start_time = time.time()
                a_cpu = tf.random.normal([3000, 3000])
                b_cpu = tf.random.normal([3000, 3000])
                c_cpu = tf.matmul(a_cpu, b_cpu)
                cpu_time = time.time() - start_time
                print(f"CPU時間: {cpu_time:.2f} 秒")
            
            with tf.device('/GPU:0'):
                start_time = time.time()
                a_gpu = tf.random.normal([3000, 3000])
                b_gpu = tf.random.normal([3000, 3000])
                c_gpu = tf.matmul(a_gpu, b_gpu)
                gpu_time = time.time() - start_time
                print(f"GPU時間: {gpu_time:.2f} 秒")
                print(f"GPU加速倍數: {cpu_time/gpu_time:.1f}x")
        else:
            print("未檢測到GPU")
            
    except ImportError:
        print("TensorFlow未安裝，請執行: pip install tensorflow")

def test_memory_bandwidth():
    """測試GPU記憶體頻寬"""
    print("\n=== GPU 記憶體頻寬測試 ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # 測試不同大小的數據傳輸
            sizes = [100, 500, 1000, 2000]
            
            for size in sizes:
                print(f"\n測試 {size}x{size} 矩陣:")
                
                # CPU到GPU傳輸
                data_cpu = torch.randn(size, size)
                start_time = time.time()
                data_gpu = data_cpu.to(device)
                torch.cuda.synchronize()
                transfer_time = time.time() - start_time
                data_size_mb = (size * size * 4) / (1024 * 1024)  # float32 = 4 bytes
                bandwidth = data_size_mb / transfer_time
                print(f"  CPU->GPU: {transfer_time*1000:.1f}ms, 頻寬: {bandwidth:.1f} MB/s")
                
                # GPU到CPU傳輸
                start_time = time.time()
                data_back = data_gpu.cpu()
                transfer_time = time.time() - start_time
                bandwidth = data_size_mb / transfer_time
                print(f"  GPU->CPU: {transfer_time*1000:.1f}ms, 頻寬: {bandwidth:.1f} MB/s")
                
    except ImportError:
        print("需要PyTorch來進行記憶體頻寬測試")

def main():
    print("🚀 eGPU 測試開始")
    print("=" * 50)
    
    # 基本檢測
    test_gpu_basic()
    
    # PyTorch測試
    test_pytorch_gpu()
    
    # TensorFlow測試
    test_tensorflow_gpu()
    
    # 記憶體頻寬測試
    test_memory_bandwidth()
    
    print("\n" + "=" * 50)
    print("✅ 測試完成！")
    
    # 給出建議
    print("\n💡 建議:")
    print("- 如果GPU可用且加速倍數 > 5x，eGPU工作正常")
    print("- 如果加速倍數 < 3x，可能是Thunderbolt頻寬限制")
    print("- 如果完全沒有GPU檢測到，檢查驅動程式安裝")

if __name__ == "__main__":
    main()