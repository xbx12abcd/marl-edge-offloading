#!/usr/bin/env python3
"""
GPU显存诊断和优化建议工具
用于快速检查RTX 4060 Ti的显存状态和优化建议
"""

import torch
import psutil
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.gpu_monitor import GPUMemoryMonitor, MemoryOptimizer, print_memory_info


def diagnose_gpu():
    """诊断GPU显存状态"""
    print("=" * 70)
    print("GPU显存诊断工具 - RTX 4060 Ti优化版")
    print("=" * 70)
    print()
    
    # 检查CUDA可用性
    print("1️⃣  CUDA 基本检查")
    print("-" * 70)
    if not torch.cuda.is_available():
        print("❌ CUDA不可用！请检查：")
        print("   - NVIDIA驱动是否已安装")
        print("   - PyTorch是否使用CUDA构建")
        return False
    
    print(f"✓ CUDA 版本: {torch.version.cuda}")
    print(f"✓ GPU数量: {torch.cuda.device_count()}")
    print(f"✓ 当前GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 显存信息
    print("2️⃣  显存详细信息")
    print("-" * 70)
    print_memory_info()
    print()
    
    # 显存占用分析
    print("3️⃣  显存占用分析")
    print("-" * 70)
    try:
        monitor = GPUMemoryMonitor()
        mem_info = monitor.get_memory_usage()
        
        allocated = mem_info['allocated_gb']
        reserved = mem_info['reserved_gb']
        percent = mem_info['utilization_percent']
        
        print(f"已分配: {allocated:.2f} GB")
        print(f"已保留: {reserved:.2f} GB")
        print(f"使用率: {percent:.1f}%")
        
        # 健康状态
        print()
        if percent > 90:
            print("⚠️  警告: 显存使用率过高 (>90%)")
            print("   → 建议降低batch_size或hidden_dim")
        elif percent > 75:
            print("⚠️  注意: 显存使用率较高 (>75%)")
            print("   → 建议启用混合精度或梯度累积")
        elif percent > 60:
            print("✓ 显存使用率正常 (60-75%)")
        else:
            print("✓ 显存使用率良好 (<60%)")
        print()
        
    except Exception as e:
        print(f"❌ 无法获取显存信息: {e}")
    
    # CPU和内存
    print("4️⃣  系统内存检查")
    print("-" * 70)
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"CPU使用率: {cpu_percent}%")
    print(f"系统内存: {memory.used / 1e9:.2f} GB / {memory.total / 1e9:.2f} GB")
    print(f"内存使用率: {memory.percent}%")
    print()
    
    if memory.percent > 85:
        print("⚠️  警告: 系统内存占用过高")
        print("   → 可能影响DataLoader的并行读取")
    print()
    
    # 推荐配置
    print("5️⃣  推荐优化方案")
    print("-" * 70)
    
    optimizer = MemoryOptimizer()
    device_info = optimizer.get_device_info()
    
    # 自动推荐
    if device_info.get('device_name', '').find('4060 Ti') >= 0:
        print("检测到 RTX 4060 Ti")
        print()
        print("推荐方案 1 (快速实验):")
        print("  config: lowmem_config.yaml")
        print("  command: python train_ippo_lowmem.py --episodes 100")
        print("  预期显存: 2-3 GB, 预期时间: 10-20 分钟")
        print()
        
        print("推荐方案 2 (完整训练):")
        print("  config: lowmem_config.yaml")
        print("  command: python train_ippo_lowmem.py --episodes 500")
        print("  预期显存: 4-5 GB, 预期时间: 2-3 小时")
        print()
        
        print("推荐方案 3 (所有优化):")
        print("  编辑 configs/lowmem_config.yaml:")
        print("    - hidden_dim: 16 (instead of 32)")
        print("    - batch_size: 4 (instead of 8)")
        print("    - num_edges: 1 (instead of 2)")
        print("  预期显存: 1-2 GB, 预期时间: 1-2 小时")
    print()
    
    # 性能建议
    print("6️⃣  性能优化建议")
    print("-" * 70)
    
    suggestions = []
    
    if percent > 80:
        suggestions.append("• 显存占用 > 80%")
        suggestions.append("  → 启用混合精度: use_mixed_precision=true")
        suggestions.append("  → 使用梯度累积: gradient_accumulation_steps=4")
    
    if cpu_percent > 80:
        suggestions.append("• CPU利用率 > 80%")
        suggestions.append("  → 减少数据加载线程")
        suggestions.append("  → 增加model_batch_size")
    
    if memory.percent > 80:
        suggestions.append("• 系统内存 > 80%")
        suggestions.append("  → 关闭其他应用")
        suggestions.append("  → 减少replay_buffer_size")
    
    if not suggestions:
        suggestions.append("✓ 当前配置与硬件匹配良好")
        suggestions.append("✓ 可以安全开始训练")
    
    for s in suggestions:
        print(s)
    print()
    
    # 故障排查
    print("7️⃣  常见问题和解决方案")
    print("-" * 70)
    print("""
问题1: 训练中显示 'CUDA out of memory'
  • 立即降低 batch_size (从8→4→2)
  • 启用混合精度训练
  • 减少网络隐藏层大小

问题2: 训练不收敛 (loss不下降)
  • 增加梯度累积步数
  • 降低学习率
  • 增加entropy_coeff (熵系数)

问题3: 训练速度很慢 (<10 episodes/min)
  • 检查CPU是否成为瓶颈
  • 启用混合精度 (应该加速 20-50%)
  • 检查磁盘IO (checkpoint保存)

问题4: 显存使用不断增加 (内存泄漏)
  • 运行: python test_env.py 验证环境
  • 检查是否有tensor被误保留
  • 定期调用 torch.cuda.empty_cache()
""")
    print()
    
    # 快速测试
    print("8️⃣  运行快速测试")
    print("-" * 70)
    print("""
建议按以下顺序测试:

1. 环境验证 (2分钟):
   python test_env.py

2. 快速训练 (5分钟):
   python quick_start.py --episodes 10

3. 完整训练 (30分钟):
   python train_ippo_lowmem.py --episodes 100

4. 长期训练 (2-3小时):
   python train_ippo_lowmem.py --episodes 500
""")
    print()
    
    return True


def detailed_memory_analysis():
    """详细的显存分析"""
    print("=" * 70)
    print("详细显存分析")
    print("=" * 70)
    print()
    
    optimizer = MemoryOptimizer()
    
    # 不同配置的显存占用预测
    configs = [
        {"hidden_dim": 16, "batch_size": 2, "name": "超小型"},
        {"hidden_dim": 32, "batch_size": 4, "name": "小型"},
        {"hidden_dim": 32, "batch_size": 8, "name": "推荐"},
        {"hidden_dim": 64, "batch_size": 16, "name": "中型"},
    ]
    
    print("配置 | 隐藏维度 | 批次大小 | 估计显存 (FP32) | 估计显存 (FP16)")
    print("-" * 70)
    
    for cfg in configs:
        hidden_dim = cfg['hidden_dim']
        batch_size = cfg['batch_size']
        name = cfg['name']
        
        # 粗略估算
        # 模型参数大约: state_dim(32) * hidden_dim * 4 * 2 (input+output layers)
        # + batch_size * hidden_dim * 4 (激活值)
        model_params_bytes = 32 * hidden_dim * 4 * 2
        activations_bytes = batch_size * hidden_dim * 4 * 2
        total_bytes_fp32 = model_params_bytes + activations_bytes
        total_bytes_fp16 = total_bytes_fp32 / 2
        
        # 转换为GB (3个agent的估计)
        total_gb_fp32 = total_bytes_fp32 * 3 / 1e9
        total_gb_fp16 = total_bytes_fp16 * 3 / 1e9
        
        marker = "✓ 推荐" if name == "推荐" else ""
        print(f"{name:6} | {hidden_dim:8} | {batch_size:8} | {total_gb_fp32:12.2f} GB | {total_gb_fp16:11.2f} GB {marker}")
    
    print()
    print("注: 估算值为3个agent的总占用，实际值取决于多个因素")
    print()


def test_mixed_precision():
    """测试混合精度是否可用"""
    print("=" * 70)
    print("混合精度测试")
    print("=" * 70)
    print()
    
    try:
        import torch.cuda.amp as amp
        
        # 创建简单模型
        model = torch.nn.Linear(32, 32).cuda()
        input_data = torch.randn(8, 32).cuda()
        
        # 测试FP16
        print("测试 FP16 混合精度... ", end="")
        with amp.autocast(dtype=torch.float16):
            output = model(input_data)
        print("✓ 成功")
        
        # 测试GradScaler
        print("测试 GradScaler... ", end="")
        scaler = amp.GradScaler()
        with amp.autocast(dtype=torch.float16):
            loss = output.sum()
        scaler.scale(loss).backward()
        scaler.step(model.parameters())
        scaler.update()
        print("✓ 成功")
        
        print()
        print("✓ 混合精度训练可用")
        print("  建议: 在config中启用 use_mixed_precision=true")
        print()
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        print("  可能原因: CUDA版本不支持")
        print()


def main():
    """主函数"""
    # 基本诊断
    if not diagnose_gpu():
        sys.exit(1)
    
    print()
    
    # 详细分析
    detailed_memory_analysis()
    
    print()
    
    # 混合精度测试
    test_mixed_precision()
    
    print("=" * 70)
    print("诊断完成 ✓")
    print("=" * 70)
    print()
    print("下一步:")
    print("1. 根据上面的推荐选择合适的配置")
    print("2. 运行: python test_env.py")
    print("3. 运行: python train_ippo_lowmem.py --episodes 100")
    print()


if __name__ == "__main__":
    main()
