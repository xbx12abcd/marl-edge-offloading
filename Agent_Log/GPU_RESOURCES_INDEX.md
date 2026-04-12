# GPU优化资源索引

**快速导航** - 根据你的需要找到相关资源 ⚡

---

## 🎯 我想快速开始

### 如果你是第一次使用

1. **诊断硬件** (3 minutes)
   ```bash
   python diagnose_gpu.py
   ```
   → 自动检测硬件，打印优化建议

2. **快速测试** (5 minutes)
   ```bash
   python train_ippo_lowmem.py --episodes 10
   ```
   → 验证环境可用，快速反馈

3. **完整训练** (2-3 hours)
   ```bash
   python train_ippo_lowmem.py --episodes 500
   ```
   → 标准配置的完整训练

**相关文件**:
- 📖 [README.md](README.md) - 项目概述和快速开始
- ⚡ [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) - 一页纸速查表
- 🔧 [diagnose_gpu.py](diagnose_gpu.py) - 诊断工具

---

## 📚 我想深入了解GPU优化

### 优化技术和原理

1. **混合精度训练** (FP16) - 节省50%显存
   → [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) § 显存优化技术

2. **梯度累积** - 相同效果更低显存
   → [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) § 梯度累积

3. **网络缩小** - 参数数量减少75%
   → [configs/lowmem_config.yaml](configs/lowmem_config.yaml) - 参数对比

4. **显存管理** - 实时监控和清理
   → [utils/gpu_monitor.py](utils/gpu_monitor.py) - 监控工具

**推荐阅读顺序**:
1. [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) (15 min)
2. [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) (60 min)
3. [GPU_OPTIMIZATION_REPORT.md](GPU_OPTIMIZATION_REPORT.md) (30 min)

---

## ⚙️ 我想调整配置参数

### 配置文件说明

| 文件 | 场景 | 相对速度 | 相对质量 |
|------|------|---------|---------|
| [default_config.yaml](configs/default_config.yaml) | 高端GPU (8GB+) | 1.0x | 最优 |
| [lowmem_config.yaml](configs/lowmem_config.yaml) | 推荐 (4060 Ti) | 1.5x | 优秀 |
| 自定义 | 特殊需求 | 可变 | 可变 |

### 参数调整指南

**想要更快的训练？**
```yaml
# configs/my_config.yaml
hidden_dim: 16        # 更小的网络
batch_size: 16        # 更大的批次 (如果显存允许)
num_epochs: 1         # 少一些epoch
use_mixed_precision: true  # 启用FP16
```

**想要更好的效果？**
```yaml
# configs/my_config.yaml
hidden_dim: 64        # 更大的网络
batch_size: 8         # 合理的批次
num_epochs: 4         # 多一些epoch
total_episodes: 1000  # 更多episodes
```

**显存不足怎么办？**
```yaml
# configs/my_config.yaml
batch_size: 4         # 减半
hidden_dim: 16        # 减小network
gradient_accumulation_steps: 8  # 增加累积步数
use_mixed_precision: true       # 务必启用
```

→ 详细参数说明: [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) § 核心参数说明

---

## 🔧 我遇到了问题

### 常见的问题和解决方案

| 问题 | 快速修复 | 详细指南 |
|------|---------|---------|
| CUDA out of memory | 降低batch_size半, 启用FP16 | [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) § 快速修复 |
| Loss为NaN | 降低learning_rate 10倍 | [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) § 故障排除 |
| 训练太慢 | 使用mixed_precision=true | [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) § 常见问题 |
| 显存继续增长 | 运行memory_leak_detector.py | [tools/memory_leak_detector.py](tools/memory_leak_detector.py) |

### 故障排除电话树

```
问题: CUDA out of memory
├─ 快速修复: batch_size: 8 → 4 → 2
├─ 中期方案: 启用 use_mixed_precision: true
├─ 长期优化: 减少 hidden_dim 或编辑 num_tasks
└─ 查看详情: [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md)

问题: Loss为NaN
├─ 检查: learning_rate (降低10倍试试)
├─ 尝试: use_mixed_precision: false
├─ 调整: entropy_coeff (增加2倍)
└─ 查看详情: [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) § 故障排除

问题: 显存泄漏
├─ 运行: python tools/memory_leak_detector.py
├─ 分析: 检查Tensor分析报告
├─ 修复: 按照建议在代码中添加del或.detach()
└─ 查看详情: [tools/memory_leak_detector.py](tools/memory_leak_detector.py)
```

---

## 📊 我想监测训练性能

### 性能监测工具

#### 1. 实时显存监控
```bash
# 方式1: GPU诊断工具
python diagnose_gpu.py

# 方式2: nvidia-smi (Windows/Linux)
nvidia-smi -l 1

# 方式3: Python查询
python -c "from utils import print_memory_info; print_memory_info()"
```

#### 2. TensorBoard 可视化
```bash
# 训练时并行运行
tensorboard --logdir results/
```
→ 在 http://localhost:6006 查看损失、奖励、指标

#### 3. 性能数据分析
```python
from tools.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(name="my_training")
# 在训练循环中
monitor.update(episode, loss, reward)
# 训练完成
monitor.print_summary()
```

→ 详细用法: [tools/performance_monitor.py](tools/performance_monitor.py)

---

## 🧪 我想进行对比实验

### 实验对比框架

```python
from tools.performance_monitor import PerformanceComparator

# 收集实验数据
comparator = PerformanceComparator()
comparator.load_experiment("实验A", Path("results/exp_a_monitor.json"))
comparator.load_experiment("实验B", Path("results/exp_b_monitor.json"))

# 对比分析
comparator.compare_memory()       # 显存对比
comparator.compare_speed()        # 速度对比
comparator.plot_memory_timeline() # 绘制显存曲线
```

→ 完整例子: [tools/performance_monitor.py](tools/performance_monitor.py#PerformanceComparator)

---

## 📁 文件总览

### 核心训练文件

| 文件 | 推荐使用 | 适用场景 |
|------|---------|---------|
| [train_ippo_lowmem.py](train_ippo_lowmem.py) | ⭐ 首选 | RTX 4060 Ti, 低显存 |
| [train_ippo.py](train_ippo.py) | - | 原始配置, 高端GPU |

### 配置文件

| 文件 | 推荐使用 | 显存占用 |
|------|---------|---------|
| [configs/lowmem_config.yaml](configs/lowmem_config.yaml) | ⭐ 推荐 | 2-3 GB |
| [configs/default_config.yaml](configs/default_config.yaml) | 高端GPU | 6-8 GB |

### 工具和诊断

| 文件 | 功能 | 运行时间 |
|------|------|---------|
| [diagnose_gpu.py](diagnose_gpu.py) | 硬件诊断 | 2 min |
| [tools/performance_monitor.py](tools/performance_monitor.py) | 性能监测 | 实时 |
| [tools/memory_leak_detector.py](tools/memory_leak_detector.py) | 泄漏检测 | 5 min |

### 文档和指南

| 文件 | 内容 | 阅读时间 |
|------|------|---------|
| [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) | 一页纸速查表 | 15 min |
| [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) | 完整优化指南 | 60 min |
| [GPU_OPTIMIZATION_REPORT.md](GPU_OPTIMIZATION_REPORT.md) | 优化完成总结 | 30 min |

---

## 🚀 推荐工作流程

### Day 1: 环境验证
```
1. python diagnose_gpu.py  (3 min)
2. python test_env.py      (2 min)
3. python train_ippo_lowmem.py --episodes 10  (5 min)
```
✅ 确认一切正常

### Day 2-3: 完整训练
```
1. python train_ippo_lowmem.py --episodes 500  (2-3 hours)
2. tensorboard --logdir results/              (实时监控)
```
✅ 获得基准性能

### Day 4+: 优化实验
```
1. 复制和修改配置
2. 运行自定义训练
3. 使用PerformanceComparator对比
4. 迭代优化
```
✅ 持续改进

---

## 💡 关键要点总结

### ✅ 必须做

- [x] 使用 [train_ippo_lowmem.py](train_ippo_lowmem.py) (不是 train_ippo.py)
- [x] 使用 [configs/lowmem_config.yaml](configs/lowmem_config.yaml) 配置
- [x] 运行 [diagnose_gpu.py](diagnose_gpu.py) 进行诊断
- [x] 启用 use_mixed_precision: true

### ❌ 不要做

- [ ] 使用原始 train_ippo.py + default_config.yaml (会OOM)
- [ ] 使用 batch_size > 16 (显存浪费)
- [ ] 忽视显存监控 (容易OOM)
- [ ] 修改state_dim或hidden_dim而不考虑显存 (风险高)

### 🎯 性能目标 (RTX 4060 Ti)

```
显存占用: 2-3 GB
平均速度: 10-20 episodes/min
任务完成率: 70-75%
训练时间: 25-50分钟 (500 episodes)
```

---

## 📞 获取帮助

### 快速问题

- 查看: [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) (一页纸)
- 问: "遇到CUDA Out of Memory？" → [常见问题快速修复](GPU_QUICK_REFERENCE.md#常见问题快速修复)

### 中等问题

- 读: [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) (详细)
- 搜: 目录中的相应章节

### 深层问题

- 运行: [diagnose_gpu.py](diagnose_gpu.py) 进行诊断
- 运行: [tools/memory_leak_detector.py](tools/memory_leak_detector.py) 检测泄漏
- 查看: [GPU_OPTIMIZATION_REPORT.md](GPU_OPTIMIZATION_REPORT.md) 的完整分析

---

## 版本信息

- **GPU优化版本**: 1.0
- **发布日期**: 2026-04-12
- **目标硬件**: RTX 4060 Ti (8GB VRAM)
- **框架版本**: PyTorch 2.0+ + Gymnasium

---

**最后更新**: 2026-04-12

**下次使用时**: 直接参考 [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) 进行快速操作

