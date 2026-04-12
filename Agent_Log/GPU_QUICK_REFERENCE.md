# GPU 优化快速参考卡 - RTX 4060 Ti

## 🚀 快速开始（3步）

### 第1步：验证环境
```bash
# 检查显存和硬件
python diagnose_gpu.py

# 验证环境可用性
python test_env.py
```

### 第2步：快速测试（5分钟）
```bash
# 快速训练10个episode
python train_ippo_lowmem.py --episodes 10 --name quicktest

# 结果查看
tensorboard --logdir results/
```

### 第3步：完整训练（2-3小时）
```bash
# 标准配置 (推荐)
python train_ippo_lowmem.py --episodes 500 --name main_training

# 或使用自定义配置
python train_ippo_lowmem.py --config configs/lowmem_config.yaml
```

---

## ⚙️ 核心参数说明

### 环境配置（configs/lowmem_config.yaml）

#### 🏠 系统规模
```yaml
environment:
  num_edge_servers: 2        # 边缘服务器数 (1-5)
  num_end_devices: 5         # 终端设备数 (2-10)  
  num_tasks: 10              # 任务数量 (5-20)
  state_dim: 32              # 状态维度 (16-64)
```

**调整建议:**
- 显存不足 → 减少num_tasks, num_end_devices
- 训练太快 → 增加num_edge_servers
- 效果不好 → 增加state_dim

#### 🤖 Agent配置  
```yaml
marl:
  num_agents: 3              # agent数量 (1-10)
  hidden_dim: 32             # 网络隐藏层 (16-128)
  num_layers: 1              # 网络深度 (1-2)
  total_episodes: 500        # 总episodes (100-10000)
```

**调整建议:**
- GPU内存不足 → 减少hidden_dim或num_layers
- 收敛太慢 → 增加total_episodes
- 收敛不稳定 → 增加hidden_dim

#### 🎓 训练算法
```yaml
algorithm:
  batch_size: 8              # 批次大小 (2-64)
  num_epochs: 2              # epoch数 (1-10)
  learning_rate: 3e-4        # 学习率 (1e-5 - 1e-3)
  entropy_coeff: 0.01        # 熵系数 (0.001 - 0.1)
  gradient_accumulation_steps: 4  # 梯度累积 (1-8)
```

**调整建议:**
- OOM错误 → batch_size ÷ 2, entropy_coeff ÷ 2
- Loss为NaN → learning_rate ÷ 10
- 不收敛 → entropy_coeff × 2
- 显存占用 → gradient_accumulation_steps × 2

#### 🚀 优化选项
```yaml
algorithm:
  use_mixed_precision: true  # FP16混合精度 (推荐)
  enable_gradient_checkpointing: false  # 梯度检查点
```

---

## 📊 显存占用快速查表

| Config | Hidden | Batch | FP32 | FP16 | 推荐场景 |
|--------|--------|-------|------|------|---------|
| 超小 | 16 | 2 | 0.8GB | 0.4GB | 调试 |
| **小** | **32** | **8** | **3.5GB** | **1.8GB** | **标准** |
| 中 | 64 | 16 | 7.0GB | 3.5GB | 高端GPU |
| 大 | 128 | 32 | 14GB | 7.0GB | A100+ |

✅ **推荐**: hidden_dim=32, batch_size=8, FP16 (显存2-3GB)

---

## 🔧 常见问题快速修复

### ❌ CUDA out of memory
```bash
# 选项1: 减小batch_size
batch_size: 8 → 4 → 2

# 选项2: 启用混合精度
use_mixed_precision: true

# 选项3: 减小network
hidden_dim: 32 → 16
num_layers: 2 → 1

# 选项4: 减小环境
num_edge_servers: 2 → 1
num_end_devices: 5 → 3
```

### ❌ Loss为NaN
```bash
# 选项1: 降低学习率
learning_rate: 3e-4 → 1e-4 → 1e-5

# 选项2: 增加entropy_coeff
entropy_coeff: 0.01 → 0.02 → 0.05

# 选项3: 使用FP32 (禁用混合精度)
use_mixed_precision: false

# 选项4: 减小网络大小
hidden_dim: 32 → 16
```

### ❌ 训练过慢
```bash
# 选项1: 启用mixed precision
use_mixed_precision: true  # +20-50% 速度

# 选项2: 增加批次大小
batch_size: 8 → 16  # 如果显存允许

# 选项3: 减少evaluation频率
eval_episodes: 10 → 5

# 选项4: 关闭TensorBoard logging
log_interval: 10 → 100
```

### ❌ 显存泄漏（持续增长）
```bash
# 运行检测工具
python tools/memory_leak_detector.py

# 检查代码
1. 确保del variable后的tensor被删除
2. 使用 torch.no_grad() 包装评估代码
3. 定期 torch.cuda.empty_cache()
```

---

## 🎯 性能预期

### RTX 4060 Ti with lowmem_config

| Metric | 预期值 |
|--------|--------|
| GPU显存占用 | 2-3 GB |
| 平均速度 | 10-20 episodes/min |
| 完整训练时间 (500 ep) | 25-50分钟 |
| GPU利用率 | 60-80% |
| 任务完成率 | 70-75% |

### 调整后的速度表

```
Hidden | Batch | FP型 | 相对速度 | 显存(GB)
16     | 2     | FP32 | 1.0x   | 0.8
32(*)  | 8     | FP16 | 1.5x   | 2.0
32     | 8     | FP32 | 1.0x   | 3.5
64     | 16    | FP16 | 2.0x   | 4.5
128    | 32    | FP32 | 0.7x   | >8GB (超出)

(*) = 推荐配置
```

---

## 🛠️ 调试技巧

### 实时监测
```bash
# 方法1: nvidia-smi (Windows/Linux)
nvidia-smi -l 1

# 方法2: Python打印
python
>>> from utils import print_memory_info
>>> print_memory_info()

# 方法3: TensorBoard
tensorboard --logdir results/
```

### 精确调参流程

```
1. 设定目标 (显存, 速度, 准确度)
   ↓
2. 从lowmem_config开始
   ↓
3. 逐个调整参数 (只改1个!)
   ↓
4. 监测效果 (gpu_monitor输出)
   ↓
5. 回到步骤3或保存最优配置
```

### 配置模板

```yaml
# 极端优化 (最少显存)
environment:
  num_edge_servers: 1
  num_end_devices: 2
  num_tasks: 5
marl:
  hidden_dim: 16
  num_agents: 1
algorithm:
  batch_size: 2
  gradient_accumulation_steps: 4
  use_mixed_precision: true

# 平衡配置 (推荐)
[使用默认 lowmem_config.yaml]

# 激进优化 (最大性能)
environment:
  num_edge_servers: 3
  num_end_devices: 8
  num_tasks: 15
marl:
  hidden_dim: 64
  num_agents: 5
algorithm:
  batch_size: 16
  gradient_accumulation_steps: 2
  use_mixed_precision: true
```

---

## 📈 监测指标

### 每10 step检查
```
✓ GPU内存占用 < 85%
✓ Loss在下降趋势
✓ 无CUDA错误
```

### 每100 step检查  
```
✓ 训练速度稳定 (>10 ep/min)
✓ GPU利用率 60-80%
✓ 内存无泄漏
```

### 完成后检查
```
✓ 峰值显存 < 6GB
✓ 平均显存 < 4GB
✓ 任务完成率 ≥ 70%
✓ 训练无异常中断
```

---

## 🗂️ 常用文件位置

```
项目根目录/
├── configs/
│   ├── lowmem_config.yaml      ← 推荐使用
│   └── default_config.yaml
├── train_ippo_lowmem.py        ← 推荐使用
├── test_env.py                 ← 环境验证
├── quick_start.py              ← 快速测试
├── diagnose_gpu.py             ← GPU诊断 ⭐
├── tools/
│   ├── performance_monitor.py  ← 性能监测
│   └── memory_leak_detector.py ← 泄漏检测
├── results/                    ← 保存目录
└── GPU_OPTIMIZATION_GUIDE.md   ← 完整指南
```

---

## 📞 故障排除电话树

```
问题: 显卡无法检测
├─ 运行: python diagnose_gpu.py
├─ 检查: NVIDIA驱动版本
└─ 解决: 更新驱动或CUDA

问题: 显存快速增长
├─ 运行: python tools/memory_leak_detector.py
├─ 检查: 是否有del或detach
└─ 解决: 修改代码清理变量

问题: Loss为NaN
├─ 检查: 学习率是否过大
├─ 尝试: use_mixed_precision=false
└─ 调整: entropy_coeff更大

问题: 训练太慢
├─ 启用: use_mixed_precision=true
├─ 检查: batch_size是否过小
└─ 优化: 减少eval_episodes

问题: 想要更好的效果
├─ 增加: total_episodes
├─ 增加: hidden_dim
└─ 确保: 显存够用
```

---

## 💾 检查清单

### 开始训练前 ✓
- [ ] 运行 `python diagnose_gpu.py`
- [ ] 确认显存 ≥ 8GB
- [ ] 运行 `python test_env.py`
- [ ] 检查配置文件无误

### 训练中监测 ✓
- [ ] 每20分钟检查一次显存占用
- [ ] 观察loss曲线是否在下降
- [ ] 检查是否有OOM错误
- [ ] TensorBoard观察reward趋势

### 训练完成后 ✓
- [ ] 检查checkpoint文件大小
- [ ] 查看最终性能指标
- [ ] 保存monitor数据进行对比
- [ ] 保存配置以便重现

---

## 🔗 相关命令速查

```bash
# 诊断和测试
python diagnose_gpu.py                    # GPU诊断
python test_env.py                        # 环境验证
python tools/memory_leak_detector.py       # 检测泄漏

# 训练
python train_ippo_lowmem.py --episodes 500  # 标准训练
python quick_start.py --episodes 10         # 快速测试
python train_ippo.py                        # 原始配置 (不推荐)

# 监测
tensorboard --logdir results/               # 可视化
python -c "from utils import print_memory_info; print_memory_info()"  # 显存查询

# 配置
cp configs/lowmem_config.yaml configs/my_config.yaml  # 复制配置
nano configs/my_config.yaml                            # 编辑配置
```

---

**最后更新**: 2026-04-12  
**版本**: 1.0  
**硬件**: RTX 4060 Ti (8GB VRAM)  
**框架**: PyTorch + Gymnasium
