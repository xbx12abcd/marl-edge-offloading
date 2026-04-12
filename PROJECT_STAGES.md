# MARL Edge Offloading - 项目阶段执行指南

## 项目状态总览

| 阶段 | 功能 | 权重 | 状态 | 进度 |
|------|------|------|------|------|
| 1 | 仿真环境 + IPPO基线 | 30% | ✓ 完成 | 100% |
| 2 | Explaboff (互信息) | 35% | ✓ 完成 | 100% |
| 3 | 大规模场景优化 | 35% | 进行中 | 50% |
| 4 | GUI可视化演示 | +10% | 计划中 | 0% |

---

## 阶段1：仿真环境与IPPO基线 (30%)

### 📋 已完成的工作：

✓ 构建边缘计算网络仿真环境
- 多端设备和边缘服务器
- 任务生成和调度
- 资源约束和能量消耗模型
- 网络通信延迟计算

✓ 实现IPPO（独立PPO）算法
- Actor-Critic 网络架构
- GAE 优势估计
- PPO损失函数 + 熵正则化
- 轨迹采样和批量更新

✓ 配置和脚本框架
- YAML配置文件管理
- 训练脚本和评估脚本
- 指标收集和TensorBoard集成

### 🚀 执行步骤：

```bash
# 1. 快速测试环境和智能体
python test_env.py

# 2. 快速训练验证（调试用）
python quick_start.py --episodes 50 --length 50

# 3. 完整IPPO训练
python train_ippo.py --config configs/default_config.yaml --episodes 500
```

### 📊 期望输出：

训练完成后在 `results/<experiment_name>/` 生成：
- `config.yaml`: 配置文件
- `results.json`: 最终指标
- `checkpoints/`: 模型文件
- `logs/`: TensorBoard日志

### 关键指标：

| 指标 | 含义 |
|------|------|
| Task Completion Rate | 成功完成的任务比例 |
| Energy Consumption | 总能量消耗 |
| Average Delay | 平均任务延迟 |
| Fairness Index | 资源分配公平性 (Jain指数) |

---

## 阶段2：Explaboff方案（互信息）(35%)

### 📋 已完成的工作：

✓ 互信息估计器（MutualInformationEstimator）
- 基于噪声对比估计（NCE）
- 估计智能体间的信息共享量
- 可在奖励函数中使用

✓ 注意力通信模块（AttentionCommunicationModule）
- 多头注意力机制
- 选择性地与top-k个智能体通信
- 消息聚合和状态更新

✓ Explaboff智能体（ExplaboffAgent）
- 扩展PPO with通信能力
- 互信息奖励融合
- 可解释性权重提取

### 🚀 执行步骤：

```bash
# 1. 对比IPPO和Explaboff性能
python evaluate_compare.py

# 2. 专用Explaboff配置训练
# (需要实现 train_explaboff.py)
python train_explaboff.py --config configs/explaboff_config.yaml
```

### 📊 新增指标：

| 指标 | 含义 |
|------|------|
| Mutual Information | 智能体间的互信息量 |
| Communication Cost | 通信开销 |
| Explainability Score | 策略可解释性得分 |

### ⚙️ 配置参数：

```yaml
# configs/explaboff_config.yaml
algorithm:
  mi_weight: 0.5              # MI奖励权重
  mi_estimator: "nwj"         # NCE估计器
  learning_rate: 5e-5         # 降低学习率防止不稳定

communication:
  enable_communication: true  # 启用通信
  comm_bandwidth: 50.0        # 通信带宽
  max_message_size: 256       # 消息大小
  top_k_agents: 3            # 通信伙伴数
  comm_method: "attention"    # 注意力机制
```

---

## 阶段3：大规模场景优化 (35%)

### 📋 需要完成的工作：

**可扩展性改进：**
- [ ] 实现分层MARL架构
- [ ] 图神经网络(GNN)用于智能体间关系建模
- [ ] 参数共享机制
- [ ] 动态智能体管理

**隐式协作改进：**
- [ ] 改进消息传递协议
- [ ] 减少通信开销
- [ ] 支持部分可观测性(POMDP)

**公平性和可解释性：**
- [ ] 多目标优化（fairness + efficiency）
- [ ] SHAP/LIME解释技术
- [ ] 奖励整形以确保公平

### 🎯 目标指标：

| 指标 | 目标 |
|------|------|
| Scalability | 支持50+智能体 |
| Communication Overhead | < 5% 计算成本 |
| Fairness Index | > 0.8 |
| Training Convergence | < 1000 episodes |

### 🚀 待实现的脚本：

```bash
# 大规模场景测试
python train_scalable.py --num_agents 50 --num_episodes 1000

# 性能分析
python analyze_scalability.py

# 通信效率分析
python analyze_communication.py
```

---

## 阶段4：GUI可视化演示 (+10%)

### 📋 需要完成的工作：

**实时演示平台：**
- [ ] PyQt5/Tkinter GUI框架
- [ ] 实时任务卸载可视化
- [ ] 网络拓扑图显示
- [ ] 性能指标实时更新
- [ ] 播放/暂停/加速控制

**可视化组件：**
- [ ] 设备状态显示（CPU/能量）
- [ ] 任务队列可视化
- [ ] 通信路径绘制
- [ ] 奖励/延迟曲线

### 📁 文件结构：

```
gui/
├── __init__.py
├── main_window.py       # 主窗口
├── visualization.py     # 可视化组件
├── simulation_widget.py  # 仿真视图
├── metrics_widget.py     # 指标显示
└── config_widget.py      # 配置界面
```

### 🎨 预期界面：

```
┌─────────────────────────────────────────┐
│   MARL Edge Offloading - 实时演示平台   │
├─────────────────────────────────────────┤
│ [启动] [暂停] [重置] 速度: [████] 2x    │
├─────────────────────────────────────────┤
│                                         │
│   [网络拓扑图]           [性能指标]    │
│                                         │
│   End Devices (5)        ▄ Completion  │
│   ███ Edge Servers (3)   ▄ Energy      │
│                          ▄ Delay       │
│   任务队列: 8/20         ▄ Fairness    │
│                                         │
└─────────────────────────────────────────┘
```

---

## 环境要求和验证

### Python环境检查：

```bash
# 1. 激活conda环境
conda activate marl-edge

# 2. 验证关键包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 3. 检查GPU可用性
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 一级依赖：

```
torch>=1.13.0           # 深度学习框架
gymnasium>=0.27.0       # 强化学习工具包
numpy>=1.21.0           # 数值计算
matplotlib>=3.5.0       # 可视化
pytorch-lightning>=1.8.0 # 训练框架
tensorboard>=2.10.0     # 日志记录
```

---

## 执行时间表

### 推荐执行顺序：

```
├─ 第1天：环境配置 + 快速测试
│  ├─ test_env.py (5分钟)
│  └─ quick_start.py (10分钟)
│
├─ 第2-3天：IPPO基线训练
│  ├─ train_ippo.py + 500 episodes (2-3小时)
│  └─ 分析结果
│
├─ 第4-5天：Explaboff方案验证
│  ├─ evaluate_compare.py (30分钟)
│  └─ train_explaboff.py (1-2小时)
│
├─ 第6-10天：大规模优化
│  ├─ 实现分层/GNN架构
│  ├─ train_scalable.py (2-3小时)
│  └─ 性能分析
│
└─ 第11-15天：GUI开发
   ├─ 基础框架 (2天)
   ├─ 可视化组件 (2天)
   └─ 集成和测试 (1天)
```

---

## 常见问题

### Q: 训练很慢怎么办？

**A:** 尝试以下优化：

```bash
# 1. 减少episode数量
python train_ippo.py --episodes 100

# 2. 使用较小的网络
# 修改 configs/default_config.yaml:
algorithm:
  hidden_dim: 64  # 从128降低到64

# 3. 减少智能体数量
environment:
  num_agents: 5  # 从10降低到5

# 4. 使用GPU（如果可用）
# PyTorch会自动检测CUDA
```

### Q: 内存不足怎么办？

**A:** 

```bash
# 减少批次大小
algorithm:
  batch_size: 32  # 从64降低到32

# 减少轨迹缓冲区
# 修改 train_ippo.py 中的 episode_length
```

### Q: 结果不收敛怎么办？

**A:** 调整超参数

```yaml
# 降低学习率
algorithm:
  learning_rate: 5e-5  # 从1e-4降低

# 增加熵系数以鼓励探索
algorithm:
  entropy_coeff: 0.05  # 从0.01增加

# 调整折扣因子
algorithm:
  gamma: 0.95  # 从0.99
```

---

## 性能基准

### IPPO基线预期性能：

```
Task Completion Rate:    70-80%
Energy Consumption:      5000-7000 J
Average Delay:           10-15 time slots
Fairness Index:          0.7-0.8
```

### Explaboff预期改进：

```
Task Completion Rate:    +5-10%
Energy Consumption:      -10-15%
Average Delay:           -5-10%
Fairness Index:          +5%
```

### 大规模场景目标：

```
支持智能体数：          50+
通信开销：             < 5%
训练收敛速度：         < 1000 episodes
```

---

## 数据和日志位置

### 输出目录结构：

```
results/
├── IPPO_<timestamp>/
│   ├── config.yaml
│   ├── results.json
│   ├── results.txt
│   ├── checkpoints/
│   │   ├── episode_000100.pt
│   │   ├── episode_000200.pt
│   │   └── final_model.pt
│   └── logs/
│       ├── events.out.tfevents.*
│       └── ...
│
├── comparison/
│   ├── comparison_results.json
│   └── comparison_plots.png
│
└── quick_test/
    └── results.json
```

### TensorBoard 查看：

```bash
tensorboard --logdir=./logs --port=6006
# 访问 http://localhost:6006
```

---

## 代码提交检查清单

在完成每个阶段后，确保：

- [ ] 所有脚本都能成功运行
- [ ] 生成的结果文件完整
- [ ] README和文档已更新
- [ ] 代码注释清晰
- [ ] 没有未使用的导入
- [ ] 错误处理完善

---

## 联系和支持

如有技术问题或建议，请参考：

- PyTorch 文档：https://pytorch.org/docs
- Gymnasium 文档：https://gymnasium.farama.org
- PPO 论文：https://arxiv.org/abs/1707.06347
- MARL 综述：https://arxiv.org/abs/1908.03963

---

更新时间：2026年4月12日
项目进度：70% 完成
