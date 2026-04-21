# MARL Edge Offloading: 多智能体强化学习在边缘计算中的应用

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.25+-orange.svg)](https://pettingzoo.farama.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目概述

本项目通过多智能体强化学习（MARL）解决边缘计算中的**任务卸载与调度**问题。每个端设备作为独立智能体，自主决定将计算任务在本地执行还是卸载至某台边缘服务器，目标是在**任务完成率、能量消耗、传输延迟和资源公平性**之间取得最优平衡。

### 核心问题

- **可扩展性**：支持 50+ 设备、多服务器的大规模调度
- **通信约束**：智能体间协作的隐式/显式通信限制
- **多目标优化**：能量、延迟、公平性的联合优化
- **可解释性**：决策过程透明可分析

### 系统架构

```
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│   End Devices      │    │   Edge Servers     │    │   Cloud Layer      │
│  (agent_0..N-1)    │───▶│  (server_0..M-1)   │───▶│  (Data Center)     │
└────────────────────┘    └────────────────────┘    └────────────────────┘
         │                          │
         └──────── MARL Controller ─┘
                  ┌──────────────┐
                  │  IPPO        │  Stage 1: 独立PPO基线
                  │  Explaboff   │  Stage 2: 互信息通信
                  │  GNN-PPO     │  Stage 3: 图神经网络（当前）
                  └──────────────┘
```

---

## 项目状态

| 阶段 | 内容 | 权重 | 状态 | 进度 |
|------|------|------|------|------|
| **1** | 仿真环境 + IPPO基线 | 30% | ✅ 完成 | 100% |
| **2** | Explaboff 互信息通信 | 35% | ✅ 完成 | 100% |
| **3** | GNN 大规模场景优化 | 35% | 🔄 进行中 | 75% |
| **4** | GUI 可视化演示 | +10% | 📅 计划中 | 0% |

**整体进度：约 82%**（最后更新：2026-04-21）

---

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | 3.10+ |
| PyTorch | 1.13+ |
| PettingZoo | 1.25+ |
| CUDA | 12.1+（推荐） |
| GPU 显存 | 8GB+（RTX 4060 Ti 已验证） |
| 内存 | 16GB+ RAM |

---

## 快速开始

### 1. 安装依赖

```bash
conda create -n marl-edge python=3.10 -y
conda activate marl-edge
pip install -r requirements.txt
```

### 2. 验证环境

```bash
python -c "import torch, pettingzoo, gymnasium; print('OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. 快速训练测试（5分钟）

```bash
python quick_start.py --episodes 50 --length 50
```

---

## 训练脚本

### `train_ippo.py` — Stage 1 & 2 标准训练

基于 PettingZoo 并行 API，为每个智能体独立训练 PPO 策略（Independent PPO）。

```bash
# 默认配置训练（10 智能体，500 episodes）
python train_ippo.py --episodes 500

# 指定配置和设备
python train_ippo.py --config configs/default_config.yaml --episodes 1000 --device cuda

# 低显存机器
python train_ippo.py --config configs/lowmem_config.yaml --episodes 500 --device cpu

# 自定义实验名称（结果保存到 results/my_exp/）
python train_ippo.py --episodes 500 --name my_exp
```

### `train_scalable.py` — Stage 3 大规模 GNN 训练

参数共享 + 图注意力网络（GAT），支持 50+ 智能体。

```bash
# 20 智能体（测试用）
python train_scalable.py --num_agents 20 --episodes 500

# 50 智能体大规模场景（目标配置）
python train_scalable.py --num_agents 50 --episodes 1000 --device cuda

# 自定义实验名
python train_scalable.py --num_agents 50 --episodes 1000 --name GNN_50agents
```

### `train_explaboff.py` — Stage 2 Explaboff 训练

```bash
python train_explaboff.py --config configs/explaboff_config.yaml
```

---

## 评估脚本

### `evaluate_checkpoint.py` — 单模型评估 / 推演演示

加载已训练的 `.pt` 检查点文件，在环境中运行若干 episode 并输出性能指标。

**评估模式**（统计多 episode 平均指标）：
```bash
# 基本用法：评估 final_model.pt，跑 10 个 episode
python evaluate_checkpoint.py results/IPPO_20260421_120000/checkpoints/final_model.pt

# 指定 episode 数量
python evaluate_checkpoint.py results/<exp>/checkpoints/final_model.pt --episodes 20

# 指定配置文件（默认自动从实验目录查找）
python evaluate_checkpoint.py results/<exp>/checkpoints/final_model.pt --config configs/default_config.yaml

# 同时显示每步的动作和奖励
python evaluate_checkpoint.py results/<exp>/checkpoints/final_model.pt --render
```

**演示模式**（逐步打印每个时步的决策细节）：
```bash
# 逐步演示 50 步，每步间隔 0.5 秒
python evaluate_checkpoint.py results/<exp>/checkpoints/final_model.pt --demo --steps 50

# 加快演示速度
python evaluate_checkpoint.py results/<exp>/checkpoints/final_model.pt --demo --steps 100 --delay 0.1
```

输出结果保存到：`results/<exp>/evaluation/<checkpoint_name>_evaluation.json`

---

### `evaluate_compare.py` — IPPO vs Explaboff 算法对比

从零训练或加载已有检查点，横向对比两种算法的性能，生成对比图表。

```bash
# 从随机初始化训练后对比（各训练 200 episode，评估 30 episode）
python evaluate_compare.py --train 200 --eval 30

# 使用已训练的检查点（跳过训练，直接评估对比）
python evaluate_compare.py \
  --ippo_ckpt results/IPPO_xxx/checkpoints/final_model.pt \
  --explaboff_ckpt results/Explaboff_xxx/checkpoints/final_model.pt \
  --eval 30

# 也可只为其中一个算法提供检查点
python evaluate_compare.py \
  --ippo_ckpt results/IPPO_xxx/checkpoints/final_model.pt \
  --train 200 --eval 30
```

输出到 `results/comparison/`：
- `comparison_results.json`：数字结果
- `comparison_plots.png`：8 项指标的对比折线图（含训练曲线 + 互信息）

---

## 两个评估脚本的核心区别

| 维度 | `evaluate_checkpoint.py` | `evaluate_compare.py` |
|------|--------------------------|----------------------|
| **用途** | 评估一个已训练好的模型 | 横向对比两种算法 |
| **输入** | 需要 `.pt` 检查点文件 | 可选检查点，无则从随机初始化训练 |
| **算法** | 单一算法（IPPO / GNN） | IPPO + Explaboff 同时运行 |
| **输出** | 该模型的绝对性能指标 | 两算法的相对提升百分比 + 对比图 |
| **MI 奖励** | 不适用 | ExplaboffAgent 每步融入 MI 奖励信号 |
| **典型场景** | 训练完成后验证模型效果 | 阶段2汇报：展示通信机制的增益 |

---

## 项目结构

```
Communication_Network_project/
│
├── train_ippo.py              # Stage 1/2 标准训练（PettingZoo + SimplePPO）
├── train_scalable.py          # Stage 3 大规模 GNN 训练
├── train_explaboff.py         # Stage 2 Explaboff 训练
├── quick_start.py             # 快速调试训练
├── evaluate_checkpoint.py     # 单模型评估 / 逐步演示
├── evaluate_compare.py        # IPPO vs Explaboff 对比
├── test_env.py                # 环境基础功能测试
│
├── envs/
│   ├── __init__.py            # 导出 EdgeOffloadingEnv, EdgeComputingEnv
│   └── edge_env.py            # PettingZoo 并行环境（主）+ 单智能体包装器（兼容）
│
├── agents/
│   ├── ppo_agent.py           # Stage 1 IPPO 智能体
│   ├── explaboff_agent.py     # Stage 2 Explaboff 智能体（互信息通信）
│   ├── networks.py            # 网络架构基类
│   └── gnn_agent.py           # Stage 3 GNN 智能体（图注意力 + 参数共享）
│
├── utils/
│   ├── helpers.py             # 配置加载、指标收集、工具函数
│   ├── task_device.py         # Task / Device 数据类
│   └── gpu_monitor.py         # GPU 显存监控与诊断
│
├── configs/
│   ├── default_config.yaml    # 标准配置（10 智能体，64维状态）
│   ├── lowmem_config.yaml     # 低显存配置（RTX 4060 Ti 8GB）
│   └── explaboff_config.yaml  # Stage 2 Explaboff 配置
│
├── results/                   # 训练输出（运行后自动生成）
│   ├── IPPO_<timestamp>/
│   │   ├── config.yaml
│   │   ├── results.json
│   │   ├── checkpoints/
│   │   │   ├── episode_000499.pt
│   │   │   └── final_model.pt
│   │   ├── logs/              # TensorBoard 日志
│   │   └── evaluation/        # evaluate_checkpoint.py 输出
│   ├── GNN_n50_<timestamp>/   # Stage 3 输出
│   └── comparison/            # evaluate_compare.py 输出
│
└── gui/                       # Stage 4（待实现）
```

---

## 关键概念

### 动作空间
| 动作值 | 含义 |
|--------|------|
| `0` | 本地执行（在端设备上运行） |
| `1` | 卸载至边缘服务器 1 |
| `2` | 卸载至边缘服务器 2 |
| `...` | ... |
| `n` | 卸载至边缘服务器 n |

### 奖励函数
```
成功完成（deadline 内）：
  reward = completion_weight(0.4)
         - energy_penalty(0.1) × energy_consumed
         - deadline_penalty(0.2) × total_delay
         + fairness_bonus(0.1) × (1 - std(server_loads))  # 仅卸载时

超时失败：
  reward = -10.0
```

### 观测空间（64维向量）
| 维度 | 内容 |
|------|------|
| 0 | 当前任务 CPU 需求（归一化） |
| 1 | 当前任务数据大小（归一化） |
| 2 | 当前任务 deadline（归一化） |
| 3 | 任务等待时间（归一化） |
| 4 | 本地队列长度（归一化） |
| 5~9 | 各边缘服务器当前负载（归一化） |
| -2 | Episode 进度（当前步/总步数） |
| -1 | 剩余未完成任务比例 |

---

## 性能基准

| 算法 | 智能体数 | 完成率 | 能量(J) | 平均延迟 | 公平性 |
|------|---------|--------|---------|---------|--------|
| IPPO（基线） | 10 | 65–75% | ~6200 | 12.5 | 0.72 |
| Explaboff | 10 | 75–82% | ~5300 | 11.8 | 0.76 |
| GNN-PPO（目标） | 50 | >80% | 更低 | 更短 | >0.85 |

### GPU 显存参考

| 配置文件 | 智能体数 | 显存占用 |
|---------|---------|---------|
| `lowmem_config.yaml` | 3–5 | ~1–2 GB |
| `default_config.yaml` | 10 | ~3–4 GB |
| `train_scalable` N=50 | 50 | ~5–7 GB |

---

## 查看训练曲线

```bash
# 启动 TensorBoard
tensorboard --logdir results/

# 浏览器访问 http://localhost:6006
```

可查看：每个 episode 的奖励曲线、任务完成率、公平性指数、各项损失（policy/value/entropy）。

---

## 常见问题

**Q: CUDA out of memory**
```bash
python train_ippo.py --config configs/lowmem_config.yaml
# 或强制 CPU
python train_scalable.py --device cpu --num_agents 20
```

**Q: ImportError: No module named 'pettingzoo'**
```bash
pip install pettingzoo>=1.25.0
```

**Q: 训练不收敛**
```yaml
# 调整 configs/default_config.yaml
algorithm:
  learning_rate: 5e-5   # 降低学习率
  entropy_coeff: 0.05   # 增加探索
```

---

## 依赖

```
torch>=1.13.0
pettingzoo>=1.25.0
gymnasium>=0.27.0
numpy>=1.21.0
matplotlib>=3.5.0
tensorboard>=2.10.0
pyyaml>=6.0
networkx>=2.8
tqdm>=4.64.0
```

---

## 参考

- PPO 论文：https://arxiv.org/abs/1707.06347
- MARL 综述：https://arxiv.org/abs/1908.03963
- PettingZoo 文档：https://pettingzoo.farama.org
- PyTorch 文档：https://pytorch.org/docs

---

## 许可证

MIT License
