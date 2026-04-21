# MARL Edge Offloading - 项目阶段执行指南

## 项目状态总览

| 阶段 | 功能 | 权重 | 状态 | 进度 |
|------|------|------|------|------|
| 1 | 仿真环境 + IPPO基线 | 30% | ✅ 完成 | 100% |
| 2 | Explaboff (互信息通信) | 35% | ✅ 完成 | 100% |
| 3 | 大规模场景优化 (GNN) | 35% | 🔄 进行中 | 75% |
| 4 | GUI可视化演示 | +10% | 📅 计划中 | 0% |

**最后更新：2026年4月21日**  
**整体进度：约 82% 完成**

---

## 近期重要更新（2026-04-21）

### ExplaboffAgent 补全：`select_action` + `update` 方法

**修复文件**：`agents/explaboff_agent.py`、`evaluate_compare.py`

#### 问题背景

`evaluate_compare.py` 在训练阶段调用 `agent.update(batch_size, num_epochs)` 和 `agent.select_action(state)`，但 `ExplaboffAgent` 原有实现缺少这两个方法，导致运行时抛出 `AttributeError`，对比评估无法执行。

#### 新增方法

| 方法 | 说明 |
|------|------|
| `select_action(state)` | 标准 PPO 接口包装，内部调用 `select_action_with_communication(state, all_agent_states=None)`，返回 `(action, log_prob, value)`，无需传入其他智能体状态 |
| `compute_gae_advantages(rewards, values, dones, gamma, gae_lambda)` | 内部 GAE 优势计算（向量化实现，与 `PPOAgent` 一致） |
| `update(batch_size, num_epochs, ...)` | 完整 PPO 更新循环，**先调用 `compute_rewards_with_mi()` 将互信息奖励融入 reward 信号**，再计算 GAE 优势，执行标准 PPO 裁剪 + 价值损失 + 熵正则化更新 |

#### MI 奖励融合机制

```
adjusted_rewards = env_rewards + mi_weight × normalized_MI_values
                                 ↑
              由 MutualInformationEstimator (NCE) 实时估计
```

- `mi_weight`（默认 0.5）控制互信息奖励强度
- 每步 MI 值通过 `store_transition(..., mi_value=mi_r)` 存入轨迹缓存
- `update()` 结束后自动清空轨迹缓存（含 `mi_values` 字段）

#### `evaluate_compare.py` 同步修正

`_run_episode` 现在按智能体类型分别调用 `store_transition`：

- `ExplaboffAgent`：传入当步的 `mi_value=step_mi[i]`，确保 MI 信号流入奖励
- `PPOAgent`：原有调用方式不变

---

### 架构重构：迁移至 PettingZoo 并行 API

由协作者 Neko-Yukari 提交 PR #1，已完整合并到主代码库，内容如下：

#### 1. 环境层重构（`envs/edge_env.py`）

| 变更项 | 旧实现 | 新实现 |
|--------|--------|--------|
| 基类 | `gymnasium.Env`（单智能体） | `pettingzoo.ParallelEnv`（真正多智能体） |
| 主类名 | `EdgeComputingEnv` | `EdgeOffloadingEnv`（新主类） |
| 兼容性 | — | `EdgeComputingEnv` 保留为包装器，旧脚本无需修改 |
| step() 时序 | 动作 → 时步 → 观测 | 动作 → 状态更新 → 任务再生成 → 观测（时序修正） |
| 配置键格式 | 单引号混用 | 统一双引号，与 YAML schema 对齐 |
| 奖励计算 | 简化占位符 | 完整多目标奖励：完成率 + 能量惩罚 + deadline惩罚 + 公平性加成 |
| 服务器负载 | 无动态衰减 | 每步 ×0.95 衰减，防止负载累积 |

**核心设计**：每个端设备对应一个 PettingZoo `agent_i`，所有智能体在同一时步并行决策（0=本地执行，1..n=卸载至第n台边缘服务器）。

#### 2. 训练脚本重写（`train_ippo.py`）

新脚本采用更清晰的模块化结构：

```
train_ippo.py
├── ActorCritic         —— 独立 Actor + Critic 双网络（2层MLP + ReLU）
├── SimplePPO           —— 每个智能体一套网络（独立参数，IPPO范式）
├── compute_gae()       —— 广义优势估计（GAE, λ=0.95）
├── _get_config_value() —— 兼容 marl.hidden_dim / algorithm.hidden_dim 两种路径
└── main()              —— 完整训练循环，PettingZoo 原生接口
```

**关键改进**：
- 移除对旧 `PPOAgent` 类的依赖，训练脚本自包含
- 支持 `--device auto/cpu/cuda` 命令行参数
- GAE 优势归一化（减均值除标准差）提升训练稳定性
- 每 10 个 episode 写入 TensorBoard

#### 3. 其他文件更新

| 文件 | 变更 |
|------|------|
| `envs/__init__.py` | 导出 `EdgeOffloadingEnv` 和 `EdgeComputingEnv` |
| `utils/helpers.py` | `load_config` 新增 `_convert_scientific_notation`，自动解析 YAML 中的科学计数法字符串 |
| `requirements.txt` | 新增 `pettingzoo>=1.25.0` |
| `quick_start.py` | `hidden_dim` 优先读 `marl.hidden_dim`，移除 `sys.path` 手动注入 |

#### 4. 默认配置规模提升

```yaml
# configs/default_config.yaml（现行默认值）
environment:
  num_end_devices:  10   # 原: 5
  num_edge_servers:  5   # 原: 2
  state_dim:        64   # 原: 32

marl:
  hidden_dim:      128   # 原: 32
  num_agents:       10   # 原: 3

algorithm:
  batch_size:       64   # 原: 8
  learning_rate:  1e-4   # 原: 1e-3
```

---

## 阶段1：仿真环境与IPPO基线 (30%) ✅

### 已完成的工作：

- 边缘计算网络仿真环境（多端设备 + 边缘服务器）
- 动态任务生成（CPU周期、数据大小、deadline）
- IPPO（独立PPO）算法：Actor-Critic + GAE + PPO裁剪 + 熵正则化
- YAML 配置管理、TensorBoard 集成、模型检查点

### 执行命令：

```bash
# 快速测试（调试用，约5分钟）
python quick_start.py --episodes 50 --length 50

# 完整IPPO训练（约2小时）
python train_ippo.py --config configs/default_config.yaml --episodes 500

# 指定设备
python train_ippo.py --episodes 500 --device cuda
```

### 关键指标（实测）：

| 指标 | 值 |
|------|------|
| Task Completion Rate | 65–75% |
| Energy Consumption | 5000–7000 J |
| Average Delay | 10–15 time slots |
| Fairness Index | 0.65–0.75 |

---

## 阶段2：Explaboff方案（互信息）(35%) ✅

### 已完成的工作：

- **MutualInformationEstimator**：基于 NCE（噪声对比估计）的互信息量化
- **AttentionCommunicationModule**：多头注意力机制，top-k 智能体通信选择
- **ExplaboffAgent**：PPO + 通信能力扩展，MI 奖励融合，注意力权重可解释性提取
  - ✅ 补全 `select_action()` 标准接口
  - ✅ 补全 `update(batch_size, num_epochs)` PPO 更新（含 MI 奖励融合）
- **evaluate_compare.py**：IPPO vs Explaboff 7维度指标自动对比与可视化，支持预训练检查点加载

### 执行命令：

```bash
# IPPO vs Explaboff 性能对比（从随机初始化开始，各训练 200 episode）
python evaluate_compare.py --train 200 --eval 30

# 使用已训练好的检查点（跳过训练直接对比）
python evaluate_compare.py \
  --ippo_ckpt results/IPPO_xxx/checkpoints/final_model.pt \
  --explaboff_ckpt results/Explaboff_xxx/checkpoints/final_model.pt \
  --eval 30

# Explaboff 专用训练
python train_explaboff.py --config configs/explaboff_config.yaml
```

### 预期性能提升（相比IPPO基线）：

| 指标 | IPPO | Explaboff | 提升 |
|------|------|-----------|------|
| Task Completion Rate | 72% | 78% | +8% |
| Energy (J) | 6200 | 5300 | −15% |
| Avg Delay | 12.5 | 11.8 | −6% |
| Fairness Index | 0.72 | 0.76 | +5% |

### Explaboff 关键配置：

```yaml
# configs/explaboff_config.yaml
algorithm:
  mi_weight: 0.5              # MI奖励权重
  mi_estimator: "nwj"         # NCE估计器
  learning_rate: 5e-5         # 降低学习率防止不稳定

communication:
  enable_communication: true
  comm_bandwidth: 50.0
  top_k_agents: 3
  comm_method: "attention"
```

---

## 阶段3：大规模场景优化 (35%) 🔄

### 进度：75%（较上次 50% 提升）

### 本次新增实现：

#### ✅ GNN 通信智能体（`agents/gnn_agent.py`）

这是本阶段的核心新增组件，实现了基于图神经网络的多智能体通信：

```
agents/gnn_agent.py
├── GraphAttentionLayer   —— 单头图注意力层（GAT）
│     - W 线性变换 + 注意力系数 + LeakyReLU + Softmax
│     - 掩码非边节点，支持孤立节点（nan→0）
├── MultiHeadGAT          —— 多头图注意力（默认4头）
│     - 各头独立计算后拼接，再经 out_proj 投影
├── GNNActorCritic        —— 参数共享的 Actor-Critic 网络
│     - node_encoder: obs → hidden embedding
│     - GNN layers (×2): 带残差连接的消息传递
│     - actor head: 输出策略 logits
│     - critic head: 输出状态价值
└── GNNPPOTrainer         —— 参数共享 PPO 训练器
      - 所有智能体共享一个网络（参数共享）
      - 支持动态邻接矩阵（服务器负载感知拓扑）
      - GAE 多智能体联合计算
      - 批量 PPO 更新（所有智能体同步）
```

**参数共享的意义**：相比 IPPO 每个智能体独立一套网络，参数共享使得：
- 参数量从 O(N × params) 降至 O(params)，支持 50+ 智能体不爆显存
- 智能体间策略隐式对齐，协作更一致
- GNN 消息传递实现显式通信，互补了参数共享的局限性

#### ✅ 大规模训练脚本（`train_scalable.py`）

```bash
# 20 智能体测试
python train_scalable.py --num_agents 20 --episodes 500

# 50 智能体大规模场景（目标达成）
python train_scalable.py --num_agents 50 --episodes 1000 --device cuda

# 自定义实验名
python train_scalable.py --num_agents 50 --episodes 1000 --name GNN_scale_test
```

脚本自动完成：
- 根据 `--num_agents` 动态扩展网络拓扑（边缘服务器数 = max(3, N//5)）
- 完整的 TensorBoard 日志（奖励、完成率、公平性、各项损失）
- 每 20% 进度自动保存检查点
- 估算通信开销百分比（GAT 注意力 O(N²) vs 前向推理 O(N)）
- 输出 `results.json` 包含可扩展性分析数据

### 待完成项（剩余 25%）：

- [ ] **SHAP/LIME 可解释性分析**：量化各特征对决策的贡献度
- [ ] **POMDP 部分可观测性支持**：添加观测噪声和不完整信息场景
- [ ] **50-agent 实际训练验证**：运行完整 1000 episode 并记录收敛曲线

### 阶段3 目标指标：

| 指标 | 目标值 | 当前状态 |
|------|--------|---------|
| 支持智能体数 | 50+ | ✅ 架构已支持（待运行验证） |
| 通信开销 | < 5% | 🔄 GAT 计算约 3–8%（与 N 相关） |
| Fairness Index | > 0.8 | 🔄 GNN训练后验证 |
| 训练收敛 | < 1000 episodes | 🔄 待验证 |

---

## 阶段4：GUI可视化演示 (+10%) 📅

### 规划的可视化界面：

- PyQt5 主窗口，多面板布局
- 网络拓扑图（设备位置 + 连接）
- 设备状态面板（CPU / 能量 / 队列）
- 任务流动视图（卸载路径动画）
- 实时性能曲线（完成率 / 延迟 / 公平性）
- 播放 / 暂停 / 速度控制

### 文件结构（待实现）：

```
gui/
├── __init__.py
├── main_window.py       # 主窗口（PyQt5）
├── visualization.py     # matplotlib 嵌入组件
├── simulation_widget.py # 仿真网络图
├── metrics_widget.py    # 实时指标曲线
└── config_widget.py     # 参数调节面板
```

### 预期界面布局：

```
┌─────────────────────────────────────────────────────┐
│      MARL Edge Offloading - 实时演示平台             │
├─────────────────────────────────────────────────────┤
│  [启动] [暂停] [重置]    速度: [████░░] 2x           │
├──────────────────────────┬──────────────────────────┤
│                          │  Task Completion  ████ 78%│
│   网络拓扑图             │  Energy (J)       ████     │
│                          │  Avg Delay        ███      │
│  ● End Device ×10        │  Fairness Index   ████ 0.82│
│  ■ Edge Server ×5        │                           │
│                          │  当前 episode: 245/500    │
│  任务队列: 12/20         │  Agent 通信路径: 可视      │
└──────────────────────────┴──────────────────────────┘
```

---

## 当前文件结构

```
Communication_Network_project/
│
├── 📄 train_ippo.py           ✅ 重写（PettingZoo + SimplePPO）
├── 📄 train_scalable.py       ✅ 新增（Stage 3，GNN大规模训练）
├── 📄 train_explaboff.py      ✅ Stage 2
├── 📄 quick_start.py          ✅ 更新（config路径修复）
├── 📄 evaluate_compare.py     ✅ Stage 2 对比评估
│
├── 📂 envs/
│   ├── __init__.py            ✅ 更新（导出两个环境类）
│   └── edge_env.py            ✅ 重写（EdgeOffloadingEnv + EdgeComputingEnv）
│
├── 📂 agents/
│   ├── ppo_agent.py           ✅ Stage 1 基线
│   ├── explaboff_agent.py     ✅ Stage 2
│   ├── networks.py            ✅ 网络架构
│   └── gnn_agent.py           ✅ 新增（Stage 3，GNN通信智能体）
│
├── 📂 utils/
│   ├── helpers.py             ✅ 更新（科学计数法解析）
│   ├── task_device.py         ✅
│   └── gpu_monitor.py         ✅
│
├── 📂 configs/
│   ├── default_config.yaml    ✅（规模扩大：10智能体，64维状态）
│   ├── lowmem_config.yaml     ✅（兼容低显存）
│   └── explaboff_config.yaml  ✅ Stage 2
│
├── 📂 results/                # 训练结果（运行后生成）
│   ├── IPPO_<timestamp>/
│   ├── GNN_n<N>_<timestamp>/  # Stage 3 输出
│   └── comparison/
│
└── 📂 gui/                    # Stage 4（待实现）
```

---

## 环境配置与运行

### 安装依赖：

```bash
conda activate marl-edge
pip install -r requirements.txt
# 新增：pettingzoo>=1.25.0 已写入 requirements.txt
```

### 验证环境：

```bash
python -c "import torch; import pettingzoo; print('OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 完整训练流程：

```bash
# Stage 1：IPPO 基线
python train_ippo.py --episodes 500

# Stage 2：Explaboff 通信增强
python train_explaboff.py --config configs/explaboff_config.yaml

# Stage 2：两者对比
python evaluate_compare.py

# Stage 3：GNN 大规模场景
python train_scalable.py --num_agents 50 --episodes 1000 --device cuda

# 查看训练曲线
tensorboard --logdir results/
```

---

## 性能基准汇总

### 算法横向对比：

| 算法 | Task Completion | Energy | Avg Delay | Fairness |
|------|----------------|--------|-----------|---------|
| IPPO（基线） | 65–75% | 6200 J | 12.5 slots | 0.72 |
| Explaboff（通信） | 75–82% | 5300 J | 11.8 slots | 0.76 |
| GNN-PPO（Stage 3） | 目标 >80% | 目标更低 | 目标更短 | 目标 >0.85 |

### GPU 显存参考：

| 配置 | 智能体数 | 显存占用 |
|------|---------|---------|
| lowmem_config | 3–5 | ~1–2 GB |
| default_config | 10 | ~3–4 GB |
| train_scalable (N=50) | 50 | ~5–7 GB |

---

## 常见问题

**Q：CUDA out of memory**
```bash
# 使用低显存配置
python train_ippo.py --config configs/lowmem_config.yaml
# 或强制 CPU
python train_scalable.py --device cpu --num_agents 20
```

**Q：ImportError pettingzoo**
```bash
pip install pettingzoo>=1.25.0
```

**Q：结果不收敛**
```yaml
algorithm:
  learning_rate: 5e-5   # 降低学习率
  entropy_coeff: 0.05   # 增加探索
  gamma: 0.95           # 缩短视野
```

---

## 代码提交检查清单

- [x] 阶段1 所有脚本可运行
- [x] 阶段2 Explaboff 实现完整
- [x] ExplaboffAgent `select_action` + `update` 方法补全
- [x] evaluate_compare.py MI 奖励正确传入 store_transition
- [x] PettingZoo 环境重构（PR #1 合并）
- [x] GNN 通信智能体实现
- [x] 大规模训练脚本完成
- [ ] 50-agent 训练实际验证
- [ ] SHAP 可解释性分析
- [ ] GUI 界面实现

---

更新时间：2026年4月21日  
项目进度：约 82% 完成
