# MARL Edge Offloading - 快速参考指南

## 🚀 快速启动 (5分钟)

```bash
# 1. 激活Conda环境
conda activate marl-edge

# 2. 进入项目目录
cd d:\Code\Communication_Network_project

# 3. 运行环境测试
python test_env.py

# 4. 快速训练验证
python quick_start.py --episodes 50
```

## 📝 常用命令

### 环境管理
```bash
# 创建环境（首次）
conda create -n marl-edge python=3.10 -y

# 激活环境
conda activate marl-edge

# 查看已安装包
pip list

# 更新依赖
pip install -r requirements.txt --upgrade

# 删除环境
conda remove --name marl-edge --all
```

### 运行训练

#### IPPO基线 (阶段1)
```bash
# 默认配置训练
python train_ippo.py

# 自定义episode数量
python train_ippo.py --episodes 1000

# 自定义实验名称
python train_ippo.py --name my_ippo_experiment

# 使用特定配置
python train_ippo.py --config configs/default_config.yaml --episodes 500
```

#### 快速测试
```bash
# 50个episodes快速训练（调试）
python quick_start.py --episodes 50 --length 50

# 100个episodes
python quick_start.py --episodes 100 --length 100
```

#### Explaboff对比 (阶段2)
```bash
# 运行IPPO vs Explaboff对比
python evaluate_compare.py

# 查看结果
cat results/comparison/comparison_results.json
```

### 查看结果

```bash
# 列出所有训练结果
dir results/

# 查看特定实验的指标
cat results/IPPO_20260412_143022/results.json

# 启动TensorBoard查看训练过程
tensorboard --logdir=./logs

# 访问 http://localhost:6006
```

## 📊 代码结构速览

```
核心代码文件
├── envs/edge_env.py           # 仿真环境
├── agents/ppo_agent.py        # IPPO智能体
├── agents/explaboff_agent.py  # Explaboff方案
├── agents/networks.py         # 神经网络架构
└── utils/*.py                 # 辅助函数

训练脚本
├── train_ippo.py              # 完整IPPO训练
├── quick_start.py             # 快速启动
├── test_env.py                # 环境测试
└── evaluate_compare.py        # 对比评估

配置文件
├── configs/default_config.yaml     # 默认配置
└── configs/explaboff_config.yaml   # Explaboff配置
```

## ⚙️ 配置参数速览

### 环境参数
```yaml
environment:
  num_edge_servers: 5          # 边缘服务器数
  num_end_devices: 10          # 端设备数
  num_tasks: 20                # 任务数
  task_cpu_min: 100            # 任务CPU周期最小值
  task_cpu_max: 500            # 任务CPU周期最大值
```

### MARL参数
```yaml
marl:
  algorithm: "IPPO"            # 算法选择
  total_episodes: 10000        # 总episodes
  episode_length: 100          # 每个episode长度
  num_agents: 10               # 智能体数
```

### 算法参数
```yaml
algorithm:
  learning_rate: 1e-4          # 学习率
  gamma: 0.99                  # 折扣因子
  gae_lambda: 0.95             # GAE参数
  clip_ratio: 0.2              # PPO裁剪比率
  entropy_coeff: 0.01          # 熵系数
  batch_size: 64               # 批次大小
```

### 奖励权重
```yaml
reward:
  task_completion_weight: 0.4      # 任务完成
  energy_efficiency_weight: 0.3    # 能源效率
  deadline_penalty_weight: 0.2     # deadline惩罚
  fairness_weight: 0.1             # 公平性
```

## 🎯 关键指标

| 指标 | 含义 | 目标值 |
|------|------|--------|
| Task Completion Rate | 完成的任务比例 | >75% |
| Energy Consumption | 总能量消耗 | 5000-7000 J |
| Average Delay | 平均任务延迟 | <15 time slots |
| Fairness Index | Jain公平性指数 | >0.7 |
| Deadline Miss Rate | 未按期限完成比例 | <25% |

## 📈 性能基准

### IPPO (阶段1)
```
Task Completion Rate:    70-75%
Energy Consumption:      6000-7000 J
Average Delay:           12-15 slots
Fairness Index:          0.68-0.75
```

### Explaboff (阶段2，预期)
```
+Task Completion:        +5-8%
-Energy Consumption:     -10-15%
-Average Delay:          -5-8%
+Fairness Index:         +3-5%
+Mutual Information:     量化通信效果
```

## 🔧 故障排除

### 问题：ImportError: No module named 'torch'
```bash
# 确保环境已激活
conda activate marl-edge

# 重新安装依赖
pip install -r requirements.txt
```

### 问题：运行速度慢
```bash
# 选项1：减少episode数量
python quick_start.py --episodes 20

# 选项2：修改配置减小网络
# configs/default_config.yaml:
algorithm:
  hidden_dim: 64  # 减小网络大小

# 选项3：减少智能体数量
environment:
  num_agents: 5
```

### 问题：内存不足
```bash
# 减小批次大小
algorithm:
  batch_size: 32

# 或减小episode长度
marl:
  episode_length: 50
```

### 问题：结果不收敛
```bash
# 降低学习率
algorithm:
  learning_rate: 5e-5

# 增加熵系数
algorithm:
  entropy_coeff: 0.05
```

## 📂 生成的文件

运行后会在 `results/` 目录生成：

```
results/
├── IPPO_<date>_<time>/
│   ├── config.yaml              # 配置文件
│   ├── results.json             # 最终指标
│   ├── results.txt              # 易读格式
│   ├── checkpoints/
│   │   └── final_model.pt       # 最终模型
│   └── logs/                    # TensorBoard日志
│
├── comparison/                  # Explaboff对比结果
│   ├── comparison_results.json
│   └── comparison_plots.png
│
└── quick_test/                  # 快速测试结果
    └── results.json
```

## 🖥️ GPU加速（可选）

### 检查GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 安装GPU版PyTorch
```bash
# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或使用CPU版本（默认）
pip install torch torchvision torchaudio
```

## 📊 分析结果

### 查看JSON结果
```bash
# Windows PowerShell
cat results/IPPO_20260412_143022/results.json | ConvertFrom-Json

# 或使用Python
python -m json.tool results/IPPO_20260412_143022/results.json
```

### 绘制对比图
```bash
# 查看对比图
start results/comparison/comparison_plots.png

# 或在Python中
from PIL import Image
img = Image.open('results/comparison/comparison_plots.png')
img.show()
```

## 💻 系统要求

| 项 | 需求 |
|----|------|
| Python | ≥3.8, 推荐3.10 |
| RAM | ≥8GB（16GB推荐） |
| 存储 | ≥2GB |
| GPU | 可选（CUDA 11.8+） |
| OS | Windows/Linux/macOS |

## 🔗 有用的链接

- **项目主文档**：README.md
- **阶段执行指南**：PROJECT_STAGES.md
- **完成情况报告**：COMPLETION_REPORT.md
- **PyTorch文档**：https://pytorch.org/docs
- **Gymnasium文档**：https://gymnasium.farama.org

## 📞 获取帮助

1. 查看 `README.md` 了解项目概述
2. 查看 `PROJECT_STAGES.md` 了解详细执行步骤
3. 运行 `python test_env.py` 验证环境
4. 查看错误堆栈跟踪了解具体问题
5. 修改 `configs/` 中的参数进行实验

## ✅ 完成清单

运行完整训练流程的检查清单：

- [ ] 环境已配置 (`conda activate marl-edge`)
- [ ] 依赖已安装 (`pip install -r requirements.txt`)
- [ ] 环境测试通过 (`python test_env.py`)
- [ ] 快速训练运行 (`python quick_start.py`)
- [ ] IPPO基线训练 (`python train_ippo.py`)
- [ ] 结果已生成 (`check results/`)
- [ ] 对比评估运行 (`python evaluate_compare.py`)
- [ ] 所有结果已保存

## 🎓 学习路径

1. **理解环境**：阅读 `envs/edge_env.py`
2. **学习算法**：阅读 `agents/ppo_agent.py`
3. **运行测试**：`python test_env.py`
4. **快速训练**：`python quick_start.py`
5. **完整训练**：`python train_ippo.py`
6. **对比分析**：`python evaluate_compare.py`
7. **自定义修改**：编辑配置和参数

---

**最后更新**：2026年4月12日  
**项目进度**：70% (Stage 1-2完成)
