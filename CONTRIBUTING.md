# 贡献指南

感谢您对 MARL Edge Offloading 项目的兴趣！我们欢迎各种形式的贡献，包括但不限于：

- 🐛 报告bug
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码修复
- 🎨 改进UI/UX
- 📊 添加实验结果

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone https://github.com/your-username/marl-edge-offloading.git
cd marl-edge-offloading

# 创建开发环境
conda create -n marl-edge-dev python=3.10 -y
conda activate marl-edge-dev

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 开发工具
```

### 2. 验证设置

```bash
# 运行基本测试
python test_env.py

# 运行代码质量检查
python -m black --check .
python -m flake8 .
python -m mypy .
```

## 📋 开发工作流

### 1. 创建特性分支

```bash
# 从main分支创建新分支
git checkout -b feature/your-feature-name

# 或者修复bug
git checkout -b fix/issue-number-description
```

### 2. 编写代码

- 遵循 [PEP 8](https://pep8.org/) 代码风格
- 添加类型提示
- 编写完整的docstring
- 保持代码简洁且可读

### 3. 编写测试

```bash
# 创建测试文件
touch tests/test_your_feature.py

# 运行测试
python -m pytest tests/test_your_feature.py -v

# 运行所有测试
python -m pytest tests/ -v
```

### 4. 提交更改

```bash
# 添加更改的文件
git add .

# 提交（使用规范的提交信息）
git commit -m "Add: 简短描述新功能

详细说明更改内容和原因

相关问题: #123"
```

### 5. 推送和创建PR

```bash
# 推送分支
git push origin feature/your-feature-name

# 在GitHub上创建Pull Request
```

## 📝 提交信息规范

我们使用以下格式的提交信息：

```
类型: 简短描述

详细说明...

相关问题: #123
```

### 类型包括：

- `Add`: 新功能
- `Fix`: 修复bug
- `Update`: 更新现有功能
- `Remove`: 删除功能
- `Refactor`: 重构代码
- `Docs`: 文档更新
- `Test`: 测试相关
- `Style`: 代码风格调整

### 示例：

```
Add: 实现GPU显存监控工具

- 添加GPUMemoryMonitor类
- 支持实时显存使用情况监控
- 集成到训练脚本中

相关问题: #42
```

## 🧪 测试要求

### 单元测试
- 所有新功能都需要对应的单元测试
- 测试覆盖率应至少达到80%
- 使用 `pytest` 框架

### 集成测试
- 关键功能需要集成测试
- 测试完整的训练流程
- 验证模型保存和加载

## 📚 代码质量标准

### Python代码规范

```python
# ✅ 好的示例
def calculate_reward(state: dict, action: int) -> float:
    """
    Calculate reward based on state and action.

    Args:
        state: Current environment state
        action: Action taken by agent

    Returns:
        Reward value
    """
    reward = 0.0

    # 计算任务完成奖励
    if state['task_completed']:
        reward += 1.0

    # 计算能量惩罚
    energy_cost = state['energy_consumption']
    reward -= 0.01 * energy_cost

    return reward

# ❌ 不好的示例
def calc_rwd(s,a): # 缺少类型提示和文档
    r=0
    if s['task_completed']:r+=1
    r-=0.01*s['energy_consumption']
    return r
```

### 文档要求

- 所有公共函数和类需要docstring
- 使用Google风格的docstring格式
- 包含参数类型和返回值说明
- 提供使用示例

## 🔧 开发工具

### 代码格式化
```bash
# 自动格式化代码
python -m black .

# 检查代码风格
python -m flake8 .

# 类型检查
python -m mypy .
```

### 性能分析
```bash
# 运行性能测试
python -m pytest tests/ --profile

# 内存分析
python -m memory_profiler your_script.py
```

## 🐛 报告问题

### Bug报告
请使用GitHub Issues报告bug，包含：

- 详细的错误描述
- 重现步骤
- 期望的行为
- 实际的行为
- 系统信息（Python版本、PyTorch版本等）

### 功能请求
对于新功能请求，请提供：

- 功能描述
- 使用场景
- 预期效果
- 可能的实现方式

## 📞 联系方式

- **项目维护者**: [您的名字]
- **邮箱**: your.email@example.com
- **讨论区**: [GitHub Discussions](https://github.com/your-username/marl-edge-offloading/discussions)

## 🙏 贡献者公约

通过贡献代码，您同意遵守我们的行为准则：

- 尊重所有贡献者
- 提供建设性的反馈
- 接受维护者的决定
- 专注于项目目标

感谢您的贡献！ 🎉