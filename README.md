## 周期性动力学的硬约束 Koopman Operator（项目概览）

本仓库实现并实验了一类针对周期性（periodic）动力学的“硬约束（hard-constraint）”Koopman Operator 模型。核心思想是：在可学习的字典空间（trainable dictionary）上构造一个满足周期性结构的 Koopman 线性算子（以若干 2×2 旋转块构成的块对角矩阵），把周期性动力学的频率/相位信息以参数化角度的形式硬编码到算子结构中，从而保证预测时的谱特性（模长为 1 的旋转子空间）与物理先验一致。

以下为 README 的要点：

### 为什么要这样做
- 许多物理系统（例如含电感、电容或其他振荡源的电路场）天然具有周期性或近周期性轨道。使用通用的可学习矩阵容易产生发散或非物理谱。通过把 Koopman 算子约束为由 2×2 旋转块（cos/sin）构成的块对角形式，可以“硬约束”其谱项为复数单元模 1，从而在结构上保证周期性行为。

### 方法概览（高层）
- 状态首先通过可训练字典（`TrainableDictionary`，位于 `src/resnet.py`）编码到较高维的字典空间。
- 在字典空间上，使用参数化网络（ResNet）把系统参数（例如介质参数、驱动频率等）映射到 Koopman 算子的参数：
  - 对于周期性分量，网络输出角度（angle）或对应的 cos/sin，进而组成每个 2×2 旋转块（保证模长为 1）。
  - 可选地同时输出变换矩阵 V（将状态映射到字典空间的线性变换）以及输入耦合矩阵 B。
- 预测过程在字典空间上迭代：x_dic_{t+1} = x_dic_t * K + u_dic_t * B（代码中以右乘张量形式实现）。
- 损失由字典空间上的 MSE 与对 V 条件数/稳定性的正则化项组成（代码中有针对 V 的近似逆范数/条件数的约束）。

### 主要实现文件（简要说明）
- `src/param_periodic_koopman.py`：实现了多个参数化模块：
  - `ParamBlockDiagonalMatrix` / `ParamBlockDiagonalMatrixWoV`：由 ResNet 输出构造 2×2 旋转块组成的 K（和可选的 V）。这是实现“硬约束”周期性的核心模块。
  - `ParamBlockDiagonalKoopmanWithInputs`：端到端模块，把字典、K（包含 V）和 B 结合用于有输入的系统预测。
  - `ParamBlockDiagonalKoopmanWithInputs3NetWorks`：把 Lambda（旋转块）和 V 分成三个网络独立预测的变体，便于更灵活的训练/正则化。
  - 还有若干用于生成特殊正交矩阵、斜对称矩阵等的辅助模块（`ParamSkewSymmetricMatrix`/`ParamSpecialOrthogonalMatrix` 等）。
- `src/periodic_koopman.py`：包含更简单的 block-diagonal Koopman 基类与时间嵌入示例（`BlockDiagonalKoopman`, `TimeEmbeddingBlockDiagonalKoopman`），用于说明单独角度参数化时的构造。
- `src/train_param_periodic.py`：训练主流程。实现了数据加载、训练/测试循环、损失计算（MSE + V 的正则化），并写入 `save_dir` 下的模型/日志/损失。
- `src/args.py`：命令行参数与 YAML 配置读取工具（`--config` 指定 YAML 配置）。
- `src/resnet.py`：实现字典编码器 `TrainableDictionary` 与 ResNet/MLP 网络（用于把参数映射到矩阵条目或角度）。
- `src/data.py`：数据集读取与 DataLoader 封装（实验数据位于 `data/` 目录）。

### 仓库结构（重要目录）
- `configs/`：不同实验的 YAML 配置文件（训练超参、数据路径、网络维度等）。
- `data/`：样本数据（numpy 文件），例如不同材料参数、不同驱动频率的样本集合。文件名通常包含参数信息，便于追踪。
- `src/`：源码实现（模型、训练、数据、工具等）。
- `results/`：训练结果、保存的模型与生成的输出示例。

### 如何运行（示例）
1. 安装依赖（示例）

```bash
# 建议使用合适的 CUDA / CPU 版本的 torch
pip install torch numpy pyyaml tqdm
```

2. 用已有配置训练（示例）

```bash
python src/train_param_periodic.py --config configs/experiment_1.yaml
```

训练时的主要配置项位于对应 YAML 文件中（例如 `save_dir`, `data_dir`, `dictionary_dim`, `A_layers`, `B_layers`, `u_layers`, `epochs`, `lr`, `sample_step` 等）。训练输出包含：
- `save_dir/model_state_dict.pth`：模型参数。
- `save_dir/dataset.pth`、`save_dir/losses.pth`、`save_dir/log.txt` 等。

### 实验/损失细节
- 训练目标是在字典空间上最小化预测序列与真实字典投影的 MSE。代码中通过 `dictionary_V` 将原始状态映射到字典空间并与由 Koopman 算子推进的预测对齐。
- 为避免 V 跨训练步数的病态（条件数过大导致数值不稳定），实现了对 V 的近似逆范数约束（regularization_loss），将该项加入总损失作为软约束。

### 设计上的关键点与假设
- 硬约束 K 的形式（由 2×2 旋转子矩阵组成）直接将周期性（以角速度/相位）作为网络输出的目标；这样可以减少对训练数据的依赖并提升物理一致性。
- 这个实现假设系统中周期性自由度能在字典空间被分离为若干相互独立的 2 维子空间（2×2 旋转块）。若真实系统存在强耦合或非周期分量，需要在模型架构（例如增加上三角耦合项或非线性还原）上做扩展。

### 可扩展方向 / 建议的后续工作
- 将字典 V 的学习与正则化策略进一步改进，比如显式最小化 V 的条件数或通过谱范数约束。
- 支持连续频率估计与谱分解的可解释性工具（绘制 Koopman 特征频谱、相位轨迹等）。
- 把模型封装为可重用的评估脚本（`src/eval_param_periodic.py` 已有雏形），用于对比不同约束（无约束、硬约束、软约束）下的泛化能力。

### 参考与致谢
- 该仓库结合了基于 Koopman 理论的动力学建模与可训练字典/深度网络来参数化线性算子。若你在科研/工程复现中使用或引用，请在论文中注明相应实现细节或联系作者获取更多说明。

---

如果你希望我把 README 翻译成英文版、补充配置文件注释模版（如注释 `configs/experiment_1.yaml` 中每一项的含义），或者自动生成一个 `requirements.txt`，我可以继续帮你完成。
