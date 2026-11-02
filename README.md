# CPPN 表示实验：FER 指标与稀疏对照

这是一个玩具实验：能否量化 FER 论文（Kumar, Clune, Lehman, Stanley，[arXiv:2505.11581](https://arxiv.org/abs/2505.11581)）中 Picbreeder 与 SGD 网络的内部表示差异，并找出差异来自稀疏、接线还是权重的找法。长期目标是把这些指标用到悬臂梁拓扑优化，再到具身任务；这两步都还没有开始。

## 当前结论

- 论文的定性现象复现了：输出几乎相同的两个网络，内部表示差别很大。
- 差距的来源**还没有回答**，主要因为稀疏网络在 5 万步内大多达不到拟合门槛（MSE < 0.01）。这也不是进化接线独有的问题：SGD 网络剪枝后同样难以达标（`phase0/out/report_sparse_fit.png`）。
- Picbreeder 的接线本身会强制对称：skull、butterfly、apple 分别有 82%、59%、38% 的不同特征图，无论权重取什么都对称（`report_wiring_forced.png`）。
- apple 上 M3a、M4a 的差距随拟合水平变化：稠密网络在 MSE 约 0.0065 提前停止时，这两个指标已接近 Picbreeder。butterfly 上差距不受此影响（`report_fit_level.png`）。
- **持续学习**：预注册的 C-011 到 C-018 全部证伪，“Picbreeder 起点更利于学习小改动”失去主要支撑。在 2000 步内，它在 skull 和 butterfly 上几乎学不会，稠密网络全部学会（`report_continual.png`）。例外是 apple 的局部改动：训练到底的稠密网络同样很少达标（2/18），最终误差和 Picbreeder 接近。图中提前停止组（G0e）的高适应率不可比：适应标准含网络自身的旧误差，G0e 的门槛因此宽松得多。apple 上它的最终误差（0.247 E0）其实高于训练到底的 G0（0.219 E0），所以这不说明“越收敛越改不动”。分开看，局部不对称是 0.253 对 0.217，局部对称反过来是 0.177 对 0.220，两者没有稳定的高低。
- **重测（C-019 到 C-021，看过前面的结果后注册，不是盲测）**：在固定 2000 步看改动区域内的剩余误差，skull 和 butterfly 上 Picbreeder 是稠密网络的 4～61 倍，降低 10 倍学习率后差距仍在。apple 的局部改动对两者都难，都卡在约 0.2。
- **区域外的破坏（C-022，事后描述，不算独立证据）**：局部改动 2000 步后，全部权重可学的 Picbreeder 在区域外的变动是稠密网络的 2～32 倍（`report_tradeoff.png`）。skull 和 butterfly 上它的区域内误差也更高；apple 上相近，局部对称时还有更低的情况，属于用更大变动换更低误差。这和“好表示破坏更小”的预期相反，但只描述这 3 个网络和这些改动。
- **连续改动（问题 1，跑在 kb-loop 分支上，见 `kbloop_q1/`）**：无回放时旧改动的遗忘量约 0.2，100% 回放时约 0.01，5% 回放几乎不改变遗忘量；区域外的背景漂移则对少量回放很敏感（事后才看的指标）。
- **减少遗忘的方法（问题 2，见 `kbloop_q2/`）**：同样 5% 的预算，把一半放到旧改动上，遗忘量降到 0.19～0.24 倍，只看没被标注的像素也有 0.26～0.32 倍。预测写于看过问题 1 之后，只在 A100 上跑了一遍。
- **Picbreeder 的连续改动（问题 3，见 `kbloop_q3/`）**：无回放时，Picbreeder 的遗忘量是稠密网络的 0.91～1.69 倍（保留梯度）或 1.28～1.34 倍（原始输出），预测“不超过一半”被证伪。只描述这 3 个网络和局部颜色反转；“旋钮”改动没有测。聚焦回放对它同样有效。
- **不存标签的聚焦回放（问题 4，见 `kbloop_q4/`）**：用网络自己的旧输出当标签，遗忘量是均匀存储回放的 0.12～0.22 倍，只看未标注像素是 0.18～0.25 倍。代价是背景漂移增加到 1.8～3.8 倍，并且仍需旧改动的位置，以及阶段开始时回放点上的输出（5% 预算下约 200 个值，阶段结束后可丢弃）。
- **旋钮改动（问题 5，见 `kbloop_q5/`）**：把一个权重改变 1 确实能得到有语义的局部改动，比如 skull 张嘴。但用梯度学习这些改动时，Picbreeder 没有学得更好（误差是稠密网络的 10～25 倍），也没有忘得更少（0.63～1.61 倍）。这轮不能说明变异或搜索用不上这些旋钮。
- **地址几何与知识复利（问题 6，见 `kbloop_q6/`）**：玩具实验没有验证成功。打乱图块后共享图样的成本比值是 1.65，原始坐标是 0.87，方向支持几何重要，但只有 3 个网络；没有可复用结构的组比值也下降到 0.57，所以比值下降不等于复利。
- **CPPN 当解题循环的数值记忆（离线，见 `aiq_memory_probe/`）**：在 30 个 ArchitectureIQ 实例上，CPPN 预测新实例的准确率高于最近邻（0.851 对 0.748），但低于线性回归（0.895）；不复习时会遗忘，8 个样本的集中复习没有帮助，三种不会遗忘的记忆都没有显示复利。
- **保留梯度不是通用修复**：它只在单个改动加 100% 回放时消除输出层陷阱。在 apple 的连续改动中，它让卡在灰色的比例下降，但误差中位数升高（`kbloop_q3/README.md`）。

每条结论的状态以 `kb/kb_claims.json` 为准，可以用下面的命令重放。

## 目录

| 路径 | 内容 |
|---|---|
| `phase0/cppn_torch.py` | 解析 XML 基因组，转换成稠密网络，以及 `DenseCPPN` |
| `phase0/train_sgd.py` | 稠密 SGD 模仿训练（G0） |
| `phase0/train_early.py` | 稠密网络在目标 MSE 提前停止（G0e） |
| `phase0/train_control.py` | 稀疏对照：G1a、G1b、G4 |
| `phase0/metrics.py`、`phase0/eval_controls.py` | 指标定义，以及把所有网络测量后写入数据库 |
| `phase0/continual.py` | 持续学习测试：已拟合的网络再学 12 种小改动 |
| `phase0/continual_v2.py`、`kb/continual_db_v2.jsonl` | 重测：固定步数记录改动区域的剩余误差和区域外变动，指标不依赖被测网络 |
| `phase0/calibrate.py`、`phase0/figures.py` | 第 0 阶段校准和图 |
| `report_figures/` | 汇报用的结果图和逐张解释，以及和 CPPN、kb-loop 的关系 |
| `phase0/report_figures.py` | 从测量表生成汇报图（`phase0/out/report_*.png`） |
| `phase0/runs/`、`phase0/out/` | 训练好的权重、校准结果和图 |
| `kb/experiments_db.jsonl` | 每个网络、每个 seed 一行的测量表 |
| `kb/continual_db.jsonl` | 持续学习测试的结果，每个网络、每种改动、每档学习率一行 |
| `kb/kb_claims.json`、`kb/commits.jsonl` | 可执行 claim 和哈希链日志 |
| `kb/verify_kb.py` | 校验器，只读表计算，不需要 GPU |
| `kb/prereg_*.py` | 生成预注册 claim 的脚本 |

## 实验分组

| 组 | 接线 | 权重 |
|---|---|---|
| PB | Picbreeder | 进化得到 |
| G0 | 全连接 | SGD，训练到底 |
| G0e | 全连接 | SGD，MSE ≤ 0.0065 时停止 |
| G1a | Picbreeder | SGD，只训练 NEAT 连接，复制边固定为 1 |
| G1b | Picbreeder | SGD，所有现存边重新初始化并训练 |
| G4 | 从 G0 剪枝到相同总边数 | SGD 微调 |

## 复现

只重放判定（无需 GPU，只用标准库）：

```bash
cd kb
python test_verify_kb.py
python verify_kb.py
```

重新训练需要 NVIDIA GPU，以及 torch、numpy、scipy、matplotlib。代码在 RTX 4060 8GB、torch 2.10+cu128 上运行过：

```bash
sh setup_refs.sh
cd phase0
sh run_controls.sh
sh run_b6.sh
```

持续学习测试在此之后运行 `python continual.py`，它会跳过 `kb/continual_db.jsonl` 里已完成的组，要重跑须先删除该文件。

`run_controls.sh` 需要 `phase0/runs/sgd_*.pt` 已存在；它们由 `python train_sgd.py --name <skull|butterfly|apple> --iters 50000 --res 128` 生成，apple 还要加 `--lr 3e-4`。`calibrate.py` 和 `figures.py` 从 `phase0/` 运行。

## 局限

- 只有 3 张图，每组只有几个 seed。这是玩具证据，不能外推到其他网络或任务。
- 拟合门槛 MSE < 0.01 比稠密网络的约 5e-4 宽松很多。比较时要匹配拟合水平，并看实际 MSE。
- `commits.jsonl` 的哈希链只能发现事后改动，证明不了写入时间。C-001 到 C-006 是在这个仓库建立之前写的；C-007 到 C-010 在首次推送时已存在，而结果在其后的提交里。
- C-006 是看到 G4 结果后才加的，C-007 是事后描述。两者都不算独立的预注册检验。
- C-011 到 C-018 注册前看过 30 步冒烟测试；PBa 组是看到其中 PBd 的 MSE 上升后才加的。首次正式运行因并发写入被清除并重跑，清除前看到过少量 skull 的适应计数。
- 适应步数在未适应时右删失于 2000 步，所以 A1c 的比值只是下界，不是测得的比值；两组都删失时比值 1.0 判为 PASS 没有意义（C-011、C-012、C-015、C-016 的 apple 项）。这些 claim 的证伪依据是其他任务的下界，结论不变。
- G0e 不能和其他组直接比较：它的适应门槛含自身旧误差；它在改动外还在继续拟合原图，所以 F 也偏高。`continual.py` 里“F 不受拟合水平影响”的假设不成立。
- 持续学习的比较里，每张图只有一个 Picbreeder 网络，同一网络的多个改动不是独立样本，所以比例和置信区间都不能当作对 Picbreeder 网络总体的推断；相同学习率对权重更大的 Picbreeder 是更小的相对步长；颜色反转需要新增局部特征，可能不利于它。这些改动没有包含论文里张嘴、眨眼那种单权重可达的“旋钮”改动。

## 第三方内容

本仓库不包含第三方代码或论文 PDF。基因组由 `setup_refs.sh` 从 [akarshkumar0101/fer](https://github.com/akarshkumar0101/fer)（Apache-2.0）拉取。`phase0/cppn_torch.py` 是按该仓库的约定用 PyTorch 重新实现的，没有加载它的 pickle 文件或运行它的代码。
