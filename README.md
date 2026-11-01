# CPPN 表示实验：FER 指标与稀疏对照

这是一个玩具实验：能否量化 FER 论文（Kumar, Clune, Lehman, Stanley，[arXiv:2505.11581](https://arxiv.org/abs/2505.11581)）中 Picbreeder 与 SGD 网络的内部表示差异，并找出差异来自稀疏、接线还是权重的找法。长期目标是把这些指标用到悬臂梁拓扑优化，再到具身任务；这两步都还没有开始。

## 当前结论

- 论文的定性现象复现了：输出几乎相同的两个网络，内部表示差别很大。
- 差距的来源**还没有回答**，主要因为稀疏网络在 5 万步内大多达不到拟合门槛（MSE < 0.01）。
- Picbreeder 的接线本身会强制对称：skull、butterfly、apple 分别有 82%、59%、38% 的不同特征图，无论权重取什么都对称。
- apple 上 M3a、M4a 的差距随拟合水平变化：稠密网络在 MSE 约 0.0065 提前停止时，这两个指标已接近 Picbreeder。butterfly 上差距不受此影响。

每条结论的状态以 `kb/kb_claims.json` 为准，可以用下面的命令重放。

## 目录

| 路径 | 内容 |
|---|---|
| `phase0/cppn_torch.py` | 解析 XML 基因组，转换成稠密网络，以及 `DenseCPPN` |
| `phase0/train_sgd.py` | 稠密 SGD 模仿训练（G0） |
| `phase0/train_early.py` | 稠密网络在目标 MSE 提前停止（G0e） |
| `phase0/train_control.py` | 稀疏对照：G1a、G1b、G4 |
| `phase0/metrics.py`、`phase0/eval_controls.py` | 指标定义，以及把所有网络测量后写入数据库 |
| `phase0/calibrate.py`、`phase0/figures.py` | 第 0 阶段校准和图 |
| `phase0/runs/`、`phase0/out/` | 训练好的权重、校准结果和图 |
| `kb/experiments_db.jsonl` | 每个网络、每个 seed 一行的测量表 |
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

`run_controls.sh` 需要 `phase0/runs/sgd_*.pt` 已存在；它们由 `python train_sgd.py --name <skull|butterfly|apple> --iters 50000 --res 128` 生成，apple 还要加 `--lr 3e-4`。`calibrate.py` 和 `figures.py` 从 `phase0/` 运行。

## 局限

- 只有 3 张图，每组只有几个 seed。这是玩具证据，不能外推到其他网络或任务。
- 拟合门槛 MSE < 0.01 比稠密网络的约 5e-4 宽松很多。比较时要匹配拟合水平，并看实际 MSE。
- `commits.jsonl` 的哈希链只能发现事后改动，证明不了写入时间。C-001 到 C-006 是在这个仓库建立之前写的；C-007 到 C-010 在首次推送时已存在，而结果在其后的提交里。
- C-006 是看到 G4 结果后才加的，C-007 是事后描述。两者都不算独立的预注册检验。

## 第三方内容

本仓库不包含第三方代码或论文 PDF。基因组由 `setup_refs.sh` 从 [akarshkumar0101/fer](https://github.com/akarshkumar0101/fer)（Apache-2.0）拉取。`phase0/cppn_torch.py` 是按该仓库的约定用 PyTorch 重新实现的，没有加载它的 pickle 文件或运行它的代码。
