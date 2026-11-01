# 减少遗忘的方法（问题 2）

和问题 1 相同的连续改动实验，跑在 kb-loop 的 `kb_lab.run_job` 入口上（`science-of-ai-prototype-cppn` 的 `fcb1185`），只用 A100，没有调用 LLM。预测 K9101～K9107 在 `research/cppn_replay_methods_20261101/seed_kb.json`，写于看过问题 1 之后，不是盲测。

## 运行情况

51 个实验跑成 49 个。缺的 2 个是 apple 的 `m0.05`：TV 0.05 组 apple 有 2 个 seed 拟合失败（MSE 约 0.043），中位数因此过高，和它配对的普通网络也在这个水平停下，全部达不到拟合门槛（MSE < 0.01），所以没有可用网络，`fitted_dense` 报错，运行三次重试后退出（见 `run.log`）。没有 claim 依赖这两组。之后没有改代码，用 `kb_cppn lab` 和 `kb_cppn settle` 从已保存的结果生成表和判定。

## 结果

见 `verdict.json`。`cppn_continual_db.jsonl` 有 2480 行。K9105 的两组和问题 1 的 A100 数字完全相同，因为是同一台机器重跑同一配置，所以它不是新证据。

聚焦回放把遗忘量降到均匀回放的 0.19～0.24（K9101）。只看从未被标注的像素，比值是 0.26 和 0.32（`focus_heldout.md`，预测先提交），所以这不只是记住了标注。
