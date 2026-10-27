# 论文 toy 形式的 20 次拟合

本实验把原始表达空间改成 6 个基因，同时固定潜空间为 3 维：

```text
12 维 RNA 观测（6 个 spliced + 6 个 unspliced）
        → 3 维潜变量
        → 6 个基因的动力学重建
```

## 数据生成

Hill 函数（希尔调控函数）只用于 toy 数据生成。对调控输入 (s)，激活和抑制项为

\[
p_{act}(s)=\frac{s^n}{h^n+s^n},\qquad
p_{rep}(s)=\frac{h^n}{h^n+s^n}.
\]

生成器使用原文的 6 基因、10 条有向边、(K/n/h) 参数结构，并用 Euler–Maruyama（欧拉–马鲁雅马随机积分）步进近似原文的 `torchsde` 模拟。当前环境没有安装 `torchsde`，所以没有执行安装；漂移项和扩散强度保留在生成器中，并把生成方式写入 provenance（来源记录）。

数据文件为 `.runtime/paper-toy/observations.json`：300 个细胞、6 个基因、10 条真实支持边。`truth` 保存了 (K,n,h,\beta,\gamma) 和观测时间，`prior.skeleton` 使用 target × regulator（靶基因 × 调控基因）方向。

## 拟合配置

- 20 个模型共享同一份 toy 数据；训练随机种子为 `11 + 18 × (run - 1)`。
- 每个模型的潜变量维度固定为 3，隐藏层宽度为 64，最多 500 轮，验证集早停。
- 使用真实本地 RegVelo `VELOVAE`、RNA ODE（常微分方程）和完整损失路径。
- `soft_constraint=False`：先验外的 W 边采用硬梯度掩码；`lam2=1` 保留 Jacobian L1 项；`simple_dynamics=True` 与论文 toy 拟合脚本一致。
- 20 个模型均使用同一个训练/验证索引划分，只有优化随机种子变化。

结果文件：

- `.runtime/paper-toy/stability/stability.json`
- `dist/paper-toy-stability-results.mjs`

复现实验的主要命令：

```powershell
python tools/generate_paper_toy.py --output .runtime/paper-toy/observations.json --n-obs 300 --seed 0
python tools/run_paper_toy_stability.py --runs 20 --epochs 500 --samples 4 --hidden 64 --threads 1 --workers 3
python tools/train_regvelo_replicas.py --models 1 --seed 11 --observations .runtime/paper-toy/observations.json --output .runtime/paper-toy/w-history/run-01 --epochs 500 --samples 4 --hidden 64 --latent 3 --threads 1 --workers 1 --no-publish
python tools/publish_paper_toy_w_history.py
```

## 结果核对

- 20/20 个模型完成，结果 JSON 均无 `NaN` 或无穷值。
- 所有模型的 `mu` 维度为 3，W 均为 6×6。
- 所有模型的先验外边权重最大绝对值为 0；每个模型只保留 10 条 toy 支持边。
- 代表模型按 3 维潜空间细胞间距离结构选出运行 4、5、7、9、11。

网页模拟实验已改用同一份 6 基因 paper toy 数据，页面形式保持原来的“表达空间 → 三维潜空间 → 重建空间”三联图，并继续提供训练曲线、椭球、UMAP 和两池动力学视图。潜空间仍固定为 3 维。直接重建页使用全部 300 个细胞做评估；动力学页读取固定训练过程中的每 10 轮 W 快照，不在浏览器中重新优化。

动力学页的 W 回放使用固定的 51 个缓存状态；β、γ、b 和时间头保持教学模型的固定参数。重新抽样按钮只改变当前细胞的显示样本，不改变缓存轨迹。

论文网络页使用一次固定的 seed=11 拟合轨迹作为教学回放，不在浏览器中再次启动优化。轨迹保存在 `dist/paper-toy-w-history.mjs`：包含第 0 轮、每 10 轮和最终训练轮的真实 `W` 快照，共 51 个 `6×6` 矩阵；页面滑块只索引这些缓存值。该轨迹的最佳验证轮为第 450 轮，最终训练状态为第 495 轮，二者保留在元数据中供解释。
