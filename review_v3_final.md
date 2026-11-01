# WEV3 / PGPD 最终审查（review_v3_final）

> 审查对象：`CC/WEV3/README.md`、`CC/WEV3/review_pgpd_design.md`（设计审查，代码尚未创建）
> 代码基线：`CC/WEV1/WHU`（GPD-SC）；对照：`CC/WHU`（UAD）、`WEV2/WHU`（HPD）
> 论文：`CC/WEV1/WHU/tmp/pdfs/WHU.txt`（Sec.4.2–4.5、Eq.3–10、Table 3/4/6/8）
> 约束：只审查与分析，未修改任何项目代码；数值测试脚本放在系统临时目录。
> 标记：**[代码事实]** / **[实验支持]** / **[机制假设]**

---

## 0. 结论

**裁决：approve with conditions。可以进入实现与 1-epoch smoke；不建议在 P0-1、P1-2、P1-3、P1-4 关闭前直接开 120 epoch。**

- **P0 = 1 个**：`GPD_TYPE='pair'` 的分支覆盖。V1 有 6 处按 `gpd_type` 分派的代码；若只加 `GPD_PAIR` 而不改全部分派，`pair` 会**静默落回 `GPD` single / `PrototypeBank`**，或跳过 warm start/bank update，运行不报错但机制不是 PGPD。这是实现前的 P0。
- **P1 = 6 个**：
  1. TIR/AG/R5 回退风险被低估；"own max_k 保护 TIR" 无数据支持。
  2. Gate 定义与 gate 数学不一致（README vs 审查 #7；GD R5 46.0 不是 V1 preservation；Tier1 `cross +0.3 & overall +0.4` 在加性关系下不可行）。
  3. `w_cross=0` 只有在 `own=1` 时才严格等于 WEV1；`own=0.6, cross=0` 是 `0.6·GPD_SC`，不是 V1。第一轮包同时改了 own 权重和 cross 目标，归因需要 own-only 控制。
  4. 诊断字段/时序不完整：V1 没有 per-gallery CE、own_ratio/usage、梯度份额/cos、pre-clip/clip；V2 的诊断是 last-block/eval/rank0 代理，且必须在训练 forward 之前执行。
  5. 单 seed 与 <0.5pp 的 gate 阈值不匹配；第一轮只能做筛选，不能做最终结论。
  6. "P_id 是 gallery cross 的均值特例" 只在原型层成立，在 loss 层不成立；`0.4 vs 0.25` 是名义预算，不是有效梯度份额。

**一句话**：PGPD 的"own max_k + 两个 gallery 专属 cross"方向合理、可实现、可归因，且比 HPD 的均值锚点更贴近 3×3 模态协议；但当前文档把 V2 已经付出代价的方向（更高 cross 预算、更低 own）当成"更直接"，把 `w_cross=0` 当成 V1 锚点，并把 gate 写成了互相矛盾的三套数。修完 P0-1 与 P1-2/P1-3/P1-4 后再跑 120ep；TIR/AG/R5 的代价要用 gate 和诊断正面回答，不能靠"own max_k 保护"。

---

## 1. 代码/论文链路核对

### 1.1 核对表

| # | 事实 | 证据（file:line） | 与方案/论文是否一致 |
|---|---|---|---|
| 1 | 检索特征是 pre-BN CLS；`NECK_FEAT='before'`；推理返回 `global_feat` | `CC/WEV1/WHU/model/make_model.py:67-80`；`CC/WEV1/WHU/configs/SCGPD_UAD.yml:64` | 一致。论文 Sec.4.2 "At inference, we ℓ2-normalize f"（`WHU.txt:453-476`） |
| 2 | CE 走 BNNeck/classifier；`cls_score=classifier(bottleneck(global_feat))` | `make_model.py:70-73`；`WHU.txt:453-476`（Eq.3） | 一致。`L_base=CE(z)+WRT(f)` |
| 3 | WRT 作用在 pre-BN `global_feat`；`loss_fn(cls_score, global_feat, target_rep)` | `CC/WEV1/WHU/processor/processor.py:398-402`；`loss/make_loss.py:25-34`；`loss/wrt_loss.py:39-64` | 一致。WRT 的输入是 `global_feat`，不是 BN 后 `feat` |
| 4 | ProCA/GPD 作用在 pre-BN `global_feat`；`PCA(global_feat)`、`GPD_SC(global_feat)` | `processor.py:405-422`；`WHU.txt:575-600`（Eq.10） | 一致。论文明确 ProCA/GPD 用 pre-BN f |
| 5 | 推理：L2 normalize + 欧氏距离；排除同相机 gallery | `utils/metrics.py:12-17`（欧氏）；`:462-465`（feat_norm）；`:136-164`（`remove=(g_camids[order]==q_camid)`） | 一致。评测协议同 V1/V2 |
| 6 | V1 `GPD_SC`：`logits=einsum('nd,ckd->nck')/τ`，`max(dim=2)`，CE；`P_scene[C,3,D]` | `processor.py:234-251` | 一致 |
| 7 | V1 `SubCenterPrototypeBank`：ProCA 模态中心 + count EMA（0.1–0.5）；warm start 每 (pid,mod) 均值、缺失回退 id_mean、counts=1、rank0+broadcast | `processor.py:76-162`、`:182-231`、`:349-371`、`:446-460` | 一致 |
| 8 | V1 `own_ratio` **没有日志/诊断**；V2 的 `own_ratio` 是 `GPD_HPD` 的 diag | V1 `processor.py:234-251` 无 own_ratio；V2 `processor.py:398-407`；V2 日志 `WEV2/logs/HPD_1000/train_log_20261031_0946.txt:1741,1780,1818` | **需修正**：review 的 "0.92–1.0" 是 V2 末期 diag，不是 V1；V2 也有 0.844–0.891 的 epoch；TIR 末期=1.0 |
| 9 | V2 `GPD_HPD`：`P_id[C,D]`（g_cent EMA）+ `P_scene[C,3,D]`；`0.25 CE_id + 0.75 CE_scene` | `WEV2/WHU/processor/processor.py:165-230`、`:360-435`；`WEV2/WHU/configs/HPD_UAD.yml:19-28` | 一致 |
| 10 | V2 `P_id=mean(P_scene)`（cos 0.999）；`cos(P_id,P_scene)=[0.956,0.959,0.919]`；pair cos `[0.914,0.80,0.809]` | 日志 `...:1744,1755,1810`；V2 分析 `analysis_hpd_mechanism.md:82-89` | 一致（原型层）。**loss 层不是严格均值特例**（见 §3 #2） |
| 11 | V2 数据：TIR→TIR −1.22；AG −0.37/−0.52；R5/R10 −0.66/−0.13；GD R5 44.82；shared/scene/base 有效份额 10.6/23.3/65.8；clip 1.73/100 | 日志 `train_log...:1826-1857`；GD `WEV2/logs/HPD_1000GD_test/test_log_20261031_1500.txt:180-209`；diag `...:1814`；clip `...:1811,1816` | 一致。注意份额是 last-block+final-norm 代理，不是全 ViT |
| 12 | V3 公式：own max_k CE（V1 `GPD_SC`）+ 每个 query 对另外两个 gallery 模态 `P_scene[:,m']` 的 cross CE；`w_own=0.6, w_cross=0.2+0.2`，总 GPD=1.0 | `CC/WEV3/review_pgpd_design.md:52-73,83-113`；`CC/WEV3/README.md:12-26` | 公式/形状正确（数值验证见 §1.3）；但 `w_cross=0` 的等价声明需修正 |
| 13 | V3 不建 P_id/HPD；复用 `SubCenterPrototypeBank`/`warm_start_subcenters`/`GPD_SC`；推理不变 | `review_pgpd_design.md:112-113`；V1 代码同上 | 一致，且是正确的最小基线 |
| 14 | 论文 GPD 是**单原型** `P[C,D]`（Eq.7–9）；V1/V2/V3 的 subcenter/hierarchical/pair 都是扩展 | `WHU.txt:526-574`；`WHU.txt:718-723`（Table 8 只有 ProCA/GPD） | 一致于 base 链路；**不能把 V3 的 cross CE 说成论文 Eq.8/9 的直接实现**；V3 是待验证扩展 |
| 15 | 论文实现细节：P=16,M=3,K=4 → 每卡 192，LR 0.008；repo V1/V2 配置：batch 256、LR 0.032、4 卡、120ep、seed 1234 | `WHU.txt:683-710`；`SCGPD_UAD.yml:47-50`；`HPD_UAD.yml:51-54`；V2 日志 `...:103` | V3 的"同配方"是 V1/V2 repo 配方，不是论文的 192/LR0.008；V1/V2/V3 可比，但不能直接拿论文绝对值当同配方 |

### 1.2 V1/V2 数据核对（用于后续判断）

- V1 1000：`11.37/30.51/48.17/56.27`；same `31.16/19.97`；cross `13.33/9.00`；AG `35.11/12.02`、`32.66/13.93`；GD `14.84/31.10/44.61/53.76`（`CC/WEV1/logs/SCGPD_1000/train_log_20261029_2209.txt:1223-1254`；`CC/WEV1/logs/SCGPD_1000GD/test_log_20261030_0954.txt:173-202`）。
- V2 1000：`11.47/30.52/47.51/56.14`；same `30.97/19.96`；cross `13.72/9.16`；AG `34.74/12.10`、`32.14/13.93`；GD `14.91/31.61/44.82/54.17`（`WEV2/logs/HPD_1000/train_log_20261031_0946.txt:1826-1857`；`WEV2/logs/HPD_1000GD_test/test_log_20261031_1500.txt:180-209`）。
- 以上与任务背景一致；V2 的 TIR→TIR 21.12 vs V1 22.34（−1.22）、R5 −0.66、R10 −0.13、AG R1 −0.37/−0.52 均已在日志中复核。

### 1.3 V3 公式/代码形态的数值验证

临时脚本：`C:/Users/GX/AppData/Local/Temp/v3_review_test.py`（CPU，torch 2.8.0）。结果：

| 检查 | 结果 |
|---|---|
| `GPD_PAIR(own=1, cross=0)` vs `GPD_SC` | `9.85894775` vs `9.85894775`；`allclose=True` |
| `GPD_PAIR(own=0.6, cross=0)` | `5.91536903` = `0.6×9.85894775`；**不是 GPD_SC** |
| `GPD_PAIR(own=0.6, cross=0.2)` | `12.34272766` = `0.6×own + 0.4×cross`（差值 0.0） |
| 每 query 的 own:cross 有效权重 | `0.05:0.03333 = 1.5 = 0.6:0.4`（正确） |
| cross 索引 `q[:,:,m2]` 与手算 | 一致（RGB→NIR 的 CE 相同） |
| `target.repeat(M)` vs `repeat_interleave(M)` | `repeat` 是 modality-major，正确；`repeat_interleave` 会错位 |

**结论**：V3 公式/代码形态在 `M=3`、`target.repeat(M)`、bank 列顺序 `[RGB,NIR,TIR]` 下是正确的；唯一硬性等价是 `own=1, cross=0`。`own=0.6, cross=0` 是 `0.6·GPD_SC` 的**不同训练目标**，不能作为"回到 V1"。

---

## 2. PGPD 逐项审查

### 2.1 机制

**结构**：对每个 query（模态 m），own 项是 V1 的 `max_k`（0.6）；cross 项对另外两个 gallery 模态 m' 各一个 CE（0.2+0.2）。这正好把 3×3 模态矩阵的 9 个单元都覆盖：对角由 own（max_k 多数选 own）覆盖，6 个 off-diagonal 由两个 cross 覆盖。相比 HPD 的一个均值 `P_id`，它把"一个均值锚点"拆成"两个 gallery 专属锚点"。

**合理之处**：

- **[代码事实]** own 项与 V1 完全同形；`own=1,cross=0` 数值等价于 `GPD_SC`（§1.3）。这给了一个干净的 V1 锚点（前提是 own=1）。
- **[机制假设]** cross 项的正样本是同一 identity 的 m' 子中心，负样本是全部 C 个 identity 的 m' 子中心；它直接对应"query 模态 m → gallery 模态 m'"的检索方向，比 HPD 的均值更具体。
- **[实验支持]** V2 的 cross 增益集中在 NIR↔RGB（NIR→RGB +1.46/+0.30、RGB→NIR +0.61/+0.24），TIR 的 cross 基本不动；V3 的 per-gallery cross 主要改变的是 TIR（pair cos 0.809 vs 均值 cos 0.919）和 own 权重。因此第一轮的主要实验信息在 TIR 和 same/AG/R5，而不是"平均 cross 再涨一点"。

**主要张力**：

- **[实验支持]** V2 的 own_ratio 末期是 `[0.922,0.938,1.0]`（TIR=1.0），说明 TIR 的 own max_k 已经**总是**选 TIR 子中心；"max_k 允许 TIR 选更合适的子中心"对 TIR 不成立，max_k ≈ hard own。V3 的 own 权重从 V2 的 0.75 降到 0.6，TIR 的保护是**更弱**，不是更强。
- **[机制假设]** V3 的 cross 目标（TIR→RGB/NIR 子中心，pair cos 0.809）比 V2 的均值（cos 0.919）**更远**；cross 预算从 0.25 提到 0.4；两者叠加，TIR→TIR 的风险高于 V2（V2 已经 −1.22）。"own max_k 保护 TIR" 在机制上不成立；能保护 TIR 的只有 own 权重，而 own 权重是下降的。
- **[机制假设]** cross 的负类是"另外 C−1 个 identity 的 m' 子中心"，比 HPD 的均值负类更硬；它可能提高 cross 的判别，也可能把 query 拉向错误的 m' 原型，R1 方向不确定；必须用 per-pair R1 和 gradient cos 诊断，而不是假设 cross R1 一定升。
- **[实验支持/机制假设]** V2 的 AG 回退（A→G R1 −0.37、G→A −0.52）来自 platform_only（同模态跨平台）；V3 的 cross 是模态级，不含平台项，AG 没有新的保护机制。R5/R10 的回退来自 scene 预算从 100%→75%；V3 的 own 60% 更少，R5/R10 风险更大。
- **[机制假设]** V3 去掉了 V2 的均值锚点。均值锚点虽然"太温和"，但它是一个 centripetal 力；V3 的 per-column cross 是否会让子中心更分散（pair cos 下降）或更对齐（pair cos 上升），没有数据。第一轮必须记录 `P_scene` pair cos 和 per-gallery cross CE。

**裁决**：机制方向合理，是"比 HPD 更具体"的正确下一步；但"更直接"同时意味着"对 TIR/AG/R5 更不温和"，文档的"own max_k 保护 TIR"是错误的安全感。机制本身可以进第一轮，但不能承诺双基线。

### 2.2 数据/梯度流

- **[代码事实]** `global_feat` 是 `[M*B,D]`，model 的 `torch.cat(list(x),dim=0)` 是 modality-major（`make_model.py:69`）；`target.repeat(M)` 是 modality-major（`processor.py:395`）；`PCA` 的 `reshape(M,B,D)` 与 `GPD_SC` 的行对齐（`processor.py:18,167`）。V3 的 `logits_all[mB:(m+1)B]`、`target_rep[mB:(m+1)B]`、`q[:,:,m2]` 与这一顺序一致（§1.3）。
- **[代码事实]** V3 的 cross 项只对 `f` 求梯度，`P_scene` 是 `bank.sub_centers.float().clone()` + `no_grad` + `F.normalize`（同 `GPD_SC`，`processor.py:239-246`）；bank 不接收梯度；bank update 仍在 `optimizer.step()` 之后（`processor.py:446-460`）。梯度流正确。
- **[代码事实]** V3 不需要 P_id；`P_scene` 的更新仍是 ProCA 模态中心 + count EMA；warm start 仍是 `warm_start_subcenters`。这与"只改 loss"的最小改动一致。
- **[实验支持/机制假设]** 名义权重不等于有效梯度份额：V2 的 shared 名义 0.25、有效 10.6%；scene 名义 0.75、有效 23.3%（last-block 代理，`...:1814`）。V3 的 cross 名义 0.4，但 cross CE 的 raw 梯度可能比 own 大（随机 bank 探针：cross 16.07 vs own 9.86；合成 warm-start 探针：6 个 cross CE 为 5.55–5.88，own 为 5.71；该合成 pair cos 仅 0.5，不是真实 0.8–0.91，仅作量级提示）。因此 V3 的 cross 有效份额可能 **>40% of GPD**，own 有效份额可能 **<60%**；必须实测。
- **[代码事实]** V3 的 cross 项与 own 项共享同一个 `logits_all`；own 的 max_k 可能在某些样本上选到 cross 列（own_ratio <1），此时 own 与 cross 对同一列重复施力；这是 V1 的既有权重，不是 bug，但 own_ratio 诊断要按模态记录。

**裁决**：数据/梯度流没有发现 P0；`GPD_PAIR` 的数值/形状/分段/权重正确。唯一实现风险是分支覆盖（P0-1）和诊断时序（P1-4）。

### 2.3 监督/冲突

- **[代码事实]** 三个 CE 都是全局 CE：own 的负类是全部 C 个 identity 的 max_k 子中心；每个 cross 的负类是全部 C 个 identity 的 m' 子中心。不是 batch-local，不是 3%–12.6% 候选覆盖问题。
- **[代码事实]** 正/负样本设计：own 正样本是 `P_scene[y]` 的 max_k；cross 正样本是 `P_scene[y,m']`；负样本是其他 identity 的同一列。`target` 对齐正确（§1.3）。
- **[机制假设]** own 与 cross 的冲突：对 RGB/NIR，own 与 cross 的正样本列接近（pair cos 0.914），cos 可能正相关；对 TIR，own 的 TIR 列与 cross 的 RGB/NIR 列 cos 0.809，冲突更大。V2 的 `cos(g_id,g_scene)=0.83–0.94` 说明均值锚点与 scene 是正相关；V3 的 per-column cross 不一定保持这个正相关，必须测 `cos(g_own,g_cross_m')`。
- **[机制假设]** 与 WRT 的重叠：WRT 已经做跨模态正对（同 pid 不同模态）；V3 的 cross CE 是全局原型版；边际收益可能被 WRT/CE 的已有跨模态对齐吃掉（V2 的 cross 只有 +0.16 mAP 是这个现象的一个证据）。V3 的 cross 更具体，但仍是分类目标，不直接优化 AP/CMC。
- **[实验支持/机制假设]** TIR 的 cross 不能创造细节：TIR 表征弱是瓶颈（论文 Table 6：TIR→RGB 9.04/6.32、TIR→TIR 20.98/14.00；V2 的 TIR 仍最弱）。cross 只能把 TIR query 拉向 RGB/NIR 原型，不能提高 TIR 类内可分性；TIR→TIR 的风险不会因为 cross 而降低。
- **[机制假设]** 负类硬度：cross 的负类是单列原型，比 HPD 的均值负类更硬；可能提高 cross 的判别，也可能增加 clip 和 R1 波动。

**裁决**：监督充分、全局、无 batch-local；没有"监督对象 ≠ 推理特征"的问题。主要风险是 TIR 冲突和 own/cross 的梯度相关性，需诊断。

### 2.4 训练/优化

- **[代码事实]** V1/V2 配方：4 卡 DDP，batch 256，LR 0.032，120ep，seed 1234，cosine + 5 warmup，AMP，`clip_grad_norm_(1.0)`，warm start，bank 不挂 DDP，count EMA（`SCGPD_UAD.yml:47-50`；`HPD_UAD.yml:51-54`；V2 日志 `...:103`）。V3 复制 V1，同配方是正确的。
- **[代码事实/机制假设]** V2 的 pre-clip 1.73、clip 100/114（`...:1811,1816`）；V3 的 cross 0.4 可能让总梯度更大，clip 触发更多；不要同时改 `max_norm`。若 clip 接近 114/114，cross 的绝对步长会被压缩，可能"名义 0.4、有效更小"。
- **[机制假设]** 第一轮只跑 `0.6/0.2+0.2` 一个点：作为筛选可以；作为归因不够，因为它同时改了 own（0.75→0.6）和 cross 目标（mean→per-gallery）。需要 `own=0.6, cross=0` 的 own-only 控制，或至少把第一轮明确标为"package test"。
- **[实验支持]** 单 seed 噪声：V2 末期 R5 −0.33、cross R1 −0.12（`analysis_hpd_mechanism.md:210-222`）；历史 seed 噪声 mAP 0.24/R1 1.13（`WEV2/review_hpd_code.md:132`）。V3 的 gate 阈值 +0.3/+0.4 与 −0.2/−0.3 都在噪声边缘；单 seed 不能裁决。
- **[代码事实/方案]** 不应同时改的模块：平台 token、TIR adapter、EMA/TTA、PCGrad、硬分配、CA-WRT、LLRD、margin、双温度、P_id、HPD。V3 第一轮只改 GPD 的 cross，是正确的最小单变量。唯一"顺带"的变量是 own 权重（0.6 vs V2 0.75/V1 1.0），这是 PGPD 定义的一部分，但要在归因里明确。
- **2337**：第一轮不跑是正确的；1000 gate 通过后再跑 2337。2337 的 bank 更大、缺失模态/长尾可能更多，cross 正样本的 id_mean 回退风险更高；不要用 2337 调第一轮权重。

**裁决**：训练/优化链路可复用 V1，不需要新模块；P1-3（归因控制）和 P1-5（seed）是条件，不是训练阻塞。

### 2.5 风险表

| 风险 | 证据 | 严重度 | 第一轮怎么回答 |
|---|---|---|---|
| TIR→TIR 回退 | V2 均值锚点已 −1.22；V3 own 0.6 < 0.75；V3 cross 目标更远（0.809 vs 0.919）；V2 TIR own_ratio=1.0 | **P1（最高）** | TIR→TIR R1/mAP、TIR cross、own_ratio(TIR)、cross CE(TIR→RGB/NIR)；gate 加 TIR→TIR |
| AG R1 回退 | V2 A→G −0.37、G→A −0.52；V3 无平台项 | **P1** | detailed-domain AG；platform_only/modality_and_platform |
| R5/R10 回退 | V2 R5 −0.66、R10 −0.13；V3 own 0.6 | **P1** | overall/同模态 R5/R10；own/cross 份额 |
| GD R5 < 46.0 | V1 44.61、V2 44.82；V3 无 R5 机制 | **P1（gate 误标）** | GD 四项；把 46.0 标成 paper+1/guardrail，不是 V1 preservation |
| cross 不够 | V2 mean cross +0.16 mAP；V3 名义 0.4 的有效份额未知 | **P1** | per-gallery cross CE、gradient share、per-pair cross R1 |
| 单 seed 噪声 | V2 末期 R5 −0.33；历史 mAP 0.24 | **P1** | 第一轮筛选；最终 2–3 seed |
| 诊断代理/时序 | V2 诊断 last-block/eval/rank0；V2 r1 P1-3 | **P1** | 诊断放在训练 forward 之前；标注代理 |

### 2.6 裁决

**方向 approve，实施有条件。** 机制是正确的"比 HPD 更具体"的下一步；代码形态可复用 V1；第一轮是一个可证伪的测试。条件是：先补 P0-1 分支覆盖、P1-2 gate、P1-3 归因锚点、P1-4 诊断；TIR/AG/R5 用 gate 正面回答；不要用"max_k 保护 TIR"当结论。

---

## 3. 对 V3 审查关键修正的逐条裁决

| # | V3 审查的修正（`review_pgpd_design.md` §6） | 裁决 | 证据 / 理由 |
|---|---|---|---|
| 1 | own 项用 WEV1 的 `GPD_SC` max_k，不用硬 own；`w_cross=0` 必须与 `GPD_SC` 数值等价 | **部分成立 / 需修正** | 用 max_k **成立**：它是 V1 的原机制，`own=1,cross=0` 数值等价（§1.3），保留 V1 锚点。**但**"`w_cross=0` 与 GPD_SC 等价"只在 `own=1` 时成立；`own=0.6,cross=0` 是 `0.6·GPD_SC`（§1.3），不是 V1。且"own_ratio 0.92–1.0"是 V2 末期 diag（`...:1818`），不是 V1；TIR own_ratio=1.0 说明 max_k 对 TIR ≈ hard own，max_k 不是 TIR 保护。 |
| 2 | cross 项用另外两个 gallery 模态的 `P_scene[:,m']`，不是均值；两个 CE | **成立** | 公式/索引/分段正确（§1.3）；每个 CE 负类全部 C，全局。**补充**：V2 的 `P_id` 是原型层的均值，但 `CE(mean P_scene) ≠ mean_m CE(P_scene[:,m])`；V3 不是 V2 的严格"均值特例"，而是不同目标，不能把 V2 的结果直接外推。 |
| 3 | 第一轮 `w_own=0.6, w_cross=0.2+0.2`，总 GPD=1.0；TIR-specific 第二轮 | **部分成立** | 总 GPD=1.0（M=3）与每 query own:cross=0.6:0.4 正确（§1.3）；作为单点筛选合理。**但** own 0.6 低于 V2 的 0.75（V2 已丢 TIR/AG/R5）；cross 0.4 是名义值，有效份额可能更高；0.6/0.4 不是数据推出的平衡点，是首个测试点。TIR-specific 放第二轮是正确流程。 |
| 4 | 从 WEV1 复制，不用 WEV2；不需要 P_id/HPD；复用 SubCenterPrototypeBank/warm_start/GPD_SC | **成立（实现有 P0 条件）** | 最小基线正确；V1 的 bank/update/warm start 可直接复用。**但** V1 有 6 处 `gpd_type` 分派（`processor.py:317-319,323-348,349-371,413-432,446-460,499-511`），`pair` 必须全部接上；否则静默落回 `GPD` single/`PrototypeBank`，或跳过 warm start/bank update（P0-1）。 |
| 5 | 新增 `GPD_PAIR`、`GPD_TYPE='pair'`、`GPD_PAIR_OWN_WEIGHT`/`GPD_PAIR_CROSS_WEIGHT`；默认 single/subcenter 不变 | **成立 / 需补充** | 接口名合理；默认路径必须逐行不变（V1 `GPD_SC` 保留）。**补充**：加 `GPD_TYPE`×`LOSS_TYPE` 守卫（V2 r1 P1-2：`base+pca` 会让 bank/warm start 空跑而 loss 不加 GPD）；加 `GPD_DIAG_PERIOD`；加 1ep smoke 断言 `pair` 真的用 `SubCenterPrototypeBank`/`GPD_PAIR`。 |
| 6 | 诊断：per-gallery CE、own/cross 份额/cos、per-pair R1、TIR、own_ratio、P_scene 几何、clip；不需要 P_id | **成立 / 需补充** | 字段方向正确；但 V1 代码**没有**这些字段，必须新增；V2 的 `diagnose_gpd_gradients` 是 last-block/eval/rank0 代理，且 V2 r1 P1-3 证明诊断必须在训练 forward/bank update **之前**（否则 own_ratio/份额被 B_{t+1}/θ_{t+1} 污染）；V2 r2 已把它移到 `processor.py:710-738`。**补充**：`usage`、pre-clip norm、AG/detailed-domain、per-pair cross/TIR；不要只报平均 cross。 |
| 7 | 第一轮 gate：preservation（same ≥WEV1−0.2、AG ≥WEV1−0.3、R5≥47.3、R10≥55.8、GD R5≥46.0）+ Tier1（cross mAP/R1 ≥+0.3、overall mAP ≥+0.4） | **部分成立 / 需修正** | preservation 的 same/AG/R5/R10 是 V1 容差（但 R5/R10 是 V1−0.87/−0.47，不是 −0.2，需写明）；**GD R5 46.0 不是 V1 preservation**（V1 44.61；46.0 是 paper GD R5 45.0+1 的 Tier2/guardrail，V2 plan §6.1 也把它标成"护栏"）。Tier1 的 `cross +0.3 & overall +0.4` 在加性关系 `Δoverall mAP≈0.908×mean₉ ΔmAP` 下不可行：same 保持时 cross +0.3 → overall ≈+0.18，不是 +0.4；要 +0.4 需 cross ≈+0.66 或 same 上涨。V2 最终审查已把同类问题标 P0-1（`WEV2/review_hpd_final.md:134,148-168`）。另外 README 的 beat-V2 gate（overall >11.47、cross >13.72/9.16）与 #7 的 +0.3/+0.4 是两套数，必须统一。 |
| 8 | 风险：own 0.6 vs V1 1.0；TIR own/cross 冲突；cross 硬负样本；R5/AG/R10；TIR 表征弱；第一轮 net 不确定 | **成立 / 需重排** | 风险列得对，但优先级应把 **TIR→TIR、AG、R5/R10** 放最前；V2 已经用 0.75 的 own 付出这些代价，V3 的 0.6 更激进。"own max_k 保护 TIR" 不成立（TIR own_ratio=1.0、own 权重下降）；"第一轮 net 不确定"低估了方向性：cross 越大，TIR/AG/R5 越可能回退。 |

---

## 4. 必须修正的 P0/P1

### P0-1（实现/可运行性，阻塞 1ep smoke）：`GPD_TYPE='pair'` 分支覆盖

**[代码事实]** V1 的 6 处分派：

1. `CC/WEV1/WHU/processor/processor.py:317-319`：允许类型 `('single','subcenter')`；加 `'pair'`，否则 `ValueError`。
2. `:323-348`：bank 创建。`pair` 必须走 `SubCenterPrototypeBank`（`if gpd_type in ('subcenter','pair')`）；若走 `else` 的 `PrototypeBank`，`GPD_PAIR` 访问 `bank.sub_centers` → AttributeError；若 `GPD_PAIR` 没被调用而落到 `GPD`，则静默变成 single。
3. `:349-371`：warm start。`pair` 必须调用 `warm_start_subcenters`；否则 bank 从零开始，第一步的 `g_cent` fallback 会把所有 K 列填成同一个 `g_cent`，不是模态均值，机制错误。
4. `:413-432`：loss 分支。`pair` 必须调用 `GPD_PAIR`；V1 的 `else` 是 `GPD`（single），若只加类型不加分支，**静默用错 loss**。
5. `:446-460`：bank update。`pair` 必须走 subcenter update（`ids.repeat(M)`、`mod_ids`、`centers_all`）；若走 `proto_bank.update(ids,g_cent)`，`PrototypeBank` 不存在或语义错误。
6. `:499-511`：bank counts 日志；`pair` 应走 subcenter counts（非正确性，但 smoke 要能看到）。

V2 的同类分派可作为模板：`WEV2/WHU/processor/processor.py:596-628`、`:630-680`、`:757-785`、`:809-833`、`:888-941`。

**修复**：所有站点用 `gpd_type in ('subcenter','pair')` 或显式 `elif gpd_type == 'pair'`；加 `GPD_TYPE != 'single' and 'gpd' not in loss_type` 守卫；1ep smoke 打印/断言 `type(proto_bank).__name__ == 'SubCenterPrototypeBank'` 且 loss 来自 `GPD_PAIR`（例如 monkeypatch 计数器），并检查 warm start 日志/counts。

**为什么 P0**：静默落回 `GPD` single 不会报错，120ep 跑完才发现机制不是 PGPD；这是最贵的失败模式。

### P1-1（机制/gate）：TIR/AG/R5 风险被低估

**[实验支持]** V2 用 own(scene) 0.75 + mean shared 0.25 已经：TIR→TIR −1.22、AG −0.37/−0.52、R5 −0.66、R10 −0.13（`...:1826-1857`）。V3 用 own 0.6 + per-gallery cross 0.4，且 TIR 的 cross 目标更远（pair cos 0.809 vs 均值 0.919）。V2 的 TIR own_ratio=1.0（`...:1818`），说明 max_k 对 TIR 已经是 hard own；"max_k 保护 TIR" 没有数据支持。gate 的 "same ≥30.96" 是三对平均；V2 的 same 30.97 已经通过，但 TIR→TIR 仍 −1.22；平均 gate 不保护 TIR。

**修复**：preservation gate 加 `TIR→TIR R1/mAP ≥ V1−0.5`（或至少硬诊断）；第一轮报告 TIR→TIR/TIR→RGB/RGB→TIR/TIR→NIR；如果 TIR 回退 >0.5，第二轮的单点是 TIR-specific own（例如 TIR own 0.8，其他 0.6）或降低 cross，而不是继续加 cross。不要把"cross 平均涨"当成九协议成功。

### P1-2（gate 定义/数学）：README 与审查 #7 是两套 gate

**[代码事实/实验支持]**

- README：beat V2 `overall mAP >11.47`、`cross >13.72/9.16`、GD `≥14.91/31.61/44.82/54.17`；guard V1 `same ≥30.96`、`AG ≥34.81/32.36`、`R5≥47.3`、`R10≥55.8`、`GD R5≥46.0`。
- 审查 #7：preservation + Tier1 `cross mAP/R1 ≥+0.3`、`overall mAP ≥+0.4`。
- 差异：Tier1 overall +0.4 = 11.77，不是 README 的 >11.47；Tier1 cross R1 +0.3 = 13.63，**低于** README 的 >13.72；Tier1 cross mAP +0.3 = 9.30，**高于** README 的 >9.16。三套数不能同时作为 gate。
- GD R5 46.0 不是 V1 preservation（V1 44.61）；它是 paper GD R5 45.0 +1 的 Tier2/guardrail。V2 plan §6.1 已注明"WEV1 44.61，护栏"（`WEV2/HPD_final_plan.md:139-143`）。
- 加性关系：`Δoverall mAP≈0.908×mean₉ ΔmAP`（V2 final review，45 实验；`WEV2/review_hpd_final.md:134`）。same 保持（Δ0）+ cross +0.3 → overall ≈+0.18；`cross +0.3 & overall +0.4` 不可行。V2→V1 的数据验证：same −0.01、cross +0.16、overall +0.10，符合 0.908×(6×0.16−3×0.01)/9≈0.094。

**修复**：写一张 pre-registered gate 表，明确 baseline：
- V1 preservation（hard）：same 3 对 ≥V1−0.2；AG 两向 ≥V1−0.3；overall R1≥30.5、R5≥47.3、R10≥55.8；GD 四项 ≥V1−0.2（GD R5 ≥44.41）。
- Beat V2（Tier1）：overall mAP >11.47；cross R1/mAP >13.72/9.16；GD ≥14.91/31.61/44.82/54.17。
- Paper+1 / guardrail（Tier2，非第一轮通过条件）：overall mAP ≥12.0；GD R5 ≥46.0；cross ≥ paper+1。
- 若要保留 `cross +0.3 & overall +0.4`，必须把 cross 门槛改成 +0.66，或把 overall 改成 +0.2；不能两个都写。
- 统一 mAP/R1 顺序（README 的 same/cross 是 R1/mAP，GD/overall 是 mAP/R1/R5/R10）。

### P1-3（归因锚点）：`w_cross=0` 不是 V1；第一轮是 package test

**[代码事实]** `GPD_PAIR(own=0.6,cross=0) = 0.6·GPD_SC`（§1.3）；只有 `own=1,cross=0` 是 V1。V3 README line 20 同时写了"`w_cross=0` 严格回到 WEV1"和"`w_own=1,w_cross=0` 是 WEV1"，前者在第一轮 own=0.6 下不成立。review #1 的"`w_cross=0` 必须与 GPD_SC 数值等价"也有同样问题。第一轮同时改了 own（0.75→0.6）和 cross（mean→per-gallery），即使指标变化，也不能归因给 cross。

**[修复]**：
1. 把 V1 锚点明确写成 `own=1, cross=0`；smoke 断言 loss/grad 与 `GPD_SC` 相等。
2. 若要做指标归因，加一个 `own=0.6, cross=0` 的 own-only 控制（同 GPD 总预算）；否则第一轮只能叫"package test"，用 per-gallery CE/梯度份额回答"cross 是否在学习"，不能回答"cross 贡献了多少指标"。
3. 文档删掉"`w_cross=0` 严格回到 WEV1"的无限定说法。

### P1-4（归因/诊断）：字段、时序、代理

**[代码事实]** V1 `GPD_SC` 没有返回 diag；V1 `do_train` 没有 own_ratio/usage/gradient share/clip 记录（`processor.py:234-251,413-460`）。V2 的 `diagnose_gpd_gradients`（`WEV2/...:438-525`）是 last-block+final-norm 代理、eval 模式、rank0 单 shard；V2 r1 P1-3（`WEV2/review_hpd_code.md:53-62`）证明它若放在 optimizer step/bank update 之后，会系统性高估 own_ratio、低估梯度；V2 r2 已移到训练 forward 之前（`processor.py:710-738`）。

**[修复]**：V3 写一个最小 `diagnose_gpd_pair_gradients`，放在训练 forward **之前**（θ_t/B_t），或 bank snapshot；记录：
- per-gallery CE：own、cross(m→m') 两个；
- `g_own`、`g_cross_m'` 的 norm、`cos(g_own,g_cross_m')`、有效份额；
- per-modality own_ratio、usage（max_k 选中列计数）；
- `P_scene` pair cos、per-column norm、`g_cent_norm`；
- pre-clip norm、clip_count；
- per-pair R1/mAP（现有 `metrics.py:470-540` 已输出 9 对）、TIR、AG（`DETAILED_DOMAIN_EVAL True`）、GD（`FILTER_GD True`）。
标注代理范围（last-block/norm、eval、rank0）；不要用 diag 的 loss 代替训练 loss。

### P1-5（实验设计）：单 seed 与 <0.5pp 阈值

**[实验支持]** V2 末期 R5 −0.33、cross R1 −0.12；历史 seed 噪声 mAP 0.24/R1 1.13（`WEV2/review_hpd_code.md:132`）；V2 plan §6.2 明确"cross +0.3 接近噪声"（`WEV2/HPD_final_plan.md:145-149`）。V3 的 gate 阈值 +0.3/+0.4、−0.2/−0.3 都在噪声边缘。

**[修复]**：第一轮 120ep 单 seed 只做 screening；若结果在 gate 边缘（<0.5pp），不要宣布 pass/fail；最终结论 2–3 seed + paired bootstrap。若第一轮效应 >1pp，可先用单 seed 做方向判断，但仍不能当作最终论文结论。

### P1-6（概念/归因）："P_id 是均值特例"与"0.4 vs 0.25"

**[代码事实/机制假设]** V2 的 `P_id` 是 `mean(P_scene)`（cos 0.999，`...:1744`），但 loss 层 `CE(mean_m P_scene)` 不等于 `mean_m CE(P_scene[:,m])`；V3 的 per-column cross 是不同目标，不是严格特例。V2 的有效份额 10.6% vs 名义 25%（`...:1814`）说明名义预算 ≠ 有效梯度；V3 的 cross 名义 0.4，但 cross CE 的 raw 梯度可能大于 own（临时探针 cross 16.07 vs own 9.86；合成 warm-start 的 6 个 cross CE 为 5.55–5.88、own 5.71，pair cos 0.5 仅量级提示），所以"0.4 vs 0.25"不能直接当成"cross 压力增加 60%"。

**[修复]**：文档改成"P_id 是均值原型；V3 用 per-column 原型目标，是更细的监督，不是严格推广"；第一轮必须报有效梯度份额；不要用名义 0.4 推断 cross 强度。

---

## 5. 第一轮实现/实验/诊断最小清单

### 5.1 实现（从 WEV1 复制）

- [ ] `WEV3/WHU` 从 `CC/WEV1/WHU` 复制；`model/`、`loss/`、`datasets/`、`utils/`、`solver/` 不改；只改 `processor/processor.py`、`config/defaults.py`（新增 key）、`configs/PGPD_UAD.yml`。
- [ ] `GPD_PAIR(feat,target,bank,ids,g_cent,num_modalities,tau,own_weight,cross_weight)`：复制 `GPD_SC` 的 normalize/fallback；`logits_all`；`loss_own`；`cross_terms`（m→m'，m'≠m）；`loss=own_w*loss_own + cross_w*(M-1)*mean(cross)`；返回 loss + per-gallery losses/diag。
- [ ] 6 处 `gpd_type` 分派全部接 `pair`（P0-1）；加 `GPD_TYPE != 'single' and 'gpd' not in loss_type` 守卫；加 `GPD_DIAG_PERIOD`。
- [ ] `configs/PGPD_UAD.yml`：`GPD_TYPE: 'pair'`、`GPD_SUB_CENTERS: 3`、`GPD_PAIR_OWN_WEIGHT: 0.6`、`GPD_PAIR_CROSS_WEIGHT: 0.2`（**per-gallery**；总 cross=(M−1)×0.2=0.4）、`GPD_TAU: 0.03`、`GPD_COUNT_EMA: True`、`GPD_EMA_MIN/MAX: 0.1/0.5`、`GPD_WARMSTART: True`、`GPD_DIAG_PERIOD: 100`、`LOSS_TYPE: base+pca+gpd`、`OUTPUT_DIR: ./logs/PGPD_1000_DDP`。
- [ ] 单元/数值测试：`own=1,cross=0` loss/grad == `GPD_SC`；`own=0.6,cross=0` == `0.6·GPD_SC`；`own=0.6,cross=0.2` == `0.6 own+0.4 cross`；cross 索引/`target.repeat(M)` 对齐；bank 不变、无 grad、无 NaN；warm start/zero-row fallback/missing modality；fake DDP all-gather 顺序；`single`/`subcenter` 回归逐元素等于 V1。
- [ ] 1ep smoke：`GPD_TYPE='pair'` 确实使用 `SubCenterPrototypeBank`/`GPD_PAIR`（打印/断言）；loss/梯度有限；bank counts 增长；`own_ratio`/usage/CE 有限；pre-clip/clip 正常；`single`/`subcenter` 回归；无 NaN；DDP bank 各 rank 一致。

### 5.2 实验（第一轮唯一主点）

- [ ] 1000 AS，4 卡 DDP，batch 256，LR 0.032，120ep，seed 1234；与 V1/V2 同配方；只改 GPD。
- [ ] 测试：`transformer_120.pth`，`NECK_FEAT='before'`，`FEAT_NORM='yes'`；1000/GD/AG/detailed-domain；`TEST.DETAILED_DOMAIN_EVAL True`；GD 用 `DATASETS.FILTER_GD True`。
- [ ] 第一轮只跑 `0.6/0.2+0.2`；不并行 0.5/0.25+0.25、0.7/0.15+0.15、TIR-specific、P_id、平台 token、TIR adapter、EMA/TTA、PCGrad、硬分配。
- [ ] 单 seed 只做 screening；最终 2–3 seed；2337 等 1000 gate 之后。

### 5.3 诊断（第一轮必须）

- [ ] per-gallery CE：own + cross(m→m') 两个；per-modality。
- [ ] `g_own`/`g_cross_m'` norm、`cos`、有效份额；base/PCA 份额；pre-clip/clip。
- [ ] own_ratio、usage；`P_scene` pair cos、per-column norm；`g_cent_norm`。
- [ ] per-pair R1/mAP（9 对）；TIR→TIR/TIR→RGB/RGB→TIR/TIR→NIR；AG A→G/G→A；GD 四项；same/cross 平均。
- [ ] 诊断放在训练 forward 之前（θ_t/B_t），标注 last-block/eval/rank0 代理。

### 5.4 归因控制（条件，若要做指标归因）

- [ ] V1 锚点：`own=1, cross=0` 重跑或复用 V1 权重（同 bank 状态不可得，但同配方 V1 已有 11.37/30.51/48.17/56.27）。
- [ ] own-only 控制：`own=0.6, cross=0`（同 GPD 总预算）；用于分离"own 降权"与"cross 目标"。
- [ ] 若预算只允许一个 120ep，第一轮结论只能写"package test + 梯度诊断"，不能写"cross 贡献 +X"。

---

## 6. 残余风险与不阻塞项

1. **论文配方 vs repo 配方**：论文每卡 192/LR 0.008；repo V1/V2 用 batch 256/LR 0.032/4 卡。V3 的"同配方"是 V1/V2 配方；V1/V2/V3 可比，但不能把论文绝对值当同配方基线。非阻塞，但写论文时要注明。
2. **单 seed/噪声**：见 P1-5；第一轮 screening，最终 2–3 seed。
3. **2337**：第一轮不跑；1000 通过后跑一次，关注 bank 大小、缺失模态、长尾、warm start 回退。非阻塞。
4. **checkpoint/resume**：bank 不在 `state_dict`；V1/V2 接受 one-shot；V3 同样。非阻塞。
5. **clip**：V2 pre-clip 1.73/100；V3 cross 可能更大；只记录，不改 `max_norm`。非阻塞。
6. **诊断代理**：last-block+norm、eval、rank0；不能当全 ViT/全 rank 精确份额。非阻塞，但日志要标注。
7. **TIR 表征弱**：cross 不能创造 TIR 细节；TIR 的瓶颈在表征，不在锚点。非阻塞，但 gate 要正面回答。
8. **AG 无平台项**：V3 是模态级，不是平台级；AG 是 side effect；detailed-domain 必须报。非阻塞。
9. **WRT/CE 的跨模态重叠**：cross 的边际收益可能被已有跨模态对齐吃掉；per-gallery CE/梯度份额回答。非阻塞。
10. **缺失模态**：warm start 用 id_mean 回退；cross 正样本对缺失 m' 的 identity 是均值，不是真实 m' 原型；1000 数据若全模态则影响小；2337 要看。非阻塞。
11. **论文+1**：V3 第一轮 gate 是相对 V1/V2，不是 paper+1；paper+1 mAP 12.0 需要更大的 cross（约 +1.1），不在第一轮承诺。非阻塞，但不要混用。

---

## 7. 证据路径

**代码**

- `D:/CODE/Doing/WHU_base/CC/WEV1/WHU/processor/processor.py`（PCA 13-32；SubCenterPrototypeBank 76-162；warm start 182-231；GPD_SC 234-251；do_train 254-613；分派 317-319/323-348/349-371/413-432/446-460/499-511；inference 562-613）
- `D:/CODE/Doing/WHU_base/CC/WEV1/WHU/model/make_model.py:28-80`
- `D:/CODE/Doing/WHU_base/CC/WEV1/WHU/utils/metrics.py:12-17,71-99,136-164,327-540`
- `D:/CODE/Doing/WHU_base/CC/WEV1/WHU/loss/make_loss.py:25-34`；`loss/wrt_loss.py:39-64`
- `D:/CODE/Doing/WHU_base/CC/WEV1/WHU/configs/SCGPD_UAD.yml:18-24,38-65`
- `D:/CODE/Doing/WHU_base/WEV2/WHU/processor/processor.py`（HierarchicalPrototypeBank 165-230；GPD_HPD 360-435；diagnose 438-525；分派 596-628/630-680/757-785/809-833/888-941）
- `D:/CODE/Doing/WHU_base/WEV2/WHU/configs/HPD_UAD.yml:19-28,42-69`
- 临时数值测试：`C:/Users/GX/AppData/Local/Temp/v3_review_test.py`；`C:/Users/GX/AppData/Local/Temp/v3_warmstart_probe.py`

**论文**

- `D:/CODE/Doing/WHU_base/CC/WEV1/WHU/tmp/pdfs/WHU.txt:453-476`（Sec.4.2，Eq.3）
- `:477-525`（Sec.4.3，Eq.4-6）
- `:526-574`（Sec.4.4，Eq.7-9）
- `:575-600`（Sec.4.5，Eq.10）
- `:608-623`（Table 3）；`:661-680`（Table 4）；`:700-705`（Table 6）；`:718-723`（Table 8）；`:683-710`（Sec.5.1 实现细节）

**日志/数据**

- `D:/CODE/Doing/WHU_base/CC/WEV1/logs/SCGPD_1000/train_log_20261029_2209.txt:100-102,1218,1223-1254`
- `D:/CODE/Doing/WHU_base/CC/WEV1/logs/SCGPD_1000GD/test_log_20261030_0954.txt:173-202`
- `D:/CODE/Doing/WHU_base/CC/WEV1/logs/SCGPD_1000_domain/domain_protocol_results.csv`
- `D:/CODE/Doing/WHU_base/WEV2/logs/HPD_1000/train_log_20261031_0946.txt:103-105,1741,1744,1748,1780,1810-1821,1826-1857`
- `D:/CODE/Doing/WHU_base/WEV2/logs/HPD_1000GD_test/test_log_20261031_1500.txt:180-209`
- `D:/CODE/Doing/WHU_base/WEV2/logs/HPD_1000/test_log_20261031_1501.txt:207-245`
- `D:/CODE/Doing/WHU_base/WEV2/analysis_hpd_mechanism.md:14-21,39-51,144-182,184-208,210-237,280-304`
- `D:/CODE/Doing/WHU_base/WEV2/analysis_hpd_system.md:14-17,69-117,154-240,254-283`
- `D:/CODE/Doing/WHU_base/WEV2/HPD_final_plan.md:135-156`
- `D:/CODE/Doing/WHU_base/WEV2/review_hpd_final.md:13-17,134,148-168`
- `D:/CODE/Doing/WHU_base/WEV2/review_hpd_code.md:10-12,24-62,132`
- `D:/CODE/Doing/WHU_base/WEV2/review_hpd_code_r2.md:8-11,31-41`

**V3 文档**

- `D:/CODE/Doing/WHU_base/CC/WEV3/README.md:12-34,53-64`
- `D:/CODE/Doing/WHU_base/CC/WEV3/review_pgpd_design.md:10-20,52-113,148-177,181-201`
