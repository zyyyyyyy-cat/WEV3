# WEV3 / TPD 最终审查（review_v3_tpd_final）

> 审查对象：`CC/WEV3/V3_design.md`、`CC/WEV3/README.md`、`CC/WEV3/review_pgpd_design.md`、`CC/WEV3/review_v3_final.md`
> 代码：完整阅读 `CC/WHU`（UAD）、`CC/WEV1/WHU`（GPD-SC）、`CC/WEV2/WHU`（HPD）的 processor / model / loss / datasets / config；推理与 metrics 未改。
> 数据：WEV1/WEV2 的 1000、GD、AG、domain 日志；论文 Table 3/4/6/8、Sec.4.2–4.5、Eq.3–10。
> 标记：**[代码事实]** / **[实验支持]** / **[机制假设]**
> 约束：只审查，不修改项目代码；数值测试只在临时目录。

---

## 0. 结论

**裁决：TPD 的结构 approve；设计点的 RGB/NIR cross-first（0.60/0.20/0.10）approve 为第一轮主方案；TIR 的 0.70/0.15/0.075 不 approve（own 0.70 < V2 0.75、非 own 0.30 > V2 0.25，TIR→TIR 没有改善机制），改为 0.80/0.10/0.05。V3 的目标不是 V1/V2 原样保留，而是保留正面机制 + 弱指标/总体净改善；**加法点 own=1.0 降为 control/ablation，不是第一轮主方案**；Gate 改为 W（弱指标/总体）+ G（正面护栏），不再用 V1 preservation。**

**P0 = 2 个：**

1. **P0-1 分支覆盖**：V2 有 6 处 `gpd_type` 分派 + 1 处诊断分派；`'tpd'` 必须全部走 hierarchical 路径，否则静默落回 HPD/single 或跳过 warm start/bank update；诊断若仍调用 `GPD_HPD`，会静默报告 V2 的两项份额，不是 TPD 的三项份额。
2. **P0-2 模态感知权重的聚合方式**：设计伪代码把“标量 CE（对 M·B 求均值）”直接乘 length-3 权重；这在 PyTorch 里要么返回 [3] 向量、要么退化为“平均权重”，**无法实现 TIR-aware 的逐 query 模态加权**。必须 per-query 或 per-modality 计算 CE，再按 query 模态加权（并除以 M 或使用加权 mean）。

**P1 = 7 个：**

1. **TIR 权重/效果（不是 preservation）**：RGB/NIR 0.60 是允许的 cross-first 取舍；但 TIR 0.70 < V2 0.75、非 own 0.30 > V2 0.25，TIR→TIR 没有改善机制；改 0.80/0.10/0.05（非 own 0.20 < V2 0.25，先保 TIR same，再用 0.10 cross 攻 TIR cross）。AG 无平台项，只做护栏；R5 靠 own。
2. **Gate 自相矛盾**：B 要求 GD 四项都 ≥ V2（14.91/31.61/44.82/54.17），但设计把 global 从 0.25 降到 0.15/0.20、own 也降；没有机制支持 GD 四项同时 ≥V2。A 的 V1 preservation 与用户“允许取舍、改善弱指标”的目标不符；改用 W+G（§5.2）。
3. **P_id/global 的“三层”只是结构层**：`P_id` 是 `id_mean`/`g_cent` 的均值锚（V2 的 `cos(P_id,mean P_scene)=0.999`、有效 shared 10.6%），不是 UAD 的独立单原型；global 项是均值正则，不是新的跨模态判别方向。`w_global=1` 也只有在 `GPD_WARMSTART=False` 时才数值等于 UAD。
4. **zero-column fallback 不够**：V1/V2 的 fallback 是“某 identity 的 K 列全零才整行替换 g_cent”；TPD 的 cross 用的是 `P_scene[:,m']` 单列，若该列缺失但其他列非零，cross 正/负样本 logits 恒为 0，成为“死类”，不是 NaN 但监督错误。1000 影响小，2337 长尾/缺失模态必须处理。
5. **诊断必须是 TPD 专用且 forward 前**：V2 的 `diagnose_gpd_gradients` 调 `GPD_HPD`、读 `GPD_SCENE_WEIGHT`；TPD 必须新写三项 CE/梯度/cos/份额、per-modality own_ratio、per-gallery cross、P_id/P_scene 几何、zero-column、pre-clip/clip，并放在训练 forward 之前（θ_t/B_t）。
6. **归因/seed**：第一轮是 package test；`w_cross=0` 只有 own .75/global .25 才是精确 V2；`w_global=0` 是两层；单 seed 只做 screening；<0.5pp 不能裁决。
7. **配置/卫生**：`GPD_TPD_*` 是 list，长度 M、非负；主方案每 query 和=1.0，但**不要写死断言**（加法 point control 是 1.40/1.20）；`GPD_TYPE` 加 `'tpd'`；`GPD_SUB_CENTERS==M`；`NECK_FEAT='before'`；默认 `single/subcenter` 逐行不变；日志标签 HPD→TPD。

**一句话**：TPD 的“三层读同一个 bank”结构正确、可实现、推理不变；**V3 不是 V1/V2 原样保留，而是保留正面机制 + 弱指标/总体净改善**。第一轮用设计点的 RGB/NIR cross-first + TIR 修正；加法点 own=1.0 是 control；Gate 用 W+G，不要求 V1 每项不降。

---

## 0.1 用户目标澄清后的修订（2026-10-31，覆盖 §4.3/§5.2/§9 的第一轮建议）

> 用户补充：V3 **不是**要原样保留 V1/V2，而是保留其中起正面作用的机制，并改善较弱指标；允许为弱指标做取舍。因此“V1 preservation”不能作为第一轮硬 gate；“加法点 own=1.0”降为归因 control，不是 V3 主方案。

- **V3 主方案（第一轮）= 设计点的 cross-first 权重 + TIR 修正**：
  - RGB/NIR：`0.60/0.20/0.10`（保持设计点；用 own 0.60 换 cross 0.20，攻 cross/九协议）；
  - TIR：`0.80/0.10/0.05`（own 从设计点的 0.70 提回 0.80，高于 V2 的 0.75；global 0.10、cross 0.05×2）；
  - 每 query 合计 1.0。
- **为什么只修 TIR**：设计点 TIR own 0.70 < V2 0.75、非 own 0.30 > V2 0.25，TIR→TIR 没有改善机制；`0.80/0.10/0.05` 的 TIR 非 own 0.20 < V2 0.25，先保 TIR same（V1 的正面），再用 0.10 cross 攻 TIR cross。RGB/NIR 保留设计点的 cross-first，因为 cross/九协议是主要弱项。
- **加法点（own=1.0，global/cross 保持设计点）降为 control/ablation**：用于回答“在 own 不变时，global/cross 是否有效”；它不是第一轮主方案，也不要求 V3 最终保留 V1。若加法点证明 cross 有效，再按设计点降低 own、提高 cross 的相对份额；若加法点 cross 都无效，说明 cross 机制本身弱，改 own 权重也不会有效。
- **Gate 改为“弱指标/总体净改善 + 正面机制护栏”**，不再用 A=V1 preservation；见 §5.2 修订表。
- **AG**：TPD 没有平台项，权重无法改善 AG；AG 只做护栏（不低于 V2−0.2）。若 AG 是必须改善项，第二轮加平台机制，不在 TPD 权重里硬调。
- **R5/GD R5**：R5 靠 own（0.60/0.80），GD R5 靠 global（0.20/0.10）；第一轮只要求不低于 V2−0.2，paper+1 仍 Tier2。

---

## 0.2 独立 agent 复审后的残余修正（2026-10-31）

> 独立 agent 报告：`CC/WEV3/review_v3_tpd_agent_final.md`。裁决 approve with conditions；P0=2/P1=6。以下只列“仍未关闭/新发现”的实质修正；P0-1/P0-2 与主方案本身不变。

- **P1-1 Gate 严格度**：主方案 RGB/NIR own 0.60、TIR own 0.80 是有意取舍；用 V2−0.2 / V1−0.3 做第一轮硬护栏，很可能因为 V2 用 own 0.75 已付出 AG −0.37/−0.52、R5 −0.66 而直接判死。**第一轮建议 screening**：W 的 overall/cross/TIR 仍是 hard；G 的 same/AG/R5/R10/GD 放宽到 `V1−0.5 / V2−0.5`；严格 V2−0.2 / V1−0.3 留第二轮（先 own 0.60→0.65 或 TIR 0.80→0.85 单变量）。W 的 TIR 方向补 **NIR→TIR 7.03/5.29**。
- **P1-2 诊断份额**：TPD 的 per-modality 权重不能“标量权重 × 未加权梯度范数”；必须 per-term/per-modality 加权梯度（或 per-modality 梯度加权）后再算份额；诊断仍 forward 前、last-block+norm/eval/rank0 代理。
- **P1-3 cross fallback 独立性**：per-column fallback 不能作用在 `L_own` 的同一个 `P_scene` 上，否则 `max_k` 几何改变，`cross=0` 不再精确等于 V2。保留 own/global 的 row-level fallback；cross 用独立 `P_scene_cross` clone 或 `valid_scene[C,K]` + per-column fallback/mask；记录 missing count。
- **P1-4 V3_design 正文**：§4.2/§7/§8.2/§9.2/§9.3/§9.4/§10.1/§11/§13 仍是旧 TIR 0.70、旧 Gate、6 处、标量×length-3；实现前必须按 §0.1/§0.2 更新或明确废弃，否则会被照抄。
- **P1-5 加性系数**：V1/V2 直接比 `overall/mean₉ = 11.47/12.761=0.8988`（V1 0.8984），不是 0.908；**cross 平均** +0.3 → mean₉ +0.2 → overall +0.18；overall +0.4 需 cross 平均 +0.66。Agent 的 +0.27/+0.45 是把“cross 平均”当成“mean₉”算了，不采纳；定性结论“不能同时写 cross+0.3 & overall+0.4”仍成立。文档把 0.908 改成“≈0.90（直接比 0.898；历史回归 0.908）”。
- **P1-6 control**：加法 control 若是主方案的 own 消融，TIR 应为 `1.0/0.10/0.05`（总 1.20），不是旧设计点的 `1.0/0.15/0.075`；RGB/NIR 仍 `1.0/0.20/0.10`（总 1.40）。这样 control 与主方案只差 own（1.0 vs 0.60/0.80），归因干净；若保留旧值，必须标注“design-point control”，不能用来归因主方案 TIR。

---

## 1. 代码链路核对

### 1.1 UAD（`CC/WHU`）

| # | 事实 | 证据 |
|---|---|---|
| 1 | `PCA` 返回 `loss, ids, g_cent`；`ids = target.reshape(-1, group_size)[:,0]`，`g_cent` 是 M 个模态中心的均值并 detach | `processor/processor.py:13-30` |
| 2 | `PrototypeBank` 零初始化 `[C,D]`；`update` 用 `g_cent` 做 EMA 0.2；无 warm start | `:35-71` |
| 3 | `GPD`：`f@P.T/τ`，CE；只有本 batch 出现的 identity 若原型为零，用 `g_cent` 替换 | `:74-88` |
| 4 | 训练总损失 `CE+WRT+0.01·ProCA+1.0·GPD`；推理 pre-BN CLS + L2 + 欧氏 | `:187-227`；`utils/metrics.py` |

**结论**：UAD 的“全局锚点”是**零初始化、首 batch g_cent fallback、之后 g_cent EMA** 的单原型；不是“warm-started id_mean + 均值正则”。`w_global=1` 要精确等于 UAD，必须 `GPD_WARMSTART=False`，且 own/cross=0。

### 1.2 WEV1（`CC/WEV1/WHU`）

| # | 事实 | 证据 |
|---|---|---|
| 1 | `SubCenterPrototypeBank`：`sub_centers[C,K,D]`，`counts[C,K]`；K=3 对应 RGB/NIR/TIR，不是聚类 K=4 | `processor.py:76-162` |
| 2 | 更新：ProCA 模态中心 + count EMA `clamp(1/(count+1),0.1,0.5)`；缺失列整行 g_cent fallback | `:109-162` |
| 3 | warm start：每 `(pid,modality)` 类均值；缺失回退 `id_mean`；counts=1；rank0+broadcast；RNG 保存/恢复 | `:182-231` |
| 4 | `GPD_SC`：`einsum('nd,ckd->nck')`，`max(dim=2)`，CE；`target.repeat(M)` | `:234-251` |
| 5 | 6 处分派：类型、bank、warm start、loss、update、counts 日志 | `:317-319,323-348,349-371,413-432,446-460,499-511` |
| 6 | 训练总损失同 UAD；推理不变 | `:397-460` |

**结论**：V1 的 `max_k` 是“在 3 个**模态**中心上取 max”，不是“每个模态 4 个聚类子中心”。`own_ratio` 是 V2 的诊断，V1 没有；V2 的 0.92–1.0 是后期值，TIR=1.0。

### 1.3 WEV2（`CC/WEV2/WHU`）

| # | 事实 | 证据 |
|---|---|---|
| 1 | `HierarchicalPrototypeBank`：`P_id[C,D]` + `P_scene[C,K,D]`；两个 buffer | `processor.py:165-230` |
| 2 | warm start：`_warm_start_centers` + `warm_start_hierarchical`；P_scene 模态均值，P_id=valid id_mean；counts=1；broadcast | `:250-337` |
| 3 | `GPD_HPD`：`0.75·CE_scene + 0.25·CE_id`；P_scene 的 max_k 与 P_id 各一个 CE；diag own_ratio/usage/stats | `:360-435` |
| 4 | 诊断：`diagnose_gpd_gradients`，last-block+norm、eval、rank0；调 `GPD_HPD`、读 `GPD_SCENE_WEIGHT` | `:438-525` |
| 5 | 训练循环：诊断在训练 forward **之前**（`n_iter` 条件、`optimizer.zero_grad` 后）；loss 分支、update、epoch 末几何日志 | `:710-738,756-791,809-833,888-941` |
| 6 | 类型/bank/warm start/守卫：允许 `single/subcenter/hierarchical`；`GPD_TYPE != 'single' and 'gpd' not in loss_type` 抛错；K==M 检查 | `:596-680` |

**结论**：TPD 从 V2 复制是对的；但 V2 的 6 处分派之外，第 7 处是**诊断**（710-738），设计文档的“6 处”漏了它。`GPD_HPD` 的 diag 是 V2 的，不是 TPD 的。

### 1.4 数据与 batch 顺序

- **[代码事实]** `PKMSampler` 每个 batch 是 P 个 identity × K=4 instances × M 模态；`train_collate_fn` 把每个模态 stack 成 `imgs[m]`，`pids` 是 [B]；每个模态的 B 相同。`make_model.forward(mode=0)` 用 `torch.cat(list(x),0)`，因此 `global_feat` 是 modality-major `[RGB B; NIR B; TIR B]`。`target.repeat(M)` 也是 modality-major。  
  证据：`datasets/sampler.py:105-140`；`datasets/make_dataloader.py:17-23`；`model/make_model.py:67-80`；`processor.py:707-708`。
- **[代码事实]** `B = target.shape[0]` 在现有 PKM 配方下等于每个模态的 batch size；TPD 的 `q = logits[m*B:(m+1)*B]`、`t = target_rep[m*B:(m+1)*B]`、`q[:,:,m']` 与该顺序一致。若以后改 sampler/最后 batch 不齐，必须用 `[x.shape[0] for x in imgs]` 计算分段，不能假设 B。
- **[代码事实]** 训练用 `model(imgs, target, camids)`（mode=0）；推理用 `model(img, mode=mode)`，`NECK_FEAT='before'` 返回 pre-BN CLS。TPD 的 bank/损失不进入推理。

---

## 2. TPD 的数据流、梯度流与损失聚合

### 2.1 形状与索引

- `f [M·B,768]`，`P_id [C,D]`，`P_scene [C,K,D]`，K=M=3。
- `logits_id = f @ P_id.T / τ` → `[M·B,C]`。
- `logits_scene = einsum('nd,ckd->nck', f, P_scene) / τ` → `[M·B,C,K]`。
- `L_own` 用 `logits_scene.max(dim=2).values` → `[M·B,C]`，CE。
- `L_cross^{m→m'}` 用 `logits_scene[mB:(m+1)B, :, m']` → `[B,C]`，CE；`m'≠m`。
- `P_id/P_scene` 是 buffer，`.clone()` + `no_grad` + `F.normalize`；三项 CE 只回传到 `f`/ViT；bank update 在 `optimizer.step()` 之后。**[代码事实]**（同 V1/V2 的 `GPD_SC`/`GPD_HPD`）

**结论**：形状、索引、modality-major 对齐正确；没有发现 TPD 公式层面的 P0。

### 2.2 P0-2：模态感知权重不能乘在“全局标量 CE”上

设计的伪代码是：

```python
loss_own = F.cross_entropy(logits_scene.max(...), target_rep)   # 标量，对 M·B 求均值
loss_global = F.cross_entropy(logits_id, target_rep)            # 标量
loss = (w_own * loss_own + w_global * loss_global + ...)        # w_own 是 length-3
```

`loss_own` 是**一个标量**；`w_own` 是 length-3。PyTorch 里 `w_own * loss_own` 返回 `[3]`，`.sum()` 等于 `(w_own[0]+w_own[1]+w_own[2])·loss_own`，即“权重之和 × 全 batch 均值”，**不是**“RGB 用 w[0]、NIR 用 w[1]、TIR 用 w[2]”。如果实现时先 `mean`/`sum` 再乘，TIR-aware 权重就静默退化为平均权重；这是核心机制的 P0。

**数值示例**（临时 CPU，仅说明聚合）：设 per-modality own CE 为 `[1.0, 1.0, 3.0]`，设计权重 `[0.6, 0.6, 0.7]`：

- 正确 per-query/per-modality：`(0.6·1 + 0.6·1 + 0.7·3)/3 = 1.10`；
- 标量均值 `1.667`；`sum(w)·mean = 1.9·1.667 = 3.167`；即使再除以 3 也是 `1.056`，仍不是 1.10，且 TIR 的 3.0 被平均权重稀释。

**正确实现（二选一，必须写进代码/文档）**：

1. per-query：`F.cross_entropy(..., reduction='none')` 得到 `[M·B]`；把 `w_own/w_global/w_cross` 展开成 `[M·B]`（每 B 个重复同一模态权重）；`loss = (w_own_q * own_loss + w_global_q * global_loss + w_cross_q * cross_sum).mean()`。这自动是“每 query 权重”，总平均为各模态权重的平均。
2. per-modality：对每个 m 计算 `CE_m`（mean over B），`loss = sum_m [w_own[m]·own_m + w_global[m]·global_m + w_cross[m]·sum_{m'≠m}cross_{m,m'}] / M`。因为每个模态 B 相同，这与 per-query 等价；`w_own=1` 时严格等于 V1 的 `GPD_SC`（全 batch CE 均值）。

**同时要明确总尺度**：主方案每 query `own+global+2·cross=1.0`，全 batch 平均也是 1.0；加法 control（§4）每 query 1.40/1.20，全 batch 平均约 1.33。**不要在代码里断言权重和=1.0**；control 会被误杀。只校验 length=M、非负、每 query 和 >0。

### 2.3 数值锚点

| 配置 | 与谁等价 | 条件 |
|---|---|---|
| `w_own=1, w_global=0, w_cross=0` | V1 `GPD_SC` | P_scene bank/warm start/update 与 V1 相同；TPD 的 P_id 仍更新但不进 loss，数值不影响 |
| `w_cross=0, own=0.75, global=0.25` | V2 `GPD_HPD` | P_id/P_scene 同 V2；此时是精确 V2，不是“V2-rebalanced” |
| `w_global=0` | 两层 PGPD | 无 global 项；P_id bank 仍更新但不进 loss |
| `w_global=1, own=cross=0` | UAD `GPD`（目标等价） | **必须 `GPD_WARMSTART=False`**；否则 P_id 是 warm-started id_mean，不是 UAD 的零初始化。TPD 用 float32，UAD 原版 `GPD` 未显式 `.float()`，AMP 下可能 fp16；是目标/结构等价，不是逐位相等 |
| `w_cross=0, own=0.6, global=0.2` | `0.6·scene+0.2·id` | 是 V2 的**缩放/不同权重**，不是 V1，也不是 V2 |

**设计文档的“`w_cross=0` = V2-rebalanced”与“`w_global=1` = UAD”需要加条件**；否则 smoke/归因会误判。

### 2.4 zero-column（P1-4）

`GPD_SC`/`GPD_HPD` 的 fallback 是：

```python
norms = P.norm(dim=2)                 # [C,K]
need = norms[ids].sum(dim=1) < 1e-6   # [P]，整行全零才替换
```

- 若某 identity 的 `P_scene[c,m']` 缺失（零），但 `P_scene[c,m]` 非零，**不会**触发 fallback；`F.normalize` 后该列仍是零。
- 对 `max_k`：零列 logit=0；如果真实列的 cosine 为负，零列可能被 max 选中，own CE 变成“全类 logit=0”的常数，梯度为零；这是 V1/V2 已有的边缘问题。
- 对 `cross`：`u·0=0`，该 gallery 类对所有 query 的 logit 恒为 0，成为“死正/死负”；不是 NaN，但监督错误。
- 1000 三模态齐全时影响小；2337 的长尾/缺失模态必须处理。**建议**：为 cross 建 `valid_scene[C,K] = norms>0`，对 `m'` 列缺失的 identity 用 `g_cent` 替换（per-column），或直接 mask 掉该 `(m,m')` 的 cross 项并记录 missing count；不要改默认 `single/subcenter` 的路径，避免破坏 V1/V2 回归。

---

## 3. 机制、监督、冲突

### 3.1 三项都是全局 CE

- `L_own`：正样本 `P_scene[y]` 的 max_k；负样本全部 C 个 identity 的 3 列。全局，不是 batch-local。
- `L_global`：正样本 `P_id[y]`；负样本全部 C。全局。
- `L_cross^{m→m'}`：正样本 `P_scene[y,m']`；负样本全部 C 的同一列。全局。

**监督充分、覆盖 3×3 的 9 个 query/gallery 组合**；没有“监督对象 ≠ 推理特征”（都用 pre-BN CLS）。这是 TPD 相对 batch-local 排序损失的结构优势。

### 3.2 P_id 不是独立的第三个源（P1-3）

- `warm_start_hierarchical` 的 `P_id` = valid `id_mean` = 模态均值的均值；`GPD_HPD/TPD` 的 `P_id` 更新用 `g_cent` = 当前 batch 的模态中心均值。**它从初始化到更新都是“同一组 P_scene 的均值”**，V2 的 `cos(P_id, mean P_scene)=0.999` 是证据。
- 因此 `L_global` 是“均值锚正则”，不是 UAD 的独立单原型判别；它和 `L_own`、`L_cross` 高度相关。V2 的名义 shared 25% 只有 10.6% 有效份额；TPD 的 global 名义 0.10/0.075（加法点）或 0.15/0.20（设计点）可能只有个位数有效份额。
- **“三层”在结构上成立，在功能上要由诊断回答**：如果 `g_global` 的独立份额 <5%、`cos(g_global,g_own/cross)` 很高，global 就不是第三个判别方向，只是均值正则；第二轮可考虑去掉 global（两层）或改 P_id 更新目标。
- **`w_global=1` 的 UAD 锚点**必须 `GPD_WARMSTART=False`；否则是“warm-started 均值 + 单原型 CE”，不是 UAD 原始 GPD。

### 3.3 own 与 cross 的冲突

- RGB/NIR 的 `own` 与 `cross` 正样本列接近（pair cos 0.914），冲突小；TIR 的 TIR 列与 RGB/NIR 列 pair cos 0.809，冲突大。
- `max_k` 在 TIR 上 `own_ratio=1.0`（V2 末期），own 永远选 TIR 列；cross 是唯一把 TIR 拉向 RGB/NIR 的力。**“max_k 保护 TIR”不成立**：它只说明 own 选 own，不说明 own 的权重大到能抵消 cross。
- 对 RGB/NIR，`own_ratio` 约 0.92–0.94；约 6–8% 的样本 own 选到别的模态列，此时 own 与某个 cross 对同一列重复施力；名义权重不等于有效权重，必须用 `own_ratio` 和 `cos(g_own,g_cross_m')` 诊断。
- **AG platform_only 没有平台项**：AG query/gallery 同模态、跨平台；cross 是“模态级”，不解决平台内 top-1。V2 的 AG −0.37/−0.52 是证据；TPD 若把 own 降到 0.60，AG 只会更差，不能靠 cross 补。这是设计点不能“守 V1 AG”的机制原因。

### 3.4 与 CE/WRT/ProCA 的关系

- CE 走 BNNeck/classifier；WRT 在 pre-BN `global_feat` 上做跨模态正对；ProCA 拉模态中心到 g_cent；TPD 三项都在 pre-BN CLS 上做全局原型 CE。监督对象一致，梯度都回 ViT。
- WRT 已经做跨模态正对，TPD 的 cross 是全局原型版；V2 的 cross 只有 +0.16 mAP，说明边际收益会被 WRT/CE 吃掉一部分；TPD 的 cross 更具体，但仍不是 AP/CMC 的直接优化。
- 三项与 ProCA 同向（都把模态中心拉近），但 TIR 的 cross 与 own 有张力；这是 TPD 的主要局部冲突，不是“模块坏掉”。

---

## 4. 权重设计审查（核心）

### 4.1 三条基线

| 方案 | own（RGB/NIR/TIR） | global | cross（每个 m'） | 每 query 合计 |
|---|---:|---:|---:|---:|
| UAD | 0 | 1.0 | 0 | 1.0 |
| V1 | 1.0 | 0 | 0 | 1.0 |
| V2 | 0.75 | 0.25 | 0 | 1.0 |
| 旧设计点（TIR 不采用） | 0.60/0.60/0.70 | 0.20/0.20/0.15 | 0.10/0.10/0.075 | 1.0 |
| **V3 主方案（设计点 RGB/NIR + TIR 修正）** | **0.60/0.60/0.80** | **0.20/0.20/0.10** | **0.10/0.10/0.05** | **1.0** |
| 加法 control（own 消融，非主方案） | 1.00/1.00/1.00 | 0.20/0.20/0.10 | 0.10/0.10/0.05 | 1.40/1.40/1.20 |

### 4.2 旧设计点的 TIR 自相矛盾

以 TIR 为例：

- V2 TIR：own 0.75，global 0.25，cross 0；TIR→TIR **21.12/14.55，R1 −1.22**；TIR `own_ratio=1.0`。
- 设计 TIR：own **0.70**（−0.05），global 0.15 + cross 0.15 = 非 own **0.30**（+0.05），cross 是拉向 RGB/NIR 的新力。
- 所以设计点在 TIR 上是“更少的 own、更多的非 own、且非 own 里有一半是更远的 gallery 列”；它没有任何机制让 TIR→TIR **不低于** V2，反而更可能更低。设计 §10.1 的“预期 TIR→TIR 不低于 V2 −1.22”与 §4.2 的权重矛盾。

以 RGB/NIR 为例：

- V2：own 0.75，global 0.25；AG −0.37/−0.52，R5 −0.66。
- 设计：own **0.60**（−0.15），global 0.20 + cross 0.20 = 非 own **0.40**（+0.15）。
- AG 是平台级、同模态，cross 不解决平台；R5/R10 靠 own；设计把 own 降得比 V2 更多，却要求 AG≥V1−0.3、R5≥47.3；没有机制支撑。V2 已经用 0.75 付出 AG/R5 代价，设计 0.60 只会更大。

以 GD R5 为例：

- V2 的 GD R5 44.82 主要来自 global 0.25（UAD 的 GD R5 46.14 也是单全局原型）；设计 global 0.15/0.20，own 更低；设计 B 却要求 GD 四项 ≥V2（甚至 GD R5 46.0 是 Tier2）。这在权重上没有支撑；能要求的是 GD ≥V1−0.2。

**根因**：设计把“每 query 总 GPD=1.0”当成硬约束。要加 cross/global，就必须从 own 里扣；一扣 own，V1 的 same/AG/R5/TIR 就保不住。**“保留 V1 + 加 cross”与“总 GPD=1.0”不能同时成立**。这不是调参问题，是归一化约束与 preservation 目标的结构矛盾。

### 4.3 加法点（control/ablation，不是第一轮主方案）

> **§0.1 修订**：第一轮主方案是“设计点 + TIR 修正”（RGB/NIR 0.60/0.20/0.10；TIR 0.80/0.10/0.05；总 1.0）。本节加法点降为 **归因 control**：用于回答“own 不变时 global/cross 是否有效”；若预算只允许一个 120ep，不跑它。

**只改一个变量：把 own 从主方案的 0.60/0.80 提回 V1 的 1.00；global/cross 与主方案完全相同。总 GPD 1.40/1.20，不归一化到 1.0。** 这是“V1 own + 主方案 global/cross”的直接加法，也是相对主方案最容易归因的 control。

| query 模态 | w_own | w_global | w_cross（每个 m'） | 每 query 合计 |
|---|---:|---:|---:|---:|
| RGB | 1.00 | 0.20 | 0.10 | 1.40 |
| NIR | 1.00 | 0.20 | 0.10 | 1.40 |
| TIR | 1.00 | 0.10 | 0.05 | 1.20 |

- **own 绝对值 = V1**：same/AG/R5/R10/TIR 的 own 力不被削弱；cross/global 是额外正则。
- **global/cross 与主方案相同**（RGB/NIR 0.20 + 0.10×2；TIR 0.10 + 0.05×2）：与主方案的唯一差别是 own 1.0 vs 0.60/0.80，归因干净。
- **总 GPD 1.40/1.20**：这是“加法”的代价；V2 的 GPD 相关有效份额约 34%（scene 23.3 + shared 10.6，另有约 0.3 的 PCA），加法点按总尺度粗略估算会升到约 42%、base 降到约 58%，但实际份额以诊断为准；记录 pre-clip/clip、base/PCA/GPD 份额。
- **如果 clip/不稳定**：先降 global/cross（例如 RGB/NIR 0.15/0.075×2；TIR 0.10/0.05×2），**不要先降 own**；否则又回到设计点的矛盾。降 RGB/NIR global/cross 后总 GPD 约 1.30/1.20，仍高于 V1；base 份额回升。
- **如果 GD R5 不够**：global +0.05（RGB/NIR 0.25；TIR 0.15），只改 global；**如果 cross 不够**：先 own 1.0→0.90/0.85 单变量（保持 global/cross 绝对值），最后才到主方案 0.60/0.80。

**为什么不是“own=0.75 + cross 0.125”**：那还是 V2 的 own，V2 已经丢了 TIR/AG/R5；要保 V1 的正面，own 必须高。加法点是“own=1.0 + 主方案 global/cross”的线性组合；它的代价是总 GPD 变强，而不是 own 变弱。

### 4.4 第二轮（按第一轮结果，单变量）

- **W 的 cross/TIR 不够**：先只加 cross +0.05（单变量）；若仍不够，说明 cross 机制本身弱，不要靠继续降 own 硬拉。
- **W 的 TIR 回退**：TIR own 0.80 保持，降 TIR cross/global（或 TIR 只保留 own；cross 对 TIR 先关）；这是 TIR-specific，不是全局。
- **G 的 AG 回退**：AG 是平台级，cross 无关；只能保持/提高 RGB/NIR own 或加平台项（不在第一轮）。AG 的损失不是 cross 造成，不要把 AG 混进 TPD 权重调参。
- **W 的 GD R5 不够**：global +0.05（只加 global），不同时加 cross。
- **G 的 same/R5 低于护栏**：RGB/NIR own 0.60→0.65 或 TIR 0.80→0.85，单变量；不要同时降 cross。

### 4.5 设计点的定位（§0.1 修订后）

设计点是 **cross-first 的 V3 主方案**：RGB/NIR `0.60/0.20/0.10` 保留设计点的 cross 预算；TIR 必须从 `0.70/0.15/0.075` 修正为 `0.80/0.10/0.05`，否则 TIR own 低于 V2、非 own 高于 V2，TIR→TIR 没有改善机制。加法点（own=1.0）是 control，不是“V3 必须守 V1”的轮次。设计点的 RGB/NIR 会承担 AG/R5 风险；这是“用部分 V1 正面换 cross/九协议”的取舍，用 §5.2 的护栏判断是否可接受；若不接受，再把 RGB/NIR own 从 0.60 提到 0.65（单变量）。

---

## 5. Gate 审查

### 5.1 现有 gate 的问题

- **A**：same 3 对 ≥V1−0.2、AG ≥V1−0.3、overall R1≥30.5、GD 四项 ≥V1−0.2、TIR ≥V1−0.5。**没有 R5/R10**；设计 §0/§10 说“守 V1 R5/R10”，但 A 不检查。
- **B**：overall mAP >11.47、cross >13.72/9.16、**GD 四项 ≥V2**、R5≥47.3/R10≥55.8。GD 四项 ≥V2 与设计的 own/global 下降矛盾；R5 47.3/R10 55.8 低于 V1 48.17/56.27，不能叫 preservation。
- **C**：paper+1 12.0/30.5/47.3/55.8、GD 15.5/32.9/46.0/53.5、GD R5≥46.0。这是 Tier2，第一轮不承诺。
- **三套数不一致**：B 的 R5≥47.3 与 A 的“守 V1 R5”不是同一件事；B 的 GD≥V2 与 A 的 GD≥V1−0.2 也不是同一件事。需要 pre-registered 一张表。

### 5.2 建议的 gate（§0.1 修订：弱指标/净改善 + 正面护栏）

> 目标不是“V1 每项不降”，而是“弱指标改善 + 总体净胜 + 正面机制不塌”。因此不再设 A=V1 preservation；改用 W（weak/net targets）+ G（guardrails）。

**W. 弱指标/总体（hard targets）**

- overall mAP/R1 > V2 `11.47/30.52`；
- cross R1/mAP > V2 `13.72/9.16`；
- TIR→TIR R1/mAP ≥ V2 `21.12/14.55`，target V1 `22.34/14.70`；TIR→RGB `7.59/5.98`、RGB→TIR `7.26/5.71`、TIR→NIR `8.06/5.39`、**NIR→TIR `7.03/5.29`** 不低于 V2；
- AG/R5/R10/GD 不在 W 里做硬 gate，见 G 的 screening（第一轮）/严格（第二轮）；AG 不要求 V1——TPD 没有平台项；
- paper+1 仍是 Tier2，不是第一轮承诺。

**G. 正面机制护栏（第一轮 screening；严格版第二轮）**

- **第一轮 screening**（S22/agent 建议，避免用权重无法满足的护栏判死）：same 3 对 ≥ V1−0.5（≥30.66/19.47）；AG/R5/R10/GD ≥ V2−0.5（AG ≥34.24/31.64；R5 ≥47.01；R10 ≥55.64；GD R5 ≥44.32；GD mAP/R1/R10 ≥14.41/31.11/53.67）。
- **第二轮严格版**（若第一轮 W 达标、G 在 screening 内）：same ≥ V1−0.3（≥30.86/19.67）；AG/R5/R10/GD ≥ V2−0.2；再先调 own（RGB/NIR 0.60→0.65 或 TIR 0.80→0.85），单变量。
- 若 W 的 cross/TIR 不达标，说明 TPD 的 cross 机制本身不够，不是权重问题；若 G screening 都失败，第二轮先调 own；不要在中间同时改 cross/global。

**Tier2（paper+1，第一轮不承诺）**

- 1000 12.0/30.5/47.3/55.8；
- GD 15.5/32.9/46.0/53.5；GD R5≥46.0。

**加性关系**：`Δoverall mAP≈0.90×Δmean₉`（V1/V2 直接比 0.898；历史回归 0.908）；same 保持时 **cross 平均** +0.3 → mean₉ +0.2 → overall +0.18；要 overall +0.4 需 **cross 平均** +0.66，或 same 上涨；不能同时写 `cross+0.3 & overall+0.4`。

### 5.3 为什么 B 的 GD 四项 ≥V2 要删

- V2 的 GD R5 44.82 主要来自 global 0.25；TPD 加法点 global 0.20/0.15，设计点 global 0.20/0.15，都低于 0.25。
- TPD 的 cross 是模态级；GD 是 ground/daytime 同模态子集，cross 不直接帮 GD R5。
- own 从 V2 的 0.75 提到 1.0（加法点）可能帮 GD R1/R10，但不帮 R5 的“覆盖/深度”。
- 因此 B 应把 GD 四项 ≥V2 降为“GD 不低于 V1−0.2”，把 46.0 放 Tier2；否则第一轮会被一个与权重设计矛盾的 gate 判死。

---

## 6. P0/P1 清单（实现前必须关闭）

### P0-1：分支覆盖（6 + 1 处）

V2 的实际分派点（`WEV2/WHU/processor/processor.py`）：

| # | 位置 | 作用 | TPD 必须 |
|---|---|---|---|
| 1 | `597-600` | 允许类型 | 加 `'tpd'` |
| 2 | `608-646` | bank 创建 | `'tpd'` → `HierarchicalPrototypeBank` |
| 3 | `647-680` | warm start | `'tpd'` → `warm_start_hierarchical`；rank0+broadcast P_id/P_scene/counts |
| 4 | `710-738` | 诊断 | `'tpd'` → `diagnose_gpd_tpd_gradients`；不能调 `GPD_HPD` |
| 5 | `756-791` | loss | `'tpd'` → `GPD_TPD` |
| 6 | `809-833` | bank update | `'tpd'` → `HierarchicalPrototypeBank.update`（id + scene） |
| 7 | `888-941` | epoch 末 counts/geometry | `'tpd'` → hierarchical 分支；日志标签改 TPD |

- 守卫：`GPD_TYPE != 'single' and 'gpd' not in loss_type`；`GPD_SUB_CENTERS == len(MODALITIES)`；`TEST.NECK_FEAT='before'`；`GPD_WARMSTART=True` 时 `'tpd'` 必须走 warm start。
- smoke 断言：`type(proto_bank).__name__ == 'HierarchicalPrototypeBank'`；`GPD_TPD` 被调用（monkeypatch 计数）；warm start 日志/counts；`GPD_HPD` 不被调用。
- **为什么 P0**：静默落回 HPD/single 不会报错；120ep 跑完才发现机制不是 TPD，这是最贵的失败模式。

### P0-2：模态感知权重的逐 query/per-modality 聚合

- 设计伪代码的“标量 CE × length-3 权重”不可实现；必须按 §2.2 的 per-query 或 per-modality 方式。
- 单测：per-modality CE 为 `[1,1,3]`、权重 `[0.6,0.6,0.7]` 时，正确 loss 1.10，不是 3.167/1.056；`own=1` 时与 `GPD_SC` 数值相等；TIR 的权重变化必须真的影响 TIR 的梯度（用假 batch 检查）。
- **为什么 P0**：TIR-aware 是 TPD 的核心；聚合错了就退化成“平均权重 TPD”，第一轮结果无法解释。

### P1-1：主方案权重/指标风险（§4）

- RGB/NIR own 0.60 < V2 0.75、非 own 0.40 > 0.25：same/AG/R5 有回退风险；TIR own 0.80 > V2 0.75、非 own 0.20 < 0.25：TIR same 有改善机制，但 cross 两列更远，TIR cross 幅度有限。
- 修复：不是改回“V1 preservation”；第一轮 W 硬、G screening（V1−0.5/V2−0.5）；严格 V2−0.2/V1−0.3 留第二轮；若 G screening 失败，先调 own（RGB/NIR 0.60→0.65 或 TIR 0.80→0.85），单变量。

### P1-2：Gate 定义（§5）

- 用 W+G；不再 A/B/C V1 preservation；W 的 TIR 四向补 NIR→TIR 7.03/5.29；G 第一轮 screening、第二轮严格；GD R5 46.0 归 Tier2；`cross+0.3 & overall+0.4` 不同写；加性系数用 0.90（直接比 0.898），cross 平均 +0.3 → overall +0.18。

### P1-3：P_id/global 的功能层（§3.2）

- P_id 是 id_mean/g_cent 的均值锚，不是 UAD 独立单原型；global 是均值正则；有效份额要实测。若 `g_global` 独立份额 <5%，三层在功能上不成立，第二轮考虑去掉/改 P_id。
- `w_global=1` 的 UAD 锚点必须 `GPD_WARMSTART=False`；否则只是“warm-started 均值 + 单原型 CE”。

### P1-4：cross 的 zero-column（§2.4）

- 现有 row-level fallback 不足；per-column 缺失会让 cross 死类；加 `valid_scene[C,K]`、per-column `g_cent` fallback 或 mask；记录每模态 missing count；1000/2337 都跑一次。
- **cross 的 fallback 必须独立于 `L_own` 的 P_scene**：用 `P_scene_cross` clone 或 mask；否则 `max_k` 几何改变，`cross=0` 不再精确等于 V2。own/global 保留 V1/V2 的 row-level fallback。

### P1-5：诊断（TPD 专用、forward 前）

- 新 `diagnose_gpd_tpd_gradients`：
  - per-term CE：own、global、cross(m→m') 6 项；
  - `g_own/g_global/g_cross_m'` norm、两两 cos、有效份额；**必须按 per-modality 加权梯度计算，不能“标量权重 × 未加权范数”**；
  - per-modality own_ratio、usage、cross CE；
  - P_id/P_scene pair cos、per-column norm、zero columns、g_cent_norm；
  - pre-clip/clip；per-pair R1/mAP、TIR、AG、GD；
  - 诊断在训练 forward 之前（θ_t/B_t）；last-block+norm、eval、rank0 代理要标注；不要用 diag loss 代替训练 loss。
- 训练循环的 meter/日志名从 HPD 改 TPD；`gpd_diag` 不能复用 HPD 的返回结构。

### P1-6：归因/seed

- 主方案第一轮是 package：RGB/NIR own 0.60 + TIR 0.80 + global/cross；不能把指标变化单独归因给 cross。
- 加法 control 是主方案的 own 消融：own=1.0；RGB/NIR 1.0/0.20/0.10（1.40）；TIR 1.0/0.10/0.05（1.20）；只差 own。
- 精确锚点：V1 `own=1,cross=0,global=0`；V2 `cross=0,own=.75,global=.25`；two-level `global=0`；UAD `global=1, warmstart=False`。
- 若预算允许，1ep/短跑 `own=1,global=.10,cross=0` 与 `own=1,global=0,cross=.20` 做梯度/CE 对照；否则第一轮只写 package + 梯度诊断。
- 单 seed screening；最终 2–3 seed + paired bootstrap；<0.5pp 不裁决。

### P1-7：配置/代码卫生

- `config/defaults.py` 加 `GPD_TPD_OWN_WEIGHT/GLOBAL_WEIGHT/CROSS_WEIGHT`（list，默认主方案或 V1）；validate length=M、非负；**不要断言和=1.0**（加法 control 1.40/1.20）。
- `GPD_TYPE` 允许 `'tpd'`；`GPD_SUB_CENTERS==M`；`GPD_WARMSTART` 分支；`GPD_DIAG_PERIOD`；`LOSS_TYPE` 守卫。
- `configs/TPD_UAD.yml`：加法点权重、`GPD_SHARED_MOMENTUM=0.2`、`GPD_DIAG_PERIOD=100`、`OUTPUT_DIR=./logs/TPD_1000_DDP`；`GPD_SCENE_WEIGHT` 不用于 TPD（保留兼容）。
- 默认 `single/subcenter` 路径逐行不变；`model/loss/datasets/utils/solver` 不改；推理不变。
- 1ep smoke：类型断言、三项 CE 有限、逐 query 权重生效、bank counts、warm start、zero-column、DDP all-gather、`single/subcenter` 回归、无 NaN、pre-clip/clip。

### P2（不阻塞）

- 论文配方（每卡 192/LR 0.008）vs repo 配方（batch 256/LR 0.032/4 卡）；V1/V2/TPD 可比，但不要拿论文绝对值当同配方。
- bank 不在 `state_dict`；one-shot 训练；resume 不恢复 bank。
- 2337 等 1000 gate 之后；长尾/缺失模态、bank 大小、warm start 回退。
- 不在第一轮加平台 token、TIR adapter、EMA/TTA、PCGrad、硬分配、CA-WRT、LLRD、margin、双温度。
- `own_ratio` 0.92–1.0 是 V2 末期诊断，不是 V1；V1 没有该日志。
- `F.normalize` 零列、`max_k` 可能选零列是 V1/V2 既有边缘；TPD 不要在默认路径改它，只在 cross 侧 mask。
- 日志标签 HPD→TPD；`GPD_SCENE_WEIGHT` 在 TPD 下未使用，避免误读。

---

## 7. 第一轮实现/实验清单（修正后）

### 7.1 实现

- [ ] `WEV3/WHU` 从 `WEV2/WHU` 复制；只改 `processor/processor.py`、`config/defaults.py`、新增 `configs/TPD_UAD.yml`、`README_TPD.md`。
- [ ] `GPD_TPD(feat,target,bank,ids,g_cent,num_modalities,tau,w_own,w_global,w_cross)`：
  - `P_id/P_scene` clone + no_grad + normalize；zero fallback；
  - per-query `reduction='none'` 的 own/global/cross；
  - 按 query 模态展开权重，加权 mean；
  - 返回 total + 三项/per-gallery CE + diag。
- [ ] 6+1 处分派全接 `'tpd'`；诊断用 `diagnose_gpd_tpd_gradients`。
- [ ] 配置主方案（设计点 + TIR 修正）：RGB/NIR `0.60/0.20/0.10`；TIR `0.80/0.10/0.05`；每 query 总 1.0。加法 control（可选）：RGB/NIR `1.0/0.20/0.10`，TIR `1.0/0.10/0.05`，总 1.40/1.20；**不写死和=1 断言**。
- [ ] 1ep smoke：V1/V2/UAD 锚点、逐 query 权重、zero-column、DDP、single/subcenter 回归。
- [ ] 独立代码审查 P0/P1=0；再 120ep。

### 7.2 实验

- 1000 AS，4 卡 DDP，batch 256，LR 0.032，120ep，seed 1234；与 V1/V2 同配方；只改 GPD。
- 测试 `transformer_120.pth`，`NECK_FEAT='before'`、`FEAT_NORM='yes'`；1000/GD/AG/detailed-domain；GD `FILTER_GD=True`。
- 第一轮只跑主方案（RGB/NIR 0.60/0.20/0.10；TIR 0.80/0.10/0.05）；加法点 control 只在预算允许时跑；不并行原 TIR 0.70 点/TIR-specific/平台 token/TIR adapter/EMA/TTA/PCGrad/硬分配。
- 单 seed screening；最终 2–3 seed；2337 等 1000 gate。

### 7.3 记录

- `WEV3/logs/TPD_1000_DDP`；`CHAT_LOG.md`；`WEV3/README_TPD.md`；`WEV3/review_v3_code.md`。
- 第一轮结论只写“package + 梯度诊断”；不写“cross 单独贡献 +X”，除非做了 own-only/global-only 控制。

---

## 8. 证据路径

**代码**

- `CC/WHU/processor/processor.py:13-30,35-71,74-88,187-227`
- `CC/WEV1/WHU/processor/processor.py:76-162,182-231,234-251,254-460,499-511`
- `CC/WEV2/WHU/processor/processor.py:165-230,250-337,360-435,438-525,596-680,710-738,756-791,809-833,888-941`
- `CC/WEV2/WHU/model/make_model.py:67-80`
- `CC/WEV2/WHU/datasets/make_dataloader.py:17-23`；`datasets/sampler.py:105-140`
- `CC/WEV2/WHU/config/defaults.py:42-64`；`configs/HPD_UAD.yml:18-28,49-71`
- `CC/WEV1/WHU/configs/SCGPD_UAD.yml:18-24,47-50`

**数据**

- V1 1000：`CC/WEV1/logs/SCGPD_1000/train_log_20261029_2209.txt:1223-1254`
- V1 GD：`CC/WEV1/logs/SCGPD_1000GD/test_log_20261030_0954.txt:173-202`
- V2 1000：`WEV2/logs/HPD_1000/train_log_20261031_0946.txt:1826-1857`
- V2 GD：`WEV2/logs/HPD_1000GD_test/test_log_20261031_1500.txt:180-209`
- V2 diag/geometry/clip：`WEV2/logs/HPD_1000/train_log_20261031_0946.txt:1741-1821`
- V2 分析：`WEV2/analysis_hpd_mechanism.md:14-21,39-51,144-182,184-208,210-237`；`WEV2/analysis_hpd_system.md:14-17,69-117,154-240`

**V3 文档**

- `CC/WEV3/V3_design.md:98-147,230-260,282-312`
- `CC/WEV3/README.md:11-39,43-79,83-89`
- `CC/WEV3/review_v3_final.md:13-24,160-235`

---

## 9. 最终裁决（§0.1 修订后）

- **TPD 结构：approve。** 三层读同一 bank、三项全局 CE、推理不变、代码基线 V2；比 HPD 的单一均值更贴近九协议，比两层 PGPD 保留了 global。
- **设计点的 RGB/NIR：approve 为 cross-first 主方案**（0.60/0.20/0.10）；它用 own 0.60 换 cross 0.20，符合“保留正面机制、攻弱项”的目标；AG/R5 风险用 §5.2 的 G 护栏判断，不再要求 V1 preservation。
- **设计点的 TIR：不 approve**（0.70/0.15/0.075）。TIR own 0.70 < V2 0.75、非 own 0.30 > V2 0.25，TIR→TIR 没有改善机制；必须改为 **0.80/0.10/0.05**（非 own 0.20 < V2 0.25，先保 TIR same，再用 0.10 cross 攻 TIR cross）。
- **第一轮主方案 = 设计点 + TIR 修正**：RGB/NIR 0.60/0.20/0.10；TIR 0.80/0.10/0.05；每 query 总 1.0。**加法点（own=1.0）降为 control/ablation**，不是第一轮主方案；它只用于回答“own 不变时 global/cross 是否有效”，不要求 V3 最终保留 V1。
- **Gate = W（弱指标/总体净改善）+ G（正面机制护栏）**：见 §5.2；不再设 A=V1 preservation。W 要求 cross/TIR/overall 超过 V2、AG/R5/GD 不低于 V2−0.5（第一轮 screening）；G 第一轮 same ≥V1−0.5、第二轮 ≥V1−0.3。若 W 不达标，说明 cross 机制不够；若 G 失败，第二轮调 own（RGB/NIR 0.60→0.65 或 TIR 0.80→0.85），单变量。
- **实现前必须关闭**：P0-1（6+1 分支）、P0-2（逐 query 权重）；P1-1/2（权重/gate）、P1-3（P_id 功能层）、P1-4（zero-column）、P1-5（TPD 诊断）、P1-6（归因/seed）、P1-7（配置/卫生）。
- **不承诺** paper+1；第一轮是“继承正面 + 弱指标净改善”的可证伪测试；AG 若必须改善，第二轮加平台机制，不在 TPD 权重里硬调。
