# WEV3 方案设计：三层原型判别（TPD, Three-level Prototype Discrimination）

> 目标：在 UAD/ViT 单模型上，继承 UAD/WEV1/WEV2 的正面部分，改善九种模态协议、TIR、AG、R5/R10、GD R5 等薄弱指标；推理不变，不做 CLIP/融合，不用 batch-local 排序损失。
> 代码基线：`WEV2/WHU`（HPD，已含 P_id/P_scene 两级原型与 warm start）；对照：`CC/WEV1/WHU`（GPD-SC）、`CC/WHU`（UAD）。
> 论文：`CC/WEV1/WHU/tmp/pdfs/WHU.txt`（Sec.4.2–4.5、Eq.3–10、Table 3/4/6/8）。
> 标记：[代码事实] / [实验支持] / [机制假设]
>
> **最终审查修正（2026-10-31，`review_v3_tpd_final.md §0.1/§0.2/§5.2` + `review_v3_tpd_agent_final.md` 权威）**：V3 不是原样保留 V1/V2，而是**保留正面机制 + 弱指标/总体净改善**；允许取舍。第一轮主方案 = **设计点 cross-first + TIR 修正**：RGB/NIR `0.60/0.20/0.10`；TIR 改为 `0.80/0.10/0.05`（原 0.70/0.15/0.075 不 approve：own < V2 0.75、非 own > V2 0.25，TIR→TIR 无改善机制）；每 query 总 1.0。**加法点 own=1.0 降为 control/ablation**，不是第一轮主方案。Gate 改为 **W（弱指标/总体净改善）+ G（第一轮 screening，严格版第二轮）**，不再设 V1 preservation；AG 无平台项，只做护栏。
>
> **⚠ 正文历史残留提示**：关键权重（§4.2/§7/§9.4/§10/§11/§13）和 6+1 分派已按主方案更新；但 §8.2 仍是旧 Gate A/B/C，§9.2 仍保留标量 CE×length-3 的旧伪代码示例。**实现/配置以 README、`review_v3_tpd_final.md §0.1/§0.2/§5.2`、`review_v3_tpd_agent_final.md` 为准**；§9.2 的正确逐 query 聚合见 review §2.2/§0.2；§8.2 的 W+G 见 review §5.2。

---

## 0. 一页结论

**V3/TPD 不是“V1 + gallery cross”的两层方案，而是“UAD 全局锚点 + WEV1 模态子中心 + WEV2/PGPD gallery 专属 cross”的三层方案，并用模态感知权重保护 TIR、用全局项保住 GD R5/cross 平均。**

- **保留的正面部分**：
  - UAD：单共享身份原型 + 全局负类（跨模态全局锚点、GD R5、整体 cross）；
  - WEV1：模态子中心 + `max_k` + ProCA 模态中心更新 + count EMA + warm start（同模态、AG、R5/R10）；
  - WEV2：`P_id` 全局项 + `P_scene` 场景项 + 两级 bank/warm start；以及 overall mAP/GD 的增量。
- **新增的正面部分**：
  - WEV2/PGPD 的 **gallery 专属 cross**（对另外两个 gallery 模态的 `P_scene[:,m']` 做全局 CE），直接对应九协议；
  - **模态感知权重**：TIR own 更高、cross 更低；RGB/NIR cross 更高，保护 TIR→TIR、提升 RGB/NIR cross。
- **去掉的薄弱部分**：
  - 不再让 `P_id` 成为唯一的跨模态桥（V2 的 mean 0.25 太温和、TIR 被拉走）；
  - 不再让 scene 从 100% 一次性降到 75% 而没有 TIR 保护（V2 的 TIR −1.22、AG −0.37/−0.52、R5 −0.66）；
  - 不做“硬 own”路由；own 仍是 WEV1 的 `max_k`。
- **指标目标（双基线）**：超过 WEV2 的 overall/cross/GD；守住 WEV1 的 same/AG/R5/R10/GD R5；TIR 单独设护栏；Paper+1 作为 Tier2，不在第一轮承诺。

---

## 1. 证据：哪些是正面、哪些是薄弱

### 1.1 UAD（单共享原型）

- 1000：11.02/29.71/46.32/54.55；same 30.40/18.96/3.25；cross 13.48/8.95/1.58；AG R1 31.63/30.12；GD 14.40/31.61/**46.14**/53.56；PCA 0.201；GPD 0.082。[实验支持]
- **正面**：单共享原型给所有模态一个共同锚点；全局负类；GD R5 是三者最高（46.14），说明共享锚点对 ground/daytime 的 R5 有好处。
- **薄弱**：单原型把模态多峰平均掉；same/AG/R5/R10 低于 V1/V2；cross 也低于 V2（V2 13.72/9.16 vs UAD 13.48/8.95）。
- **结论**：UAD 的正面是“全局锚点 + 全局负类 + GD R5”，不是它的具体单原型几何；V3 应保留一个**全局项**，但不能让它成为唯一跨模态力。

### 1.2 WEV1（GPD-SC，模态子中心）

- 1000：11.37/30.51/48.17/56.27；same 31.16/19.97/3.51；cross 13.33/9.00/1.60；AG R1 35.11/32.66；GD 14.84/31.10/44.61/53.76；PCA 0.212；GPD 0.065。[实验支持]
- **正面**：模态子中心 + `max_k` 给同模态/AG/R5/R10；`own_ratio`（V2 diag）0.92–1.0；warm start/count EMA 稳定；推理不变。
- **薄弱**：cross 平；没有全局锚点；GD R5 低；PCA 高。
- **结论**：V3 必须保留 `P_scene`/`max_k`/warm start/count EMA，作为 same/AG/R5 的锚。

### 1.3 WEV2（HPD，P_id 均值 + P_scene）

- 1000：11.47/30.52/47.51/56.14；same 30.97/19.96/3.49；cross 13.72/9.16/1.61；AG R1 34.74/32.14；GD 14.91/31.61/44.82/54.17；PCA 0.205；GPD 0.069。[实验支持]
- `P_id=mean(P_scene)` cos 0.999；shared 有效份额 10.6%、scene 23.3%、base 65.8%；clip 1.73/100；TIR `own_ratio=1.0`。[实验支持]
- **正面**：整体 mAP 最高；cross 最高（13.72/9.16）；GD mAP/R1/R10 最高；PCA 比 V1 低；两级原型不塌缩；overall 比 V1 +0.10。
- **薄弱**：TIR→TIR −1.22；AG R1 −0.37/−0.52；R5 −0.66、R10 −0.13；GD R5 44.82（<46.0）；`P_id` 是均值，太温和；shared 10.6%；TIR 被均值拉走。
- **结论**：V2 的正面是“全局项 + 场景项 + 跨模态方向 + overall/GD 增量”；V2 的薄弱是“全局项太弱/太均值、scene 一次降太多、无 TIR 保护”。V3 应**保留全局项和场景项，加入 gallery cross + 模态感知权重**，而不是把全局项删掉。

### 1.4 为什么不是“V1 + gallery cross”的两层

- 两层方案（own 0.6 + cross 0.4）删掉了 UAD/V2 的全局项，也把 scene 从 0.75 降到 0.6；它在 final review 中被指出 TIR/AG/R5 风险被低估（P1-1），且“own max_k 保护 TIR”不成立（TIR own_ratio=1.0）。
- 两层方案的 cross 0.4 是名义值，有效份额未知；没有全局项，GD R5/cross 平均可能回退。
- **结论**：两层方案保留作**消融**（`w_global=0`），不是 V3 主方案。

---

## 2. 设计原则

1. **保留正面、替换薄弱**：全局项（UAD）、场景项（V1）、gallery cross（V2/PGPD）三者都保留；去掉“均值唯一桥”“scene 一次降太多”“硬 own”这些薄弱选择。
2. **一个统一机制，不是模块堆叠**：三项都是全局原型 CE，共享 pre-BN CLS、共享 `P_scene`/`P_id` bank、共享 warm start、共享推理；没有新分支、没有新特征、没有 batch-local 排序、没有后处理。
3. **模态感知**：TIR 是离群/弱模态，给 TIR 更高的 own、更低的 global/cross；RGB/NIR 给更高的 cross。权重按 query 模态，不按样本学习，稳定可解释。
4. **推理不变**：pre-BN CLS + L2 + 欧氏；不加载 bank；不改 768D。
5. **可归因**：own/global/cross 三项分开记录；per-gallery CE；梯度份额/cos；TIR/AG/R5/GD 分开看；gate 分层。
6. **一次一个主方案**：第一轮只跑 TPD 一个权重点；两层（global=0）、V2（gallery=0）、V1（own=1）都是消融锚点。

---

## 3. 架构与数据流

```
imgs [RGB B; NIR B; TIR B]
  -> 共享 ViT-B/16 -> pre-BN CLS f [3B,768]
     ├─ BNNeck -> classifier -> CE（只服务身份分类）
     ├─ WRT（pre-BN f）
     ├─ ProCA（centers[m] -> g_cent, detach, 0.01）
     └─ TPD:
          u = L2(f)
          L_own    = CE(max_k cos(u, P_scene[c,k]) / τ)          # V1 scene
          L_global = CE(cos(u, P_id[c]) / τ)                     # UAD global
          L_cross  = CE(cos(u, P_scene[c,m']) / τ), m'≠m         # gallery cross
          L_GPD = w_own(m) L_own + w_global(m) L_global
                  + w_cross(m) Σ_{m'≠m} L_cross^{m'}
检索：pre-BN CLS -> L2 -> 欧氏；不加载 bank
```

- `P_id[C,D]`：UAD 的全局身份原型；`g_cent` EMA 更新。
- `P_scene[C,3,D]`：V1 的模态子中心；ProCA 模态中心 + count EMA 更新。
- `u`：`f` 的 L2；三项 CE 都作用于 `u`；`P_id`/`P_scene` 是 buffer，不接收梯度。
- query 模态 m 由 batch 分段决定（[RGB B; NIR B; TIR B]）；gallery 模态 m' 循环；`target_rep=target.repeat(3)` 与分段对齐。
- 负类：每个 CE 都是全部 C 个身份；全局覆盖，不是 batch-local。

---

## 4. 损失与权重

### 4.1 三项

$$
\mathcal L_{own}=\mathrm{CE}\left(\max_k\frac{u^\top P_{scene}[c,k]}{\tau},y\right)
$$

$$
\mathcal L_{global}=\mathrm{CE}\left(\frac{u^\top P_{id}[c]}{\tau},y\right)
$$

$$
\mathcal L_{cross}^{m'}=\mathrm{CE}\left(\frac{u^\top P_{scene}[c,m']}{\tau},y\right),\quad m'\ne m
$$

$$
\mathcal L_{GPD}=w_{own}(m)\mathcal L_{own}+w_{global}(m)\mathcal L_{global}
+w_{cross}(m)\sum_{m'\ne m}\mathcal L_{cross}^{m'}
$$

- `τ=0.03`；三项都 float32；`loss_gpd` 单标量；总损失 `CE+WRT+0.01·ProCA+1.0·L_GPD`。
- `P_id`/`P_scene` 都 L2；`u` L2；CE 在 cosine/τ 上。

### 4.2 模态感知权重（第一轮主方案）

| query 模态 | w_own | w_global | w_cross（每个 m'） | 合计 |
|---|---:|---:|---:|---:|
| RGB | 0.60 | 0.20 | 0.10 | 0.60+0.20+0.10×2 = 1.00 |
| NIR | 0.60 | 0.20 | 0.10 | 1.00 |
| TIR | **0.80** | **0.10** | **0.05** | 0.80+0.10+0.05×2 = 1.00 |

- **为什么 TIR own 更高**：TIR 表征弱、离均值远（0.919）、TIR→TIR 是 V2 的最大回退（−1.22）；TIR 需要更高的 own、较低的 global/cross。
- **为什么 RGB/NIR cross 更高**：RGB/NIR 近（pair cos 0.914），gallery cross 更有效；V2 的 cross 增益集中在 NIR↔RGB（+1.46/+0.61 R1）；V3 给 RGB/NIR 0.10×2 的 cross。
- **global 0.20/0.10**：保留 UAD 的全局锚点；RGB/NIR 0.20、TIR 0.10；V2 的 global 0.25；V3 的 global 更低，但 gallery cross 补上。
- **own 0.60/0.80**：V2 的 scene 0.75；RGB/NIR 0.60 是 cross 的代价；TIR 0.80 高于 V2，先用 own 保住 TIR→TIR，再用 0.10 cross 攻 TIR cross。

### 4.3 权重与消融锚点

| 方案 | w_own | w_global | w_cross | 说明 |
|---|---:|---:|---:|---|
| V1 | 1.00 | 0 | 0 | GPD-SC |
| UAD | 0 | 1.00 | 0 | 单原型 |
| V2/HPD | 0.75 | 0.25 | 0 | mean global + scene |
| 两层 PGPD | 0.60 | 0 | 0.20×2 | 删掉 global 的消融 |
| **V3/TPD** | **0.60/0.80** | **0.20/0.10** | **0.10/0.05** | **主方案** |

- `w_cross=0` 时 V3/TPD 不是精确 V2（V2 的 own 0.75/global 0.25）；V2 锚点要单独用 V2 的 120ep 权重/日志。
- `w_global=0` 时是两层 PGPD；`w_own=1` 时是 V1；`w_global=1` 时是 UAD。
- 第一轮只跑 V3/TPD 的模态感知权重点；不做多臂。

---

## 5. 初始化、更新、DDP、AMP

### 5.1 P_id（UAD 全局锚点）

- `[C,D]`；warm start 用 valid-modality identity mean（V2 的 `id_mean`）；零行 fallback `g_cent`。
- 更新：`PrototypeBank.update`，`g_cent` EMA 0.2；无 count EMA（匹配 UAD/V2）。
- DDP：`ids`/`g_cent` all-gather；bank 不挂 DDP model。

### 5.2 P_scene（V1 模态子中心）

- `[C,3,D]`；warm start 每 `(pid,modality)` 类均值；缺失模态回退 `id_mean`；counts=1。
- 更新：`SubCenterPrototypeBank.update`，ProCA 模态中心 + count EMA `clamp(1/(count+1),0.1,0.5)`；DDP `ids/mod_ids/centers` all-gather。
- 零行 fallback `g_cent` 复制 K 份。

### 5.3 联合 warm start

- 一次三模态 pass（V2 的 `warm_start_hierarchical`）：P_scene=modality mean；P_id=valid id_mean；缺失回退；counts_scene=1；rank0+barrier+broadcast（P_id、P_scene、counts）；RNG 保存/恢复；finite 检查。
- 不额外遍历训练集。

### 5.4 训练/优化

- 120ep 从头 DDP，4 卡，batch 256，LR 0.032，SGD，cosine+5 warmup，seed 1234；与 V1/V2 同配方；只改 GPD。
- 全局 `clip_grad_norm_(1.0)`；HPD 的 pre-clip 1.73/100；V3 的三项可能更大；记录 pre-clip/clip，不改 max_norm。
- 不开 EMA/TTA/CA-WRT/LLRD/margin；不加平台 token/TIR adapter/PCGrad/硬分配。
- 1ep smoke 后 120ep；单 seed screening；最终 2–3 seed；2337 等 1000 gate 之后。

---

## 6. 梯度与监督

### 6.1 梯度流

```
L_own    -> f -> ViT
L_global -> f -> ViT
L_cross  -> f -> ViT
ProCA    -> centers -> f -> ViT (g_cent detached)
CE       -> BNNeck/classifier -> f -> ViT
WRT      -> f -> ViT
P_id/P_scene are buffers, no grad
optimizer.step()
P_id <- EMA(P_id, g_cent)
P_scene[c,m] <- EMA(P_scene[c,m], centers[m][c])
```

- 三项 CE 都只回传到 `f`/ViT；bank 无梯度；`g_cent` detach。
- 三项共享 `u`；own/global/cross 的梯度可能在 TIR 上冲突（own=TIR、global=mean、cross=RGB/NIR），权重是平衡阀；必须记录 per-term grad norm/cos。
- base CE/WRT 仍占大头（V2 代理 65.8%）；TPD 是次级正则；global/cross 的有效份额需实测，不能用名义权重推断。

### 6.2 监督是否足够

- 三项都是全局 CE（全部 C 个身份）；覆盖同模态（own）、全局（global）、跨 gallery（cross）三类监督；没有 batch-local；没有“监督对象 ≠ 推理特征”。
- `P_id`/`P_scene` 的更新来自 ProCA 中心；bank 的模态语义由更新规则保证；不需要一致性 loss。
- **冲突**：TIR 的 own vs global/cross（pair cos 0.809）；RGB/NIR 的 own/global/cross 更一致（pair cos 0.914）；TIR 的权重 0.80/0.10/0.05 是保护；仍需 per-term cos 诊断。

### 6.3 指标对齐

- own/global/cross 都是分类目标，不是 AP/CMC；mAP/R1/R5 是涌现。
- own 保留 same/AG/R5/R10；global 保留 UAD 的 GD R5/cross 平均；cross 改善九协议/TIR；但 R5/AG 仍可能回退；gate 正面回答。
- 不承诺 paper+1；第一轮是相对 V1/V2 的可证伪测试。

---

## 7. 为什么 TPD 应改善九协议

| 协议族 | own | global | cross | 预期 |
|---|---|---|---|---|
| RGB→RGB / NIR→NIR | 高（0.60） | 0.20 | 0.10+0.10 | 同模态守住；R5/R10 由 own 保 |
| RGB→NIR / NIR→RGB | 0.60 | 0.20 | 0.10（对对方 gallery） | cross 直接；V2 的强项继续 |
| TIR→TIR | **0.80** | 0.10 | 0.05+0.05 | TIR own 高于 V2 0.75；先保 TIR，避免 V2 −1.22 |
| TIR→RGB / RGB→TIR | 0.80/0.60 | 0.10/0.20 | 0.05/0.10 | TIR cross 由 gallery 直接拉；TIR 表征弱仍是上限 |
| AG platform_only | own 0.60/0.80 | global 0.20/0.10 | cross 0.10/0.05 | global 保留 UAD 的 GD/cross；platform 无显式项；R1 是风险 |
| GD R5 | own | global 0.20/0.10 | cross | global 是 UAD 的 GD R5 来源；比两层更可能守住 |

- **关键**：V3/TPD 保留 UAD 的 global（GD R5/cross 平均）、V1 的 own（same/AG/R5）、V2 的 cross（九协议）；TIR 用更高 own 保护；不是“去掉 global 的两层”。
- **代价**：RGB/NIR own 0.60 < V2 0.75；R5/AG/R10 有回退风险；TIR own 0.80 > 0.75，TIR same 有改善机制但 TIR cross 仍弱；cross 名义 0.20/0.10 不一定等于有效份额。

---

## 8. 诊断、gate、实验

### 8.1 诊断（训练 forward 之前，θ_t/B_t）

- per-term CE：`L_own`、`L_global`、`L_cross^{m'}`（两个）；per-modality。
- 梯度：`||g_own||`、`||g_global||`、`||g_cross^{m'}||`、两两 cos、有效份额（乘权重后）；pre-clip/clip。
- 原型：`P_id`/`P_scene` 的 pair cos、per-column norm、`g_cent_norm`；own_ratio、usage。
- 指标：per-pair R1/mAP（9 对）、TIR 四向、AG A→G/G→A、GD 四项、same/cross 平均。
- 标注 last-block/eval/rank0 代理；不要用 diag loss 代替训练 loss。

### 8.2 Gate（**以下 A/B/C 为旧版，已废弃**；权威 W+G 见 review_v3_tpd_final §5.2 与 README）

**A. V1 preservation（hard）**

- same 3 对 R1/mAP ≥ V1−0.2（≥30.96/19.77）；
- AG 两向 R1 ≥ V1−0.3（≥34.81/32.36）；
- overall R1 ≥30.5；
- GD 四项 ≥ V1−0.2（GD R5 ≥44.41）；
- **TIR→TIR R1/mAP ≥ V1−0.5（≥21.84/14.20）**；若回退 >0.5，第二轮先 TIR-specific own，不再加 cross。

**B. Beat V2（Tier1）**

- overall mAP >11.47；
- cross R1/mAP >13.72/9.16；
- GD ≥14.91/31.61/44.82/54.17；
- R5≥47.3、R10≥55.8（paper+1 guardrail，不是 V1−0.2）。

**C. Paper+1（Tier2，第一轮不承诺）**

- 1000 12.0/30.5/47.3/55.8；
- GD 15.5/32.9/46.0/53.5；GD R5≥46.0 属 Tier2/护栏，不是 V1 preservation。

> `Δoverall mAP≈0.90×Δmean₉`（V1/V2 直接比 0.898；历史回归 0.908）：same 保持时 **cross 平均** +0.3 → mean₉ +0.2 → overall≈+0.18；要 overall+0.4 需 **cross 平均** +0.66，或 same 上涨。不能同时写 `cross+0.3 & overall+0.4`。

### 8.3 实验顺序

1. `WEV3/WHU` 从 `WEV2/WHU` 复制；保留 P_id/P_scene bank、warm start、diagnostics；把 `GPD_HPD` 扩展为 `GPD_TPD`（三项 CE + 模态感知权重）。
2. 6 处 `gpd_type` 分派全部接 `'tpd'`（P0-1）；加 `GPD_TYPE != 'single' and 'gpd' not in loss_type` 守卫；加 `GPD_DIAG_PERIOD`。
3. 1ep smoke：`w_cross=0` 是 V2-rebalanced（own 0.6/global 0.2），不是精确 V2；`w_global=0` 是两层 PGPD；`w_own=1` 是 V1；`w_global=1` 是 UAD；三项 CE 有限；bank/warm start/DDP 正常；`single`/`subcenter` 回归。
4. 120ep：第一轮只跑 TPD 模态感知权重；单 seed screening；最终 2–3 seed；2337 等 1000 gate。
5. 消融（第二轮，按需）：V2 锚点、两层 PGPD、own 0.65/global 0.2/cross 0.075、TIR own 0.8、global 0.25；一次一个变量。

---

## 9. 代码修改方案（WEV3/WHU）

### 9.1 基线

- 从 `WEV2/WHU` 复制（已有 P_id/P_scene、`HierarchicalPrototypeBank`、`warm_start_hierarchical`、`diagnose_gpd_gradients`、6+1 处分派）。
- 不修改 `model/`、`loss/`、`datasets/`、`utils/`、`solver/`。
- 修改 `processor/processor.py`、`config/defaults.py`；新增 `configs/TPD_UAD.yml`；新增 `README_TPD.md`。

### 9.2 核心函数

```python
def GPD_TPD(feat, target, bank, ids, g_cent, num_modalities,
            tau=0.03, w_own=(0.6,0.6,0.8), w_global=(0.2,0.2,0.1),
            w_cross=(0.1,0.1,0.05)):
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat.float(), dim=1)
    # P_id / P_scene clone + zero fallback (same as GPD_HPD/GPD_SC)
    logits_id = (f @ P_id.t()) / tau                       # [N,C]
    logits_scene = einsum('nd,ckd->nck', f, P_scene) / tau # [N,C,K]
    loss_global = CE(logits_id, target_rep)
    loss_own = CE(logits_scene.max(dim=2).values, target_rep)
    B = target.shape[0]
    loss_cross = []
    for m in range(num_modalities):
        q = logits_scene[m*B:(m+1)*B]                      # [B,C,K]
        t = target_rep[m*B:(m+1)*B]                        # [B]
        for m2 in range(num_modalities):
            if m2 == m: continue
            loss_cross.append(CE(q[:,:,m2], t))
    # modality-aware weights: per query modality
    # w_own(m), w_global(m), w_cross(m)
    loss = (w_own * loss_own + w_global * loss_global
            + w_cross * 2 * mean(loss_cross per query))
    return loss, loss_own, loss_global, per_gallery_losses, diag
```

- `w_own`/`w_global`/`w_cross` 是 length-3 的 per-query-modality 权重；第一轮 `(0.6,0.6,0.8)/(0.2,0.2,0.1)/(0.1,0.1,0.05)`；每个 query 的 own+global+2×cross=1.0。**注意：上面的标量 CE 伪代码不能实现逐 query 权重；正确聚合见 review §2.2/§0.2（reduction='none' 或 per-modality /M）。**
- 负类全部 C；`P_id`/`P_scene` buffer；`target.repeat(M)` modality-major；`q[:,:,m2]` 是 gallery 模态 m2。
- `w_cross=0` = V2-rebalanced（own/global 0.6/0.2 或 0.7/0.15）；`w_global=0` = 两层 PGPD；`w_own=1` = V1；`w_global=1` = UAD。

### 9.3 6+1 处分派（P0-1）

V2 的 6 处 `gpd_type` 分派 + 1 处诊断（`WEV2/WHU/processor/processor.py:597-600, 608-646, 647-680, 710-738, 756-791, 809-833, 888-941`）全部接 `'tpd'`；诊断必须 `diagnose_gpd_tpd_gradients`，不能调 `GPD_HPD`：

- bank 创建：`HierarchicalPrototypeBank`（P_id+P_scene）；
- warm start：`warm_start_hierarchical`；
- loss：`GPD_TPD`；
- update：`P_id` ids/g_cent + `P_scene` ids.repeat(M)/mod_ids/centers_all；
- 日志：P_id/P_scene counts、三项 CE、per-gallery、TIR/AG；
- 守卫：`GPD_TYPE != 'single' and 'gpd' not in loss_type`；`K==M`；`NECK_FEAT='before'`。

### 9.4 配置

```yaml
MODEL:
  GPD_TYPE: 'tpd'
  GPD_SUB_CENTERS: 3
  GPD_TPD_OWN_WEIGHT: [0.6, 0.6, 0.8]
  GPD_TPD_GLOBAL_WEIGHT: [0.2, 0.2, 0.1]
  GPD_TPD_CROSS_WEIGHT: [0.1, 0.1, 0.05]
  GPD_TAU: 0.03
  GPD_COUNT_EMA: True
  GPD_EMA_MIN: 0.1
  GPD_EMA_MAX: 0.5
  GPD_WARMSTART: True
  GPD_SHARED_MOMENTUM: 0.2
  GPD_DIAG_PERIOD: 100
SOLVER:
  LOSS_TYPE: 'base+pca+gpd'
  MAX_EPOCHS: 120
  IMS_PER_BATCH: 256
  BASE_LR: 0.032
TEST:
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
OUTPUT_DIR: './logs/TPD_1000_DDP'
```

### 9.5 测试

- `compileall`；配置解析；`GPD_TPD` 的 `w_cross=0`/`w_global=0`/`w_own=1`/`w_global=1` 数值锚点（同 bank 状态）；三项 CE 有限；bank 无梯度；warm start/zero fallback/missing modality；fake DDP all-gather；`single`/`subcenter` 回归逐元素等于 V1。
- 1ep smoke：`GPD_TYPE='tpd'` 真的用 `HierarchicalPrototypeBank`/`GPD_TPD`；loss/梯度有限；bank counts；own_ratio/usage；pre-clip/clip；DDP 不挂；无 NaN。
- 120ep 前独立代码审查 P0/P1=0。

---

## 10. 预期、风险、回退

### 10.1 预期

- **cross**：gallery cross 0.10/0.05 + global 0.20/0.10；比 V2 的 mean 0.25 更具体；预期 RGB/NIR cross 不低 V2，TIR cross 有改善空间但幅度受 TIR 表征限制；
- **same/AG/R5**：RGB/NIR own 0.60 < V2 0.75，same/AG/R5 有回退风险（用第一轮 screening）；TIR own 0.80 > V2 0.75，TIR same 有改善机制；global 0.20/0.10 帮助 GD/cross 平均；
- **GD R5**：global 0.20/0.10 < V2 0.25；第一轮只要求不低于 V2−0.5（screening），严格 V2−0.2/46.0 是第二轮/Tier2；
- **TIR**：TIR own 0.80；预期 TIR→TIR ≥ V2，向 V1 靠近；TIR cross 0.10，改善幅度有限；
- **overall**：预期 >V2 11.47 是第一轮的关键；paper+1 是 Tier2。

### 10.2 风险

- TIR：own 0.80 高于 V2，但 cross 0.10 的两列更远；TIR cross 可能不足；TIR→TIR gate；
- AG：无平台项；platform_only R1 风险；AG 只做 screening 护栏；
- R5/R10：RGB/NIR own 0.60 < V2 0.75；R5/R10 有风险；
- GD R5：global 0.20/0.10 < V2 0.25；GD R5 有风险；
- cross 硬负样本：gallery 原型比均值更硬；per-pair R1/最难负样本；
- 有效份额：名义 0.2/0.1 不等于有效；诊断；
- 单 seed：2–3 seed；
- 代码：6+1 处分派（P0-1）；诊断时序（P1-4）。

### 10.3 回退

- TIR same 回退：TIR own 0.80→0.85、global 0.10→0.05、cross 0.05→0.05（总 1.0，先保 own）；
- same/AG/R5 回退：RGB/NIR own 0.60→0.65、cross 0.10→0.075（总 1.0）；TIR 保持 0.80/0.10/0.05；
- cross 不够：RGB/NIR cross 0.10→0.125、own 0.60→0.55（总 1.0）；TIR cross 0.05→0.075、own 0.80→0.75（总 1.0）；
- GD R5 不够：global 0.20/0.10→0.25/0.15，cross 相应减 0.05（总 1.0）；
- 一次一个变量；不同时上平台 token/TIR adapter/EMA/TTA/PCGrad/硬分配。

---

## 11. 与 V1/V2/UAD 的关系

| | UAD | V1 | V2 | **V3/TPD** |
|---|---|---|---|---|
| 全局锚点 | 1.0 | 0 | 0.25 | 0.20/0.10 |
| 场景项 | 0 | 1.0 | 0.75 | 0.60/0.80 |
| gallery cross | 0 | 0 | 0 | 0.10/0.05×2 |
| 同模态 | 弱 | 强 | 中强 | 中强（own 0.60/0.80） |
| 跨模态 | 中 | 弱 | 中 | 中强（global+cross） |
| GD R5 | 46.14 | 44.61 | 44.82 | 目标 ≥ V2，向 46.0 |
| 代码基线 | — | — | P_id/P_scene | **V2** + `GPD_TPD` |
| 指标目标 | — | preservation | overall/cross | **>V2 + 守 V1** |

- V3/TPD 保留 UAD 的 global、V1 的 scene、V2 的 bank/warm start；新增 gallery cross 和模态感知权重。
- 两层 PGPD（global=0）是消融；V2（cross=0）是锚点；V1（own=1）是锚点；UAD（global=1）是锚点。

---

## 12. 验收与记录

- 第一轮：1ep smoke → 120ep → 1000/GD/AG/detailed-domain；单 seed screening；2–3 seed 最终；2337 等 1000 gate。
- Gate：A（V1 preservation）、B（Beat V2）、C（Paper+1/Tier2）、TIR guardrail；按 §8.2。
- 记录：`CHAT_LOG.md`；`WEV3/V3_design.md`；实现后 `WEV3/README_TPD.md`、`WEV3/review_v3_code.md`；日志 `logs/TPD_1000_DDP`。
- 不承诺 paper+1；第一轮是“>V2 + 守 V1”的可证伪测试；如果 TIR/AG/R5 回退，按 §10.3 单点回退，不继续加模块。

---

## 13. 一句话

**V3/TPD = UAD 的全局锚点（0.20/0.10） + V1 的模态子中心（0.60/0.80） + V2/PGPD 的 gallery cross（0.10/0.05×2）；模态感知权重保护 TIR，全局项保住 GD R5/cross 平均；保留正面、替换薄弱，推理不变，一次一个主方案。**
