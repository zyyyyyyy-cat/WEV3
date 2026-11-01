# WEV3 / TPD：方案索引与一页说明

> 状态：**代码第一版完成**（`WEV3/WHU`）；独立代码审查 **P0=0 / P1=0**（`review_v3_tpd_code.md`）；下一步 1ep smoke → 120ep。
> **V3 主方案为 TPD（三层原型判别）**；详细设计见 `V3_design.md`；**最终审查见 `review_v3_tpd_final.md`（权威）**。
> **最终审查修正（用户目标澄清后）**：V3 不是原样保留 V1/V2，而是**保留起正面作用的机制 + 改善弱指标**；允许取舍。第一轮主方案 = **设计点 cross-first + TIR 修正**：RGB/NIR `0.60/0.20/0.10`；TIR `0.80/0.10/0.05`；每 query 总 1.0。设计点原 TIR `0.70/0.15/0.075` 不 approve（own 0.70 < V2 0.75、非 own 0.30 > V2 0.25，TIR→TIR 无改善机制）。**加法点 own=1.0 降为 control/ablation**，不是第一轮主方案。Gate 改为 **W（弱指标/总体净改善）+ G（正面机制护栏）**，不再设 V1 preservation。详见 `review_v3_tpd_final.md §0.1/§5.2`。
> 之前的“两层 PGPD（V1 + gallery cross）”降为 **消融**（`w_global=0`）；`review_v3_final.md` 的 P0/P1 条件仍适用于 V3/TPD。
> 代码基线：`WEV2/WHU`（HPD，含 P_id/P_scene 两级原型、warm start、诊断）；对照：`CC/WEV1/WHU`（GPD-SC）、`CC/WHU`（UAD）。
> 论文：`CC/WEV1/WHU/tmp/pdfs/WHU.txt`（Table 3/4/6、Sec.4.2–4.5、Eq.4–10、Table 8）。
>
> **已敲定（2026-10-31）**：第一轮 **screening**、第二轮 **strict**（决策树见 Gate）；AG 第一轮只做护栏，平台机制第二轮；最终目标仍是 paper+1 全指标，第一轮是 Pareto/机制测试，不是最终声明；加法 control 第一轮不跑；2337 等 1000 W/G 之后。

---

## 一页方案

**V3/TPD = UAD 的全局锚点 + WEV1 的模态子中心 + V2/PGPD 的 gallery 专属 cross，用模态感知权重保护 TIR。**

$$
\mathcal L_{GPD}=w_{own}(m)\mathcal L_{own}
+w_{global}(m)\mathcal L_{global}
+w_{cross}(m)\sum_{m'\ne m}\mathcal L_{cross}^{m'}
$$

- `L_own`：WEV1 `GPD_SC` 的 `max_k` CE（场景项，保 same/AG/R5）；
- `L_global`：UAD 的 `P_id` 全局 CE（全局锚点，保 GD R5/cross 平均）；
- `L_cross^{m'}`：对另外两个 gallery 模态 `P_scene[:,m']` 的全局 CE（九协议/TIR）；
- 权重（**第一轮主方案：设计点 cross-first + TIR 修正**）：

| query 模态 | w_own | w_global | w_cross（每个 m'） | 合计 |
|---|---:|---:|---:|---:|
| RGB | **0.60** | 0.20 | 0.10 | 1.00 |
| NIR | **0.60** | 0.20 | 0.10 | 1.00 |
| TIR | **0.80** | 0.10 | 0.05 | 1.00 |

- **RGB/NIR 保留设计点的 cross-first**：own 0.60 换 cross 0.20，攻 cross/九协议；AG/R5 风险用 G 护栏判断。
- **TIR 修正**：own 0.80 > V2 0.75，非 own 0.20 < V2 0.25；先保 TIR→TIR（V1 的正面），再用 0.10 cross 攻 TIR cross。设计点原 TIR 0.70/0.15/0.075 会让 TIR same 更差，不采用。
- **加法点（control/ablation，不是第一轮主方案）**：own=1.0；RGB/NIR global 0.20 + cross 0.10×2（总 1.40）；TIR global 0.10 + cross 0.05×2（总 1.20）。它是主方案的 own 消融（只差 own），只用于回答“own 不变时 global/cross 是否有效”；预算只够一个 120ep 时不跑。
- 更新：`P_id` 用 `g_cent` EMA 0.2；`P_scene` 用 ProCA 模态中心 + count EMA 0.1–0.5；联合 warm start；DDP 两 bank；推理不变（pre-BN CLS + L2 + 欧氏）。
- **不是**“V1 + gallery cross”的两层；两层是 `w_global=0` 的消融；V2 是 `w_cross=0, own=0.75, global=0.25` 的精确锚点；V1 是 `w_own=1`；UAD 是 `w_global=1` **且 `GPD_WARMSTART=False`**。

**为什么这样设计**：

- **目标不是 V1/V2 原样保留**，而是保留起正面作用的机制，并改善弱指标（cross/九协议/TIR/AG/R5/GD R5）；允许为弱指标做取舍。
- UAD 的正面：全局锚点 + 全局负类 + GD R5（46.14）；V1 的正面：模态子中心/max_k/warm start + same/AG/R5；V2 的正面：global+scene 两级、overall mAP 11.47、cross 13.72/9.16、GD 14.91；TPD 新增 gallery cross → 九协议。
- **第一轮主方案用设计点的 RGB/NIR cross-first**（0.60/0.20/0.10）：用 own 0.60 换 cross 0.20，直接攻 cross/九协议；AG/R5 的代价由 G 护栏判断，不要求 V1 不降。
- **TIR 必须修正**：设计点 TIR own 0.70 < V2 0.75、非 own 0.30 > V2 0.25，TIR→TIR 只会更差；改成 0.80/0.10/0.05，先继承 V1 的 TIR 正面，再用 0.10 cross 攻 TIR cross。
- **AG 没有平台项**：TPD 权重无法改善 AG；AG 只做 G 护栏（第一轮 screening ≥V2−0.5）。若 AG 是必须改善项，第二轮加平台机制，不在 TPD 权重里硬调。
- **R5 靠 own，GD R5 靠 global**：第一轮只要求不低于 V2−0.5（screening），严格 V2−0.2 第二轮；paper+1 仍 Tier2。加法点（own=1.0）是 control，不是主方案。

---

## Gate（W + G，不是 V1 preservation）

### W. 弱指标/总体（hard targets）

- overall mAP/R1 > V2 `11.47/30.52`；
- cross R1/mAP > V2 `13.72/9.16`；
- TIR→TIR R1/mAP ≥ V2 `21.12/14.55`，target V1 `22.34/14.70`；TIR→RGB `7.59/5.98`、RGB→TIR `7.26/5.71`、TIR→NIR `8.06/5.39`、**NIR→TIR `7.03/5.29`** 不低于 V2；
- AG/R5/R10/GD 不在 W 里做硬 gate，见 G 的 screening（第一轮）/严格（第二轮）；
- paper+1 是 Tier2，不是第一轮承诺。

### G. 正面机制护栏（第一轮 screening；严格版第二轮）

- **第一轮 screening**（agent 复审建议，避免用权重无法满足的护栏判死）：same ≥ V1−0.5（≥30.66/19.47）；AG/R5/R10/GD ≥ V2−0.5（AG ≥34.24/31.64；R5 ≥47.01；R10 ≥55.64；GD R5 ≥44.32；GD mAP/R1/R10 ≥14.41/31.11/53.67）。
- **第二轮严格版**（W 达标、G 在 screening 内）：same ≥ V1−0.3（≥30.86/19.67）；AG/R5/R10/GD ≥ V2−0.2；先调 own（RGB/NIR 0.60→0.65 或 TIR 0.80→0.85），单变量。
- 若 W 的 cross/TIR 不达标 → cross 机制本身不够，不是权重；若 G screening 都失败 → 第二轮先调 own；不要在中间同时改 cross/global。

### Tier2（paper+1，第一轮不承诺）

- 1000 12.0/30.5/47.3/55.8；
- GD 15.5/32.9/46.0/53.5；GD R5≥46.0。

> `Δoverall mAP≈0.90×Δmean₉`（V1/V2 直接比 0.898；历史回归 0.908）：same 保持时 **cross 平均** +0.3 → mean₉ +0.2 → overall≈+0.18；要 overall+0.4 需 **cross 平均** +0.66，或 same 上涨。**不能同时写 `cross+0.3 & overall+0.4`。**

---

## P0/P1 前置条件（120ep 前）

| # | 条件 | 处理 |
|---|---|---|
| P0-1 | 分支覆盖：V2 的 **6 处** `gpd_type` 分派 + **1 处诊断**（`WEV2/WHU/processor/processor.py:597-600,608-646,647-680,710-738,756-791,809-833,888-941`）必须全部接 `'tpd'` | bank 必须 `HierarchicalPrototypeBank`；warm start 必须 `warm_start_hierarchical`；loss 必须 `GPD_TPD`；update 必须 P_id+P_scene；诊断必须 `diagnose_gpd_tpd_gradients`（不能调 `GPD_HPD`）；1ep smoke 打印/断言；加 `GPD_TYPE != 'single' and 'gpd' not in loss_type` 守卫 |
| **P0-2** | **逐 query 权重聚合**：设计伪代码把标量 CE × length-3 权重，会退化成平均权重 | 用 `reduction='none'` + 展开 per-query 权重，或 per-modality CE 求均值再 /M；单测 `[1,1,3]`/`[0.6,0.6,0.7]` 正确值 1.10；`own=1` 与 `GPD_SC` 数值相等 |
| P1-1 | 权重/效果 | RGB/NIR 0.60 是设计点 cross-first，保留；**TIR 0.70 不 approve**（own < V2 0.75、非 own 0.30 > V2 0.25）；改用 TIR 0.80/0.10/0.05；AG 无平台项，只做 G 护栏；R5 靠 own |
| P1-2 | Gate 定义 | 用 W（overall/cross/TIR）+ G（第一轮 screening V1−0.5/V2−0.5，第二轮严格 V1−0.3/V2−0.2）；补 NIR→TIR；paper+1 Tier2；不同写 cross+0.3 & overall+0.4 |
| P1-3 | P_id/global 功能层 | `P_id` 是 id_mean/g_cent 均值锚，不是 UAD 独立单原型；global 是均值正则；有效份额 <5% 则三层仅结构成立；`w_global=1` 仅 `GPD_WARMSTART=False` 时等于 UAD |
| P1-4 | cross zero-column | 现有 row-level fallback 不足；per-column 缺失会让 cross 死类；加 `valid_scene[C,K]`、per-column g_cent fallback 或 mask；记录 missing count；2337 必看 |
| P1-5 | 诊断字段/时序 | TPD 专用三项 CE；`g_own/g_global/g_cross` norm/cos/份额；per-modality own_ratio/usage；P_id/P_scene pair cos/norm/zero-column；pre-clip/clip；per-pair R1/TIR/AG/GD；诊断在训练 forward 之前（θ_t/B_t），标注 last-block/eval/rank0 代理 |
| P1-6 | 归因/seed | 设计点主方案是 package（cross-first + TIR 修正）；加法点是 control；`w_cross=0` 仅 own .75/global .25 是精确 V2；`w_global=0` 是两层；`w_own=1` 是 V1；`w_global=1` 是 UAD（无 warm start）；单 seed screening；最终 2–3 seed |
| P1-7 | 配置/卫生 | `GPD_TPD_*` list、length=M、非负、主方案和=1.0，**但不要写死断言**（加法点 control 1.40/1.20）；`GPD_TYPE` 加 `'tpd'`；K==M；`NECK_FEAT='before'`；默认 `single/subcenter` 逐行不变 |

**Agent 复审补充（`review_v3_tpd_agent_final.md`）**：

- **P1-2 诊断**：份额必须按 per-modality 加权梯度算，不能“标量权重 × 未加权范数”。
- **P1-3 cross fallback**：cross 的 per-column fallback 用独立 `P_scene_cross`/mask，不能改 `L_own` 的 P_scene；否则 `cross=0` 不再精确等于 V2。
- **P1-4 文档**：`V3_design.md` 正文仍是旧 TIR 0.70/旧 Gate/6 处/标量权重；实现前更新或明确废弃，不要照抄。
- **P1-5 系数**：overall/mean₉ 直接比 0.898（历史回归 0.908）；cross 平均 +0.3 → overall +0.18；overall +0.4 需 cross 平均 +0.66（agent 的 +0.27/+0.45 是 cross/mean₉ 混用，不采纳）。
- **P1-6 control**：加法 control 若是 own 消融，TIR 用 1.0/0.10/0.05（总 1.20），不是旧设计点 1.0/0.15/0.075；RGB/NIR 1.0/0.20/0.10（总 1.40）。

---

## 第一轮

- 从 `WEV2/WHU` 复制 `WEV3/WHU`；保留 P_id/P_scene、warm start、诊断；把 `GPD_HPD` 扩展为 `GPD_TPD`。
- **6+1 处分派**（含诊断）全部接 `'tpd'`；配置 `configs/TPD_UAD.yml`；第一轮只跑**设计点 + TIR 修正**（RGB/NIR 0.60/0.20/0.10；TIR 0.80/0.10/0.05）。
- 1ep smoke：`own=1,cross=0,global=0` = V1；`cross=0, own=.75, global=.25` = V2；`w_global=0` = 两层；`global=1, warmstart=False` = UAD；逐 query 权重生效；zero-column；三项 CE 有限；bank/warm start/DDP 正常；`single`/`subcenter` 回归。
- 120ep：4 卡 DDP、batch 256、LR 0.032、seed 1234；单 seed screening；最终 2–3 seed；2337 等 1000 gate。
- 若 G（same/AG/R5）低于护栏：第二轮 RGB/NIR own 0.60→0.65 或 TIR 0.80→0.85，单变量；若 W（cross/TIR）不达标：cross 机制本身不够，先 cross +0.05（单变量），不要同时降 own。加法点 control 只在预算允许时跑。不同时上平台 token、TIR adapter、EMA/TTA、PCGrad、硬分配。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| **`review_v3_tpd_final.md`** | **TPD 最终审查（权威）**：P0=2/P1=7；设计点+TIR 修正、W+G gate、加法点 control、6+1 分派、zero-column、TPD 诊断 |
| **`review_v3_tpd_agent_final.md`** | **独立 agent 复审（P0=2/P1=6）**：主方案 approve with conditions；G screening、cross fallback 独立、per-modality 诊断、V3_design 正文旧值、0.898 系数、control TIR |
| `V3_design.md` | V3/TPD 详细设计（主文档）；关键权重/6+1 已更新；§8.2 旧 Gate 和 §9.2 旧标量伪代码仍为历史，实现以 README/审查为准 |
| `review_v3_final.md` | 两层 PGPD 的最终审查（P0=1/P1=6）；其 P0/P1 条件适用于 V3/TPD 实现 |
| `review_pgpd_design.md` | 早期 PGPD 审查；其中 `w_cross=0`/gate 表述已被最终审查修正 |
| `../../WEV2/HPD_final_plan.md` | V2/HPD 单方案设计 |
| `../../WEV2/review_hpd_final.md` | V2 方案最终审查 |
| `../../WEV2/review_hpd_code.md` | V2 代码审查 r1（P0=0/P1=3） |
| `../../WEV2/review_hpd_code_r2.md` | V2 复审（P0=0/P1=0） |
| `../../WEV2/analysis_hpd_mechanism.md` | V2 逐现象机理 |
| `../../WEV2/analysis_hpd_system.md` | V2 系统级机理 |

## 关键数据

- WEV1：11.37/30.51/48.17/56.27；same 31.16/19.97；cross 13.33/9.00；GD 14.84/31.10/44.61/53.76。
- WEV2：11.47/30.52/47.51/56.14；same 30.97/19.96；cross 13.72/9.16；GD 14.91/31.61/44.82/54.17；TIR→TIR −1.22；AG R1 −0.37/−0.52；R5/R10 −0.66/−0.13；GD R5 44.82。
- UAD：11.02/29.71/46.32/54.55；GD 14.40/31.61/46.14/53.56。
- V3/TPD 目标：**保留正面机制 + 弱指标/总体净改善**。第一轮主方案 RGB/NIR 0.60/0.20/0.10 + TIR 0.80/0.10/0.05；W 要求 cross/TIR/overall 超过 V2（TIR 四向含 NIR→TIR）；G 第一轮 screening same ≥V1−0.5、AG/R5/R10/GD ≥V2−0.5，严格版第二轮。不要求 V1 每项不降；paper+1 是 Tier2。加法点 own=1.0 是 control。

## 约束

- 单模型 UAD/ViT；不做 CLIP/分数融合；推理不变。
- 一次一个核心机制；第一轮只一个权重点；<0.5pp 需 2–3 seed。
- 代码默认 `single`/`subcenter` 路径逐行不变；1ep smoke；独立审查 P0/P1=0。
- 不在第一轮同时加平台 token、TIR adapter、EMA/TTA、PCGrad、硬分配。
