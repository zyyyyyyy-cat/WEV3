# WEV3 / PGPD 方案审查：机制、协同、监督与修正

> **最终审查更新（`review_v3_final.md`，权威）**：本文件的 `w_cross=0` 等价需限定为 `own=1`（`own=0.6,cross=0` 是 `0.6·GPD_SC`，不是 V1）；GD R5 46.0 属 paper+1/Tier2，不是 V1 preservation（V1 44.61）；Tier1 `cross +0.3 & overall +0.4` 在加性关系下不可行；TIR/AG/R5 风险高于本文件评估。**以最终审查为准。**
>
> 审查对象：WEV3 的 PGPD 命题（Pair-aware Gallery Prototype Discrimination）
> 代码基线：`CC/WHU`（UAD）、`CC/WEV1/WHU`（GPD-SC）、`WEV2/WHU`（HPD）
> 数据基线：WEV1 / WEV2 的 1000/GD/domain 日志；H014/H015；论文 Table 3/4/6
> 标记：**[代码事实]** / **[实验支持]** / **[机制假设]**

---

## 0. 结论

**PGPD 方向合理，而且比 HPD 更贴近九协议的结构；但当前 WEV3 命题需要一处关键修正：**

- **own 项必须用 WEV1 的 `GPD_SC`（max_k），不能用“硬 own 模态原型”。**
  - `w_cross=0` 时，`GPD_PAIR` 必须严格退化为 WEV1 的 `GPD_SC`；这是消融/归因的锚点。
  - WEV1 的 `max_k` 是被证明有效的场景项；`own_ratio` 0.92–1.0 说明它绝大多数时候已经选 own；硬 own 是未经验证的新路由，没有证据优势。
- **cross 项用“另外两个 gallery 模态”的专属原型**，不是均值；每个 query 有两个 cross CE；权重 `w_own=0.6, w_cross=0.2+0.2`，总 GPD=1.0。
- **实现基线用 WEV1，不用 WEV2**：不需要 P_id / HierarchicalPrototypeBank；复用 `SubCenterPrototypeBank`、`warm_start_subcenters`、`GPD_SC`；新增 `GPD_PAIR` 与 `GPD_TYPE='pair'`；默认 `single`/`subcenter` 逐行不变。
- **第一轮只跑 own 0.6 / cross 0.2+0.2 一个点**；TIR-specific own 权重是第二轮，不同时上。
- **预期**：cross 比 HPD 更直接（0.4 vs 0.25 的 GPD 预算，且是 gallery 专属），但 own 从 0.75 降到 0.6，R5/AG/R10 有回退风险；TIR same 由 own 保护、TIR cross 由 gallery 项直接拉；净效果需第一轮验证。

---

## 1. 三条代码链路的复核

### 1.1 UAD（`CC/WHU`）

- `GPD`：`P[C,D]`，`f@P.T/τ`，CE；`P ← EMA(P,g_cent)`；单原型。[代码事实]
- 优势：全局身份原型 + 全局负类；跨模态锚点强。
- 薄弱：单原型把三模态平均；同模态多峰被压缩；WEV1 的 same/AG 收益它没有。

### 1.2 WEV1（`CC/WEV1/WHU`）

- `GPD_SC`：`logits = einsum('nd,ckd->nck', f, P_scene)/τ`；`logits.max(dim=2).values`；CE；`P_scene[C,3,D]`。[代码事实]
- `P_scene` 用 ProCA 模态中心更新；count EMA；warm start；推理不变。
- 优势：同模态/AG 强；`max_k` 是 proven 的场景项；`own_ratio` 0.92–1.0。
- 薄弱：cross 平；没有显式的 gallery 模态监督。

### 1.3 WEV2（`WEV2/WHU`）

- `GPD_HPD`：`P_id[C,D]`（g_cent EMA）+ `P_scene[C,3,D]`；`0.25 CE_id + 0.75 CE_scene`；`P_id = mean(P_scene)`（cos 0.999）。[代码事实/实验支持]
- 结果：cross +0.39/+0.16 vs WEV1；same −0.19/−0.01；TIR→TIR −1.22；AG R1 −0.37/−0.52；R5/R10 −0.66/−0.13；GD R5 44.82 <46.0。
- 诊断：shared 10.6%、scene 23.3%、base 65.8%；cos(g_id,g_scene)=0.83–0.94；clip 1.73/100。
- 根因：`P_id` 是均值，pair-agnostic；TIR 离均值远（0.919）；一个均值无法适配九协议。

---

## 2. PGPD 的正确形式

### 2.1 损失

对每个 query（模态 m），保留 WEV1 的 scene 项：

$$
\mathcal L_{own}=\mathrm{CE}\left(\max_k \frac{u^\top P_{scene}[c,k]}{\tau},y\right)
$$

对另外两个 gallery 模态 m'（m'≠m）：

$$
\mathcal L_{cross}^{m'}=\mathrm{CE}\left(\frac{u^\top P_{scene}[c,m']}{\tau},y\right)
$$

总 GPD：

$$
\mathcal L_{PGPD}=w_{own}\,\mathcal L_{own}+w_{cross}\sum_{m'\ne m}\mathcal L_{cross}^{m'}
$$

- `w_own + (M-1)·w_cross = 1.0`；第一轮 `w_own=0.6, w_cross=0.2`（M=3）。
- `w_cross=0` 严格退化为 WEV1 的 `GPD_SC`；这是最重要的消融锚点。
- 负类：每个 CE 都是全部 C 个身份；own 的负类是 max_k 子中心；cross 的负类是同一 gallery 模态 m' 的子中心；全局覆盖，不是 batch-local。
- 更新：`P_scene` 仍用 ProCA 模态中心 + count EMA；warm start 不变；**不需要 P_id**；推理不变。

### 2.2 为什么 own 用 max_k 而不是硬 own

- `max_k` 是 WEV1 的原机制，已有 120ep 证据；`own_ratio` 0.92–1.0，说明它绝大多数时候已经选 own；硬 own 只是把 8% 的边缘情况固定，收益未知，却破坏了“`w_cross=0` = WEV1”的等价性。
- `max_k` 允许样本选“最像自己的子中心”；在 TIR 特征弱、子中心有噪声时，硬 own 会把样本强行拉向可能不准的 TIR 子中心；`max_k` 更稳健。
- `w_cross=0` 时，`GPD_PAIR` 必须与 `GPD_SC` 数值等价（同一 bank 状态）；这是代码审查/单元测试的硬要求。

### 2.3 代码形态（WEV3 建议）

```python
def GPD_PAIR(feat, target, bank, ids, g_cent, num_modalities,
             tau=0.03, own_weight=0.6, cross_weight=0.2):
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat.float(), dim=1)
    with torch.no_grad():
        P = bank.sub_centers.float().clone()
        # zero-row fallback: g_cent expanded (same as GPD_SC)
        ...
        P = F.normalize(P, dim=2)

    logits_all = torch.einsum('nd,ckd->nck', f, P) / tau   # [N,C,K]
    loss_own = F.cross_entropy(logits_all.max(dim=2).values, target_rep)

    B = target.shape[0]
    cross_terms = []
    for m in range(num_modalities):
        q = logits_all[m * B:(m + 1) * B]                  # [B,C,K]
        t = target_rep[m * B:(m + 1) * B]                  # [B]
        for m2 in range(num_modalities):
            if m2 == m:
                continue
            cross_terms.append(F.cross_entropy(q[:, :, m2], t))
    loss_cross = sum(cross_terms) / len(cross_terms)       # mean over 2 terms
    return own_weight * loss_own + cross_weight * (num_modalities - 1) * loss_cross
```

- `w_cross=0` → `GPD_SC`；`w_own=1, w_cross=0` 是 WEV1。
- `cross_weight * (M-1)` 使 `0.6 + 0.2*2 = 1.0`；`loss_cross` 是 2 项的均值，乘 0.4 后总 cross 0.4。
- 代码复用 WEV1 的 `SubCenterPrototypeBank`、`warm_start_subcenters`、`GPD_SC`；新增 `GPD_PAIR` 与 `GPD_TYPE='pair'`；默认 `single`/`subcenter` 不变。
- 不需要 `P_id`/`HierarchicalPrototypeBank`/`warm_start_hierarchical`；WEV3 从 WEV1 复制，代码更小、更可审。

---

## 3. 机制协同

### 3.1 own 与 cross 的协同

- own（WEV1 scene）保留同模态/AG/多峰；cross（gallery 专属）直接监督另外两个 gallery 模态；两者共享 `P_scene`，更新同源，天然一致。
- `w_cross=0` 是 WEV1；`w_cross>0` 是在 WEV1 上增加 pair-aware 监督；不是新分支、不是新特征、不是新推理。
- 对 RGB/NIR：own 与 cross 的原型接近（pair cos 0.914），方向一致，协同好。
- 对 TIR：own（TIR）与 cross（RGB/NIR）方向差（pair cos 0.809），存在张力；`w_own=0.6` 是平衡阀；TIR-specific own 权重是第二轮。

### 3.2 与 ProCA / CE / WRT 的协同

- ProCA 0.01 把模态中心拉向 g_cent；`P_scene` 用模态中心更新；own/cross 共享 `P_scene`；三者同源。
- CE/WRT 是 base（65.8%），PGPD 是次级正则；WRT 已经做跨模态正对，PGPD 的 cross 是全局版本；边际收益仍存在，但比 HPD 的均值更具体。
- BNNeck/CE 的混合 batch 统计也在做跨模态；PGPD 的 cross 是显式的 gallery 监督；互补。

### 3.3 与 HPD 的对比

| | HPD | PGPD |
|---|---|---|
| cross 锚点 | 一个均值 `P_id` | 两个 gallery 专属原型 |
| cross 预算 | 0.25 | 0.4（0.2+0.2） |
| own 预算 | 0.75（scene） | 0.6（WEV1 max_k） |
| TIR same | 被均值拉走 | own max_k 保护 |
| TIR cross | 经过均值 | 直接对 RGB/NIR gallery |
| 归因 | P_id 均值/折中 | 每个 cross CE 可单独看 |

- PGPD 的 cross 更直接、预算更大；own 更小；R5/AG/R10 的回退风险更大；TIR same/cross 的净效果需第一轮验证。
- PGPD 的 `w_cross=0` 严格回到 WEV1；HPD 的 `shared=0` 是 WEV1，但 P_id bank 仍在更新（不影响 loss）；PGPD 的消融更干净。

---

## 4. 损失监督：充分性、冲突、指标对齐

### 4.1 充分性

- own + 两个 cross = 三个全局 CE；每个负类都是全部 C 个身份；覆盖九个 query/gallery 组合；不需要 P_id；不需要一致性 loss；不需要 batch-local。
- `P_scene` 更新由 ProCA 中心提供；cross CE 不直接更新原型（buffer）；原型的模态语义由更新规则保证；监督充分。

### 4.2 冲突

- own 与 cross 对同一特征施加不同方向；RGB/NIR 的 own/cross 原型接近（0.914），冲突小；TIR 的 own/cross 原型远（0.809），冲突大。
- `w_own=0.6` vs cross 0.4 是平衡；如果 `cos(g_own,g_cross)` 强负，TIR 会受损；必须记录 per-m' 的梯度 cos/份额。
- 与 HPD 不同：HPD 的 shared 是均值，与 scene 正相关（0.83–0.94）；PGPD 的 cross 是 gallery 原型，可能与 own 冲突更大；这是 PGPD 的主要风险，不是 HPD 的“稀释”。

### 4.3 指标对齐

- own 保留 WEV1 的 same/AG/R5/R10；cross 直接改善跨模态平均排序/top-1；但 cross CE 仍是分类目标，不直接优化 AP/CMC；R1/R5 的边界仍是涌现。
- cross 的负类是 gallery 原型，可能比 HPD 的均值更硬；cross 可能抬高最难负样本，R1 可能升或降；需要 per-pair R1/最难负样本诊断。
- TIR：cross 直接拉向 RGB/NIR gallery；TIR 的 same 由 own 保护；TIR 的表征弱，cross 不能创造细节；TIR-specific 是第二轮。

---

## 5. 训练/优化

- 120ep 从头 DDP，batch 256/LR 0.032，seed 1234；与 WEV1/WEV2 同配方；只改 GPD；不换 backbone；不加 EMA/TTA/CA-WRT/LLRD/margin。
- 第一轮只跑 `w_own=0.6, w_cross=0.2+0.2` 一个点；不做多臂；TIR-specific own 是第二轮。
- 全局 `clip_grad_norm(1.0)`：HPD 的 pre-clip 1.73/100；PGPD 的 cross 0.4 可能让总梯度更大；监控 pre-clip/clip；不要同时改 max_norm。
- warm start：WEV1 的 `warm_start_subcenters`；P_scene=每 (pid,mod) mean；counts=1；rank0+broadcast；RNG；不需要 P_id。
- DDP：`SubCenterPrototypeBank.update` 的 ids/mod_ids/centers all-gather；bank 不挂 DDP model；顺序固定。
- 诊断：own CE、cross CE per gallery（2 个）、own/cross 梯度份额/cos、per-pair R1、TIR-TIR/TIR→RGB/RGB→TIR、own_ratio、P_scene 几何、pre-clip/clip；不需要 P_id 几何。
- 1ep smoke：`w_cross=0` 与 `GPD_SC` 等价；`w_own=0.6` 的 own/cross CE 有限；bank counts；DDP 不挂；single/subcenter 回归。

---

## 6. 必须修正/确认的点

| # | 点 | 修正/确认 |
|---|---|---|
| 1 | own 项形式 | **用 WEV1 `GPD_SC` 的 max_k**，不是硬 own 模态原型；`w_cross=0` 必须与 `GPD_SC` 数值等价 |
| 2 | cross 项 | 每个 query 用**另外两个 gallery 模态**的 `P_scene[:,m']`；不是均值；两个 CE |
| 3 | 权重 | 第一轮 `w_own=0.6, w_cross=0.2+0.2`；总 GPD=1.0；TIR-specific 是第二轮 |
| 4 | 实现基线 | 从 **WEV1** 复制，不用 WEV2；不需要 P_id/HPD bank；复用 `SubCenterPrototypeBank`/`warm_start_subcenters`/`GPD_SC` |
| 5 | 代码接口 | 新增 `GPD_PAIR`、`GPD_TYPE='pair'`、`GPD_PAIR_OWN_WEIGHT`/`GPD_PAIR_CROSS_WEIGHT`；默认 single/subcenter 逐行不变 |
| 6 | 诊断 | per-gallery CE、own/cross 份额/cos、per-pair R1、TIR、own_ratio、P_scene 几何、clip；不需要 P_id |
| 7 | 第一轮 gate | preservation（same ≥WEV1−0.2、AG ≥WEV1−0.3、R5≥47.3、R10≥55.8、GD R5≥46.0）+ Tier1（cross mAP/R1 ≥+0.3、overall mAP ≥+0.4） |
| 8 | 风险 | own 0.6 vs WEV1 1.0；TIR own/cross 冲突；cross 硬负样本；R5/AG/R10；TIR 表征弱；第一轮 net 不确定 |

---

## 7. 裁决

- **PGPD 方向合理**：它把 HPD 的“一个均值锚点”换成“两个 gallery 专属锚点”，直接对应九协议；保留 WEV1 的 `max_k` scene；`w_cross=0` 严格回到 WEV1；无 P_id、无 batch-local、无推理改动。
- **WEV3 命题需要上述 1–8 的修正**；其中 #1（own 用 max_k）和 #4（从 WEV1 复制）是最重要的，能显著降低实现风险与归因歧义。
- **第一轮只跑 own 0.6/cross 0.2+0.2**；预期 cross 比 HPD 更直接（0.4 vs 0.25，且 gallery 专属），但 R5/AG/R10/TIR 的代价需实测；如果 cross 仍不够，第二轮再调 own/cross 或 TIR-specific，不同时上。
- **不要**同时加平台 token、TIR adapter、EMA/TTA、PCGrad、硬分配；一次一个变量。
