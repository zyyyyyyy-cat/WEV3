# WEV3 / TPD 独立最终审查（review_v3_tpd_agent_final）

> 审查对象：`WEV3/review_v3_tpd_final.md`（权威）、`WEV3/README.md`、`WEV3/V3_design.md`、`WEV3/review_v3_final.md`（旧两层，仅参考）。
> 代码基线：`CC/WHU`（UAD）、`CC/WEV1/WHU`（GPD-SC）、`CC/WEV2/WHU`（HPD）。论文：`CC/WEV1/WHU/tmp/pdfs/WHU.txt`。
> 数据：WEV1/WEV2 的 1000、GD、AG、detailed-domain 日志。
> 标记：**[代码事实]** / **[实验支持]** / **[机制假设]**。数值均给日志/代码来源。
> 约束：只审查，不改项目代码；数值验证只在 `%TEMP%` 临时脚本中完成。
> 与 `review_v3_tpd_final.md` 的关系：接受其主方案（RGB/NIR `0.60/0.20/0.10`，TIR `0.80/0.10/0.05`）与 P0-1/P0-2；本文件只报“仍未关闭 / 新发现”的实质问题，不重复已关闭项。

---

## 0. 裁决

**裁决：approve with conditions。**

**一句话理由**：TPD 的“P_id 全局锚点 + P_scene 模态 max_k + 对另外两个 gallery 模态的 cross CE”结构正确、可运行、推理不变；主方案权重（RGB/NIR 0.60/0.20/0.10，TIR 0.80/0.10/0.05）可以进入第一轮实现，但必须先关闭 P0-1/P0-2，并修正 P1-1/P1-4 的 gate 与文档错误；P1-2/P1-3/P1-5/P1-6 是实现/归因条件。

- **P0 = 2**
  - **P0-1 分支覆盖（6+1）**：V2 有 6 处 `gpd_type` 分派 + 1 处诊断分派；`'tpd'` 必须全部走 hierarchical 路径，否则静默落回 HPD/single 或跳过 warm start/bank update；诊断若仍调 `GPD_HPD`，会报告 V2 的两项份额。
  - **P0-2 逐 query/per-modality 权重聚合**：V3_design §9.2 仍把标量 CE 乘 length-3 权重；这在 PyTorch 里返回 `[3]` 或退化为“平均权重”，无法实现 TIR-aware 的逐 query 模态加权。
- **P1 = 6**
  - **P1-1 Gate 护栏与主方案 own 下降不自洽**：AG/R5/R10/GD/same 的 V2−0.2 / V1−0.3 护栏，没有对应主方案 `own 0.75→0.60`、`global 0.25→0.20/0.10` 的机制支撑；且 “TIR 其他方向” 漏了 NIR→TIR。
  - **P1-2 诊断的有效份额必须按 per-modality 加权梯度计算**：用标量权重乘未加权梯度范数，会错误归因 TIR 的 global/cross 份额。
  - **P1-3 cross 单列 fallback 不能改变 `L_own` 的 P_scene**：否则 V1/V2 锚点的 `max_k` 行为改变，`cross=0` 不再精确等于 V2。
  - **P1-4 V3_design 正文仍是旧方案**：§4.2/§7/§8.2/§9.2/§9.3/§9.4/§10.1/§11/§13 仍写 TIR `0.70/0.15/0.075`、旧 Gate A/B/C、“6 处分派”漏诊断、标量×length-3 伪代码；实现者可能复制错。
  - **P1-5 Gate 数值/噪声**：`>V2` 精确阈值 + 单 seed + `<0.5pp` 不裁决；`0.908×mean₉` 与 `+0.18/+0.66` 不能由 V1/V2 日志复现（实际约 0.898；+0.3→+0.27；+0.4 需 +0.45）。
  - **P1-6 加法 control 的 TIR 权重与主方案不一致**：control 的 TIR `1.0/0.15/0.075` 是旧设计点，不是主方案的 `1.0/0.10/0.05`；拿它归因主方案会混淆 TIR own 与 global/cross。

**主方案是否可进入实现**：**可以，但需先完成 P0-1/P0-2 的实现规格，并把 P1-4 文档更新、P1-3 cross fallback 规格、P1-2 诊断规格写进实现；P1-1/P1-5/P1-6 的 gate/归因需用户明确接受或修正。** 主方案本身不 reject；AG/R5/GD/same 护栏是主要风险，若它们是不可谈判的硬 gate，主方案很可能被一个权重设计无法满足的护栏判死。

---

## 1. 代码/论文事实核对表

| # | 事实 | 证据 |
|---|---|---|
| 1 | UAD `PCA` 返回 `loss, ids, g_cent`；`ids=target.reshape(-1,group_size)[:,0]`；`g_cent` 是 M 个模态中心均值并 detach | `CC/WHU/processor/processor.py:13-30` |
| 2 | UAD `PrototypeBank` 零初始化 `[C,D]`；`update` 用 `g_cent` EMA 0.2（new 权重）；无 warm start | `CC/WHU/processor/processor.py:35-71` |
| 3 | UAD `GPD`：`f@P.T/τ`，CE；仅本 batch 出现的 identity 若原型为零，用 `g_cent` 替换；`F.normalize(feat)` 未显式 `.float()` | `CC/WHU/processor/processor.py:74-88` |
| 4 | UAD 训练总损失 `CE+WRT+0.01·ProCA+1.0·GPD`；推理 pre-BN CLS + L2 + 欧氏 | `CC/WHU/processor/processor.py:187-227`；`utils/metrics.py` |
| 5 | V1 `SubCenterPrototypeBank`：`sub_centers[C,K,D]`、`counts[C,K]`；K=3 是模态数，不是聚类 K=4 | `CC/WEV1/WHU/processor/processor.py:76-162` |
| 6 | V1 更新：ProCA 模态中心 + count EMA `clamp(1/(count+1),0.1,0.5)`；缺失列整行 `g_cent` fallback | `CC/WEV1/WHU/processor/processor.py:109-162` |
| 7 | V1 warm start：每 `(pid,modality)` 类均值；缺失回退 `id_mean`；counts=1；rank0+broadcast；RNG 保存/恢复 | `CC/WEV1/WHU/processor/processor.py:182-231` |
| 8 | V1 `GPD_SC`：`einsum('nd,ckd->nck')`，`max(dim=2)`，CE；`target.repeat(M)` | `CC/WEV1/WHU/processor/processor.py:234-251` |
| 9 | V1 6 处分派：类型、bank、warm start、loss、update、counts 日志 | `CC/WEV1/WHU/processor/processor.py:317-319,323-348,349-371,413-432,446-460,499-511` |
| 10 | V2 `HierarchicalPrototypeBank`：`P_id[C,D]` + `P_scene[C,K,D]`；两个 buffer；`update` 先 id 后 scene | `CC/WEV2/WHU/processor/processor.py:165-230` |
| 11 | V2 warm start：`_warm_start_centers` + `warm_start_hierarchical`；P_scene 模态均值，P_id=valid id_mean；counts=1；broadcast | `CC/WEV2/WHU/processor/processor.py:250-337` |
| 12 | V2 `GPD_HPD`：`0.75·CE_scene + 0.25·CE_id`；P_scene 的 max_k 与 P_id 各一个 CE；diag own_ratio/usage/stats | `CC/WEV2/WHU/processor/processor.py:360-435` |
| 13 | V2 诊断：last-block+norm、eval、rank0；调 `GPD_HPD`、读 `GPD_SCENE_WEIGHT` | `CC/WEV2/WHU/processor/processor.py:438-525` |
| 14 | V2 训练循环：诊断在训练 forward 之前（`optimizer.zero_grad` 后、forward 前）；loss 分支、update、epoch 末几何日志 | `CC/WEV2/WHU/processor/processor.py:710-738,756-791,809-833,888-941` |
| 15 | V2 类型/bank/warm start/守卫：允许 `single/subcenter/hierarchical`；`GPD_TYPE != 'single' and 'gpd' not in loss_type` 抛错；K==M 检查 | `CC/WEV2/WHU/processor/processor.py:596-680` |
| 16 | `PKMSampler` 每 batch 是 P 个 identity × K=4 instances × M 模态；`PKMSampler_DDP` 每卡 64 样本（16 pid×4），×3 模态=192 图 | `WEV2/WHU/datasets/sampler.py:105-140`；`sampler_ddp.py:201-287`；`make_dataloader.py:66-98` |
| 17 | `train_collate_fn` 把每个模态 stack 成 `imgs[m]`；`make_model.forward(mode=0)` 用 `torch.cat(list(x),0)`；`global_feat` modality-major | `WEV2/WHU/datasets/make_dataloader.py:17-23`；`model/make_model.py:67-80` |
| 18 | 训练用 `model(imgs,target,camids)`（mode=0）；推理用 `model(img,mode=mode)`，`NECK_FEAT='before'` 返回 pre-BN CLS | `model/make_model.py:67-80`；`processor.py:740-741,987-991` |
| 19 | 论文 GPD Eq.7–9 是单个身份原型、全体训练身份 CE；ProCA Eq.4–6；总目标 Eq.10 | `WHU.txt:477-580` |
| 20 | 论文配方：P=16、M=3、K=4、每卡 192 图、LR 0.008/GPU、SGD 120ep、5ep warmup+cosine、α=0.8、τ=0.03、λ_ProCA=0.01、λ_GPD=1.0 | `WHU.txt:591-602` |
| 21 | 论文 Table 3 UAD：1000-GD 14.5/31.9/45.0/52.5；1000 11.0/29.5/46.3/54.8；2337 9.6/25.7/40.9/48.2 | `WHU.txt:608-622` |
| 22 | 论文 Table 4 AG：UAD A→G 11.5/31.9/58.3，G→A 13.3/30.5/54.7；Table 6 九协议；Table 8 消融 | `WHU.txt:661-706,718-723` |
| 23 | V1 1000：11.37/30.51/48.17/56.27；same 31.16/19.97；cross 13.33/9.00；AG 35.11/32.66；GD 14.84/31.10/44.61/53.76 | `WEV1/logs/SCGPD_1000/train_log_20261029_2209.txt:1223-1231,1248-1254`；GD `...SCGPD_1000GD/test_log_20261030_0954.txt:173-181,199-202` |
| 24 | V2 1000：11.47/30.52/47.51/56.14；same 30.97/19.96；cross 13.72/9.16；AG 34.74/32.14；GD 14.91/31.61/44.82/54.17；TIR→TIR 21.12/14.55 | `WEV2/logs/HPD_1000/train_log_20261031_0946.txt:1826-1834,1851-1857`；GD `...HPD_1000GD_test/test_log_20261031_1500.txt:180-188,206-209` |
| 25 | V2 末轮诊断：share id/scene/base/pca 0.106/0.233/0.658/0.002；own_ratio [0.922,0.938,1.0]；usage [63,65,64]；cos(g_id,g_scene) 0.831；g_cent_norm 0.922；pre-clip 1.730、clip 100/114；cos(P_id,mean P_scene) 0.999；pair cos [0.914,0.800,0.809] | `WEV2/logs/HPD_1000/train_log_20261031_0946.txt:1814`；几何日志 `:1838-1850` |
| 26 | 训练集 C=500（bank counts [500,500,500]）；每卡 B=64、M·B=192 | `WEV2/logs/HPD_1000/...:1814`；`datasets/sampler_ddp.py:201-287` |
| 27 | repo 配方：4 卡 DDP、global batch 256、每卡 64 样本×3=192 图、LR 0.032/GPU（`make_optimizer` 不做 world-size 缩放）、SGD 120ep | `configs/HPD_UAD.yml:49-63`；`solver/make_optimizer.py:4-29`；`train.py:108-118` |

---

## 2. 数据流、梯度流、损失聚合逐项

### 2.1 数据流

- **[代码事实]** `train_collate_fn` 按 `m in range(num_modalities)` 把每个模态 stack 成 `imgs[m]`；`ImageDataset.__getitem__` 按 `cfg.DATASETS.MODALITIES=['RGB','IR','Thermal']` 顺序返回。`make_model.forward(mode=0)` 用 `torch.cat(list(x),0)`，因此 `global_feat` 是 modality-major `[RGB B; NIR B; TIR B]`（`make_dataloader.py:17-23`；`make_model.py:67-73`）。
- **[代码事实]** `target` 是 identity-major `[P 个 identity × K=4 instances]`；`target.repeat(M)` 得到 `[RGB 段; NIR 段; TIR 段]`，与 `global_feat` 的 modality-major 分段一致（`processor.py:707-708`）。
- **[代码事实]** `ids = target.reshape(-1,group_size)[:,0]`，形状 `[P]`，与 `g_cent [P,D]`、`centers[m] [P,D]` 对齐；`PKMSampler` 的 `final_idxs` 以 identity-major 组织，`target` 的每个 4-instance 组是同一 pid（`sampler.py:127-140`；`processor.py:29,748-753`）。
- **[代码事实]** `B=target.shape[0]` 是每个模态的 batch size（当前 64）；`q=logits_scene[m*B:(m+1)*B]`、`t=target_rep[m*B:(m+1)*B]`、`q[:,:,m']` 与这个顺序一致。若以后改 sampler 或最后 batch 不齐，仍应使用 `[x.shape[0] for x in imgs]` 分段；当前 loader 每个模态 B 相同，`B` 可用。
- **[代码事实]** 训练是 `model(imgs,target,camids)`（mode=0）；推理是 `model(img,mode=mode)`；`NECK_FEAT='before'` 返回 pre-BN CLS。TPD 的 bank/损失不进入推理（`make_model.py:67-80`；`processor.py:740-741,987-991`）。
- **[代码事实]** `PKMSampler`/`PKMSampler_DDP` 对缺失模态的 `cur=0` 会 `divmod(need,0)` 抛 ZeroDivisionError；因此 2337 的 **训练集** 必须每个 identity 三模态齐全；query/gallery 缺失模态由 `dataset.query.get(modality,[])` 处理。若 2337 训练集有缺失模态，这不是 TPD 的 cross fallback 能解决的，必须先改 sampler（不在第一轮范围）。

### 2.2 梯度流

- **[代码事实]** `L_own`、`L_global`、`L_cross` 都只回传到 `global_feat`/ViT；`P_id/P_scene` 是 `register_buffer`，loss 里用 `.clone()` + `no_grad` + `F.normalize`，bank 不接收梯度（`processor.py:374-394` 的 V2 形态；TPD 复用）。
- **[代码事实]** `g_cent` 由 `PCA` 产生并 `detach`；`centers[m]` 有梯度，用于 `L_PCA` 回传；bank 更新在 `no_grad` 下用 `centers`/`g_cent`（`processor.py:26-32,748-753,809-833`）。
- **[代码事实]** `optimizer.step()`/`scaler.step` 在 bank update 之前；bank update 在 `optimizer.step` 之后（`processor.py:799-833`）。**[代码事实]** bank 不是 `model` 的子模块，`model_to_save.state_dict()` 不含 bank；resume 不恢复 bank（`processor.py:618,957-963`）。
- **[代码事实]** DDP all-gather 顺序：`PrototypeBank.update` 先 `ids` 后 `g_cent`；`SubCenterPrototypeBank.update` 先 `ids` 后 `mod_ids` 后 `centers`；`HierarchicalPrototypeBank.update` 先 id bank 后 scene bank。所有 rank 调用顺序一致，tensor 形状一致（`processor.py:48-55,120-134,216-225`）。
- **[代码事实]** warm start：rank0 跑 `warm_start_hierarchical`，barrier 后 `HierarchicalPrototypeBank.broadcast` 广播 P_id/P_scene/counts；RNG 保存/恢复（`processor.py:647-680,227-230`）。**注意**：rank0 与其他 rank 在 warm start 前的模型参数必须相同；当前 `set_seed` + 同一 ImageNet 初始化满足。
- **[代码事实]** AMP/float：V2 `GPD_SC`/`GPD_HPD` 对 `feat`、`P_scene`、`P_id`、`g_cent` 显式 `.float()`；UAD `GPD` 未显式 `.float()`。ViT 的 `forward_features` 以 `self.norm(x)` 结尾，autocast 下 LayerNorm 输出通常为 fp32，所以 UAD 的实际 logits 也多为 fp32；但 TPD 应保持 V2 的显式 `.float()`，不要依赖这个细节。UAD 锚点是“目标/结构等价”，不是逐位相等。
- **[代码事实]** `find_unused_parameters=True`；`base.fc` 不参与 loss；bank 不挂 DDP。诊断用 `model_meta`（unwrapped）在 rank0 上做额外 forward，不触发 DDP collective（`processor.py:562-568,710-738`）。

### 2.3 损失聚合（P0-2 的正确实现）

**唯一权威解释**：`w_own/w_global/w_cross` 是 length-M 的 **per-query-modality** 权重；`w_cross[m]` 是 **每个 gallery 模态 m'** 的权重，per query 的总 cross 贡献是 `2·w_cross[m]`。主方案每 query 合计 1.0；加法 control 合计 1.40/1.30。

**正确实现（二选一，数值等价；因为每个模态 B 相同）**：

```python
# 形式 A：per-modality，mean over B，再 /M
loss = 0.0
for m in range(M):
    own_m   = F.cross_entropy(logits_scene[m*B:(m+1)*B].max(dim=2).values, target)
    glob_m  = F.cross_entropy(logits_id[m*B:(m+1)*B], target)
    cross_m = sum(
        F.cross_entropy(logits_scene[m*B:(m+1)*B][:, :, mp], target)
        for mp in range(M) if mp != m
    )
    loss += (w_own[m] * own_m + w_global[m] * glob_m + w_cross[m] * cross_m) / M

# 形式 B：per-query reduction='none'，权重展开到 [M*B]，再 mean
own_q   = F.cross_entropy(..., reduction='none')          # [M*B]
glob_q  = F.cross_entropy(..., reduction='none')          # [M*B]
cross_q = torch.empty(M*B)
for m in range(M):
    cross_q[m*B:(m+1)*B] = sum(cross_ce[m,mp] for mp != m) # [B]
w_own_q   = torch.repeat_interleave(w_own, B)
w_global_q= torch.repeat_interleave(w_global, B)
w_cross_q = torch.repeat_interleave(w_cross, B)
loss = (w_own_q*own_q + w_global_q*glob_q + w_cross_q*cross_q).mean()
```

**数值验证（`%TEMP%` CPU，fake per-sample CE）**：per-modality own CE `[1,1,3]`、权重 `[0.6,0.6,0.8]`、global 0.5、cross 分别取不同常数：

| 实现 | 结果 |
|---|---:|
| per-modality /M（正确） | **2.8000** |
| per-query `reduction='none'` + mean（正确） | **2.8000** |
| 标量 CE × `sum(w)`（错误） | 6.3333 |
| 标量 CE × `mean(w)`（错误） | 2.1111 |

旧审查的 `[1,1,3]` / `[0.6,0.6,0.7]` 例子同样：正确 1.10，标量均值 1.056，`sum(w)·mean` 3.167。

**锚点数值验证（同一 fake 数据）**：

| 配置 | TPD 结果 | 对照 | 条件 |
|---|---:|---:|---|
| `w_own=1, w_global=0, w_cross=0` | 1.6667 | V1 `GPD_SC`（全 batch CE）1.6667 | scene bank/warm start/update 与 V1 相同；P_id 仍更新但不进 loss |
| `w_cross=0, own=.75, global=.25` | 1.3750 | V2 `GPD_HPD` 1.3750 | P_id/P_scene 同 V2；精确 V2 |
| `w_global=1, own=cross=0` | 0.5000 | UAD `GPD` 0.5000 | **必须 `GPD_WARMSTART=False`**；否则 P_id 是 warm-started id_mean，不是 UAD 的零初始化；float 差异是目标等价，不是逐位相等 |
| `w_cross=0, own=.6, global=.2` | — | 0.6·scene+0.2·id | 是 V2 的缩放/不同权重，不是 V2 |

**必须写进代码/文档的约束**：
- 不要写死 `sum(w)=1.0` 断言；主方案 1.0，加法 control 1.40/1.30。只校验 length=M、每个元素非负、每 query 和 >0。
- `w_cross[m]` 是“每个 gallery m'”的权重；实现时 `Σ_{m'≠m}` 已包含两个方向，**不要**再把 `w_cross` 当总 cross 权重再乘 2（除非文档明确它是总权重；当前表是 per-m'，所以总 cross = 2·w_cross[m]）。
- `w_own/w_global/w_cross` 从 YACS list 读入后转 tensor/float；不要用 Python tuple 直接和 tensor 乘。
- 训练总损失仍是 `loss_fn`（CE+WRT）+ `0.01·L_PCA` + `1.0·L_TPD`；TPD 的权重和（1.0）不改变总 GPD 相对 base 的尺度；加法 control 1.40/1.30 会改变，必须单独记录。
- `GPD_SCENE_WEIGHT` 不参与 TPD；日志不要打印它作为 TPD 的 alpha。

### 2.4 bank 更新、DDP、AMP、warm start

- **[代码事实]** P_id：`PrototypeBank.update`，`g_cent` EMA 0.2（new 权重，等价论文 Eq.7 的 α=0.8 old）；无 count EMA。
- **[代码事实]** P_scene：`SubCenterPrototypeBank.update`，ProCA 模态中心 + count EMA `clamp(1/(count+1),0.1,0.5)`；warm start 后 counts=1，首次更新 m_eff=0.5，之后趋近 0.1。
- **[代码事实]** 缺失列 fallback：V1/V2 只在 `norms[ids].sum(dim=1) < 1e-6`（整行全零）时用 `g_cent` 替换整行；**单列缺失不会触发**。TPD 的 cross 必须单独处理（P1-3）。
- **[代码事实]** warm start：`_warm_start_centers` 逐模态计算类均值，`id_mean` 在填充缺失前计算；`warm_start_hierarchical` 初始化 P_scene/P_id；rank0+broadcast；finite 检查。TPD 复用，不要改。
- **[代码事实]** DDP：bank 不入 `state_dict`；每个 iteration 的 all-gather 强制同步；warm start 后各 rank 一致；eval 的 `model.module.buffers()` 广播不含 bank（bank 不用于 eval）。
- **[代码事实]** AMP：训练 forward 在 `amp.autocast` 内；bank update 在 `no_grad`；`scaler.unscale_` 后 `clip_grad_norm_(1.0)`；TPD 的三项 CE 都应在 float32 下计算（至少 `f/P` 显式 `.float()`）。
- **[代码事实]** 诊断在训练 forward 之前：V2 在 `optimizer.zero_grad()` 后、`model(imgs,...)` 前；TPD 必须保持这个位置，否则 own_ratio/份额会被 `θ_{t+1}/B_{t+1}` 污染。
- **[代码事实]** 训练集 C=500（bank counts [500,500,500]），每卡 B=64、M·B=192；`target.repeat(3)` 后 [192]；`P_scene [500,3,768]`、`P_id [500,768]`。

---

## 3. 主方案权重与 W+G gate 的数值/机制评估

### 3.1 三条基线与主方案

| 方案 | own（RGB/NIR/TIR） | global | cross（每个 m'） | 每 query 合计 |
|---|---:|---:|---:|---:|
| UAD | 0 | 1.0 | 0 | 1.0 |
| V1 | 1.0 | 0 | 0 | 1.0 |
| V2 | 0.75 | 0.25 | 0 | 1.0 |
| **V3/TPD 主方案** | **0.60/0.60/0.80** | **0.20/0.20/0.10** | **0.10/0.10/0.05** | **1.00** |
| 加法 control（非主方案） | 1.00/1.00/1.00 | 0.20/0.20/0.15 | 0.10/0.10/0.075 | 1.40/1.40/1.30 |

- **主方案 RGB/NIR**：own 0.60 < V2 0.75；非 own 0.40 > V2 0.25；cross 0.20 是直接跨模态项。**机制假设**：cross 直接对 `P_scene[:,m']` 做全局 CE，应改善 cross/九协议；代价是 same/AG/R5 的 own 力下降。
- **主方案 TIR**：own 0.80 > V2 0.75；非 own 0.20 < V2 0.25；global 0.10 < V2 0.25；cross 0.05×2。**机制假设**：先保 TIR same，再用 0.10 cross 攻 TIR cross。**但** cross 的两个 gallery 列是 RGB/NIR，和 TIR 的 pair cos 约 0.80/0.81（V2 几何日志 pair cos [0.914,0.800,0.809]；cos(P_id,P_scene) TIR 0.919）；cross 比 V2 的均值锚点更“远”，所以“非 own 0.20 < 0.25”只是必要条件，不是 TIR same 改善的充分条件。
- **加法 control**：own=1.0 是 V1；RGB/NIR global/cross 与主方案相同；**TIR global/cross 是旧设计点 0.15/0.075，不是主方案的 0.10/0.05**；总 GPD 1.40/1.30。它是“V1 own + 旧设计点 global/cross”的 package control，不是主方案的单变量 own 消融。

### 3.2 机制逐项

**（1）TIR same/cross**
- **[实验支持]** V2 TIR→TIR 21.12/14.55 vs V1 22.34/14.70（R1 −1.22）；V2 TIR own_ratio=1.0，cos(P_id,TIR)=0.919，pair cos 0.80/0.81；V2 的 0.25 global 是“向 RGB/NIR 主导的均值拉”的力。
- **[机制假设]** 主方案 TIR own 0.80、global 0.10、cross 0.05×2：own 比 V2 高 0.05，非 own 低 0.05；TIR same 有改善机制。但 cross 的两列比均值锚更远，0.10 的 cross 可能比 0.15 的 global 更伤 TIR same；净效果必须由 gate 和 per-term 诊断回答。
- **[机制假设]** TIR cross（TIR→RGB、TIR→NIR、RGB→TIR、NIR→TIR）：TIR query 的 cross 0.05×2 直接拉向 RGB/NIR gallery；RGB/NIR query 的 cross 0.10 都包含 TIR gallery。四个 TIR 相关方向都有直接监督；这是 TPD 相对 HPD 的新增机制。

**（2）RGB/NIR cross/九协议**
- **[实验支持]** V2 的 cross 增益集中在 NIR↔RGB（NIR→RGB +1.46/+0.30，RGB→NIR +0.61/+0.24 vs V1；V2 cross 13.72/9.16 vs V1 13.33/9.00）；RGB↔NIR pair cos 0.914，TIR 是离群。
- **[机制假设]** 主方案 RGB/NIR cross 0.10×2 + global 0.20；cross 是直接的 gallery 分类，应该比 V2 的 mean global 更具体；但 V2 的 cross 只有 +0.16 mAP，WRT/CE 已经做了大量跨模态对齐，边际收益会被吃掉。第一轮是 package，不能单独归因给 cross。

**（3）AG**
- **[实验支持]** V2 AG A→G 34.74/G→A 32.14 vs V1 35.11/32.66（R1 −0.37/−0.52）；V2 的 own 0.75 已经付出 AG R1 代价；主方案 RGB/NIR own 0.60 更低，且 TPD 没有平台项。
- **[机制假设]** AG 是 same-modality、cross-platform；cross 是模态级，不解决平台内 top-1；主方案 AG R1 很可能低于 V2−0.2（34.54/31.94）。AG 只能做护栏；若 AG 是必须改善项，第二轮加平台机制，不在 TPD 权重里硬调。

**（4）R5/R10/GD R5**
- **[实验支持]** R5：V1 48.17（own 1.0）> V2 47.51（own .75/global .25）> UAD 46.32（global 1.0）；R5 靠 own。GD R5：UAD 46.14（global 1.0）> V2 44.82（global .25）> V1 44.61（global 0）；GD R5 靠 global。
- **[机制假设]** 主方案 RGB/NIR own 0.60 < V2 0.75 → R5/R10 风险；global 0.20/0.10 < V2 0.25 → GD R5 风险。V2 的 R5 −0.66 vs V1、GD R5 44.82 < 46.0 已经是证据；主方案再降 own/global，R5/GD R5 很可能低于 V2−0.2（47.31/44.62）。这是 P1-1 的核心。

**（5）P_id/global 的功能层**
- **[实验支持]** V2 `cos(P_id,mean P_scene)=0.999`；shared 有效份额 10.6%、scene 23.3%、base 65.8%；P_id 是均值锚，不是独立的第三判别方向。
- **[机制假设]** TPD 的 global 0.20/0.10 有效份额可能只有个位数；它更像“向心正则”，不是强 GD R5 锚。三层在结构上成立，功能上要由诊断回答；若 `g_global` 独立份额 <5%，第二轮考虑去掉/改 P_id，而不是继续加 global。
- **[代码事实]** `w_global=1` 要精确等于 UAD，必须 `GPD_WARMSTART=False`；主方案 warm start True，P_id 是 warm-started id_mean，不是 UAD 零初始化。

**（6）own 与 cross 的冲突**
- **[实验支持]** TIR own_ratio=1.0，RGB/NIR own_ratio 约 0.89–0.98（V2 末轮 0.922/0.938；TIR 1.0）；TIR pair cos 0.80/0.81，RGB/NIR pair cos 0.914。TIR 的 own 与 cross 张力最大。
- **[机制假设]** `max_k` 只说明 own 选 own，不说明 own 的权重大到能抵消 cross；RGB/NIR 约 3–8% 的样本 own 选到别的模态列，own 与某个 cross 对同一列重复施力；名义权重不等于有效权重，必须看 per-term grad share/cos。

### 3.3 Gate 数值核算（用 V1/V2 日志）

| Gate | 阈值 | 来源/核算 |
|---|---|---|
| W overall | mAP/R1 > 11.47/30.52 | V2 1000 `...:1854-1857`；V2 比 V1 11.37/30.51 只 +0.10/+0.01 |
| W cross | R1/mAP > 13.72/9.16 | V2 六对 cross 均值：R1 (27.12+7.26+25.25+7.03+7.59+8.06)/6=13.72；mAP (16.50+5.71+16.11+5.29+5.98+5.39)/6=9.16 |
| W TIR→TIR | ≥21.12/14.55；target V1 22.34/14.70 | V2/V1 1000 日志 |
| W TIR 其他方向 | 不低于 V2 | **明确列了 TIR→RGB 7.59/5.98、RGB→TIR 7.26/5.71、TIR→NIR 8.06/5.39；漏了 NIR→TIR 7.03/5.29** |
| W AG | 两向 R1 ≥34.54/31.94 | V2 34.74/32.14 −0.2 |
| W R5/R10 | ≥47.31/55.94 | V2 47.51/56.14 −0.2 |
| W GD | R5 ≥44.62；mAP/R1/R10 ≥14.71/31.41/53.97 | V2 GD 14.91/31.61/44.82/54.17 −0.2 |
| G same | 3 对 R1/mAP ≥30.86/19.67 | V1 same 31.16/19.97 −0.3；V2 same 30.97/19.96 已高于该阈值 |
| G AG/R5/R10/GD | V2−0.2 | 同 W |
| Tier2 paper+1 | 1000 12.0/30.5/47.3/55.8；GD 15.5/32.9/46.0/53.5；GD R5≥46.0 | 论文 Table 3 UAD 11.0/29.5/46.3/54.8 与 GD 14.5/31.9/45.0/52.5 各 +1.0；`WHU.txt:608-622` |

**数值自洽性**：
- 阈值本身与 V1/V2 日志自洽；`TIR→TIR` 21.12/14.55、`cross` 13.72/9.16、`same` 30.97/19.96、`AG` 34.74/32.14、`R5/R10` 47.51/56.14、`GD` 14.91/31.61/44.82/54.17 都可从日志复算。
- **W 的 TIR 方向清单漏 NIR→TIR**（7.03/5.29）；如果“TIR 其他方向”是硬 gate，必须补上，否则 TIR 的一个弱方向可以回退而 gate 仍过。
- **G 的 same 阈值 30.86/19.67 低于 V2 的 30.97/19.96**：主方案 same 只要不低于 V1−0.3 即可，这是“允许取舍”的体现；但主方案 RGB/NIR own 0.60，same 很可能跌到 30.86 以下；这是风险不是数值错误。
- **W/G 的 AG/R5/R10/GD ≥V2−0.2 与主方案 own/global 下降不自洽**：V2 的 own 0.75 已经让 AG R1 −0.37/−0.52、R5 −0.66 vs V1；主方案 own 0.60 更低，AG/R5 的 0.2 容差很可能不够；GD R5 的 global 0.20/0.10 < V2 0.25，0.2 容差也很紧。这是 P1-1。
- **加性关系**：从 V1/V2 1000 日志复算，`overall mAP / mean₉ mAP`：V2 11.47/12.761=0.8988，V1 11.37/12.656=0.8984；不是 0.908。same 不变、cross mAP +0.3 → overall mAP ≈ +0.27，不是 +0.18；要 overall +0.4 需 cross mAP ≈ +0.45，不是 +0.66。定性结论“不能同时承诺 cross+0.3 与 overall+0.4”仍成立，但 authoritative review 的 0.908/+0.18/+0.66 三个数互相不一致，必须改。这是 P1-5。
- **paper+1 是 Tier2**：V2 的 R5 47.51 > 47.3、R10 56.14 > 55.8，已在 paper+1 的 1000 R5/R10 之上；paper+1 的 mAP 12.0 仍差 0.53；GD 15.5/32.9/46.0/53.5 是 Tier2。不要把 paper+1 当第一轮硬 gate。

### 3.4 Gate 的最小修正（只给一个主方案）

**推荐（与“允许取舍”一致）**：
- **W 硬指标**：overall mAP/R1 > V2；cross R1/mAP > V2；TIR→TIR ≥ V2；TIR 四个其他方向（含 NIR→TIR）不低于 V2。
- **G 第一轮 screening 护栏**：same ≥ V1−0.5（≈30.66/19.67，或 V2−0.5），AG/R5/R10/GD ≥ V2−0.5（或 V1−0.5）；把 V2−0.2 / V1−0.3 留给第二轮（若第一轮 cross/TIR 达标、但 same/AG/R5 回退，第二轮 own 0.60→0.65 或 0.80→0.85 单变量后再用严格护栏）。
- **near-threshold**：单 seed 只做 screening；`<0.5pp` 不裁决；最终 2–3 seed + paired bootstrap。W 的 `>V2` 精确阈值在单 seed 下会被噪声左右，建议近阈值加 0.1–0.2pp 余量或 2 seed。
- **如果用户坚持 V2−0.2/V1−0.3 是硬 gate**：主方案很可能失败在 AG/R5/GD/same 上；此时唯一的最小修正不是改 gate，而是改主方案 own（RGB/NIR 0.60→0.65、TIR 0.80→0.85 或只改一项），但那会削弱 cross-first，且是另一个主方案；不推荐在第一轮同时做。

---

## 4. P0/P1 表

> 已关闭项不重复：`review_v3_tpd_final.md` 的 P0-1/P0-2、P1-1（TIR 0.80）、P1-2（W+G）、P1-3（P_id 均值锚）、P1-4（zero-column）、P1-5（TPD 诊断）、P1-6（锚点/seed）、P1-7（配置/卫生）均被接受；下表是“仍未关闭 / 新发现”的残余。

| 编号 | 问题 | 证据 | 修法 | 为什么是 P0/P1 |
|---|---|---|---|---|
| **P0-1** | 7 处 `gpd_type` 分派未全部接 `'tpd'`；V3_design §9.3 仍写“6 处”且漏诊断（710-738） | V2 实际分派：`processor.py:597-600`（允许类型）、`608-646`（bank）、`647-680`（warm start）、`710-738`（诊断）、`756-791`（loss）、`809-833`（update）、`888-941`（epoch counts/geometry）；`V3_design.md:316-325` 的 6 处清单不含诊断；README 已改为 6+1 | 用 `is_hier = gpd_type in ('hierarchical','tpd')` 统一 7 处；`'tpd'`：bank=`HierarchicalPrototypeBank`、warm start=`warm_start_hierarchical`、loss=`GPD_TPD`、update=P_id+P_scene、诊断=`diagnose_gpd_tpd_gradients`、epoch 日志=hierarchical 分支；加 `'tpd'` 到 allowed tuple、`GPD_TYPE != 'single' and 'gpd' not in loss_type` 守卫、`GPD_SUB_CENTERS==M`；1ep smoke 断言 `type(proto_bank).__name__=='HierarchicalPrototypeBank'`、`GPD_TPD` 被调用、`GPD_HPD` 不被调用 | **P0**：任一处漏接会静默落回 HPD/single 或跳过 warm start/bank update；诊断漏接会报告 V2 的两项份额；120ep 跑完才发现机制不是 TPD |
| **P0-2** | 标量 CE × length-3 权重；V3_design §9.2 仍保留 `w_own*loss_own + ...` 的伪代码；`w_cross` 的“per m' vs 总 cross”也有歧义 | `V3_design.md:286-314`；`review_v3_tpd_final.md:113-133`；数值验证：per-modality/per-query 2.8 vs 标量均值 2.111 vs 标量×sum 6.333 | 用 §2.3 的形式 A 或 B；`w_cross[m]` 是 per-gallery m'，总 cross=2·w_cross[m]；`reduction='none'` + per-query 权重，或 per-modality mean /M；不写死 sum=1；校验 length=M、非负、和>0；单测 [1,1,3]/[0.6,0.6,0.8] → 2.8；锚点 V1/V2/UAD | **P0**：TIR-aware 是核心；聚合错了就退化成“平均权重 TPD”，第一轮结果无法解释 |
| **P1-1** | W/G 的 AG/R5/R10/GD/same 护栏与主方案 own/global 下降不自洽；“TIR 其他方向”漏 NIR→TIR | V2 own 0.75 已使 AG R1 −0.37/−0.52、R5 −0.66 vs V1（V1/V2 日志）；主方案 own 0.60、global 0.20/0.10；GD R5 的 global 0.20/0.10 < V2 0.25；W 明确列了 TIR→RGB/RGB→TIR/TIR→NIR，漏 NIR→TIR 7.03/5.29 | 第一轮按 §3.4：W overall/cross/TIR 硬；G same/AG/R5/R10/GD 用 V1−0.5/V2−0.5 screening；补 NIR→TIR；严格 V2−0.2/V1−0.3 留第二轮（own 0.60→0.65 或 0.80→0.85 单变量） | **P1**：可能用一个权重设计无法满足的护栏判死主方案；漏 NIR→TIR 会放过一个 TIR 弱方向回退 |
| **P1-2** | TPD 诊断的有效份额若用标量权重×未加权梯度范数，会错误归因 per-modality global/cross | V2 `diagnose_gpd_gradients`：`alpha=GPD_SCENE_WEIGHT`，`eff_id=(1-alpha)*n_id`、`eff_scene=alpha*n_scene`（`processor.py:492-498`）；TPD 权重是 length-3 | 新 `diagnose_gpd_tpd_gradients`：per-term CE（own/global/6 cross）、per-modality 加权梯度（或 per-modality 梯度加权）、norm/cos/份额；own_ratio/usage；P_id/P_scene pair cos/per-column norm/zero-column；pre-clip/clip；在训练 forward 之前、last-block+norm/eval/rank0 代理；不调 `GPD_HPD`、不读 `GPD_SCENE_WEIGHT` | **P1**：归因错误会误导第二轮（例如把 TIR 的 global 份额算大/算小）；机制判断依赖诊断 |
| **P1-3** | cross 单列缺失的 per-column fallback 如果作用在同一个 `P_scene` 上，会改变 `L_own` 的 max_k，V1/V2 锚点不再精确；只做 row-level 又会让 cross 出现死列 | V2 row-level：`need_scene = norms_scene[ids].sum(dim=1)<1e-6`（`processor.py:382-388`）；cross 用 `P_scene[:,m']` 单列；V1/V2 锚点要求 `L_own` 与 GPD_SC/GPD_HPD 同几何 | 保留 `L_own`/`L_global` 的 row-level fallback；为 cross 建独立的 `P_scene_cross`（clone）或 `valid_scene[C,K]` + per-column fallback/mask：batch 内缺失列用 `g_cent` 替换，非 batch 缺失类 mask 或依赖 warm start；记录 missing count；不要改默认 `single/subcenter` | **P1**：对主 run（warm start、1000 三模态齐全）影响小，但 2337 长尾/缺失模态和无 warmstart 锚点会出问题；且可能破坏锚点回归 |
| **P1-4** | V3_design 正文仍是旧方案，可能被实现者直接复制 | `V3_design.md` §4.2（TIR 0.70/0.15/0.075）、§7（TIR 0.70）、§8.2（旧 Gate A/B/C）、§9.2（旧权重 + 标量×length-3）、§9.3（“6 处”漏诊断）、§9.4（YAML TIR 0.7/0.15/0.075）、§10.1（TIR 0.70 预期）、§11/§13（0.60/0.70） | 把正文改为权威主方案：RGB/NIR 0.60/0.20/0.10；TIR 0.80/0.10/0.05；W+G；6+1；per-query/per-modality 伪代码；control 标明是旧设计点/或改为 TIR 1.0/0.10/0.05；在文档顶部明确“以 review_v3_tpd_final + 本文件为准” | **P1**：配置文件/伪代码是直接实现输入；旧 TIR 0.70 会被误抄，且旧 Gate 会误判 |
| **P1-5** | Gate 精确 V2 阈值 + 单 seed + <0.5pp 不裁决；0.908/+0.18/+0.66 互相不一致 | V1/V2 1000：overall/mean₉ mAP = 11.47/12.761=0.8988、11.37/12.656=0.8984；correct +0.3→+0.27、+0.4→+0.45 | 定义 near-threshold 余量/2 seed；把 0.908 改为约 0.898；把 +0.18/+0.66 改为 +0.27/+0.45；`cross+0.3 & overall+0.4` 仍不同时写；单 seed 只 screening | **P1**：会实质影响“过/不过”和下一轮判断；数值必须可复算 |
| **P1-6** | 加法 control 的 TIR 权重（1.0/0.15/0.075）与主方案（1.0/0.10/0.05）不一致；control 不是主方案的单变量 own 消融 | 任务给的 control TIR global 0.15 + cross 0.075×2（总 1.30）；主方案 TIR global 0.10 + cross 0.05×2（总 1.00）；RGB/NIR control 与主方案 global/cross 相同 | 若 control 是主方案的 own 消融：TIR 用 1.0/0.10/0.05（总 1.20），RGB/NIR 1.0/0.20/0.10（总 1.40）；若保留旧值：明确标注“design-point control”，不要拿它归因主方案的 TIR；无论如何，control 总 GPD 1.40/1.30 vs 主方案 1.0，需记录 base/GPD 份额 | **P1**：混淆 TIR own 与 global/cross，归因不干净；control 本来就是可选，但一旦跑了就会被误读 |

---

## 5. 第一轮实现/1ep smoke/诊断最小清单

### 5.1 实现

- [ ] `WEV3/WHU` 从 `WEV2/WHU` 复制；只改 `processor/processor.py`、`config/defaults.py`；新增 `configs/TPD_UAD.yml`、`README_TPD.md`；不改 `model/loss/datasets/utils/solver`。
- [ ] `config/defaults.py` 加 `GPD_TPD_OWN_WEIGHT`/`GPD_TPD_GLOBAL_WEIGHT`/`GPD_TPD_CROSS_WEIGHT`（list，默认主方案或 V1）；validate length=M、非负、每 query 和>0；**不写死 sum=1**。
- [ ] `GPD_TYPE` 允许 `'tpd'`；`GPD_SUB_CENTERS==M`；`GPD_WARMSTART=True` 时 `'tpd'` 走 `warm_start_hierarchical`；`GPD_DIAG_PERIOD`；`LOSS_TYPE` 守卫；`TEST.NECK_FEAT='before'`（建议对 `gpd_type in ('subcenter','hierarchical','tpd')` **无条件**检查，不只 warm start；TPD loss 始终在 pre-BN CLS 上）。
- [ ] `GPD_TPD(feat,target,bank,ids,g_cent,num_modalities,tau,w_own,w_global,w_cross)`：
  - P_id/P_scene clone + `.float()` + no_grad + normalize；
  - `L_own`/`L_global` 用 row-level fallback（与 V1/V2 同）；
  - cross 用独立的 `P_scene_cross`/mask + per-column fallback，记录 missing；
  - per-query `reduction='none'` 或 per-modality `/M`；`w_cross[m]` per gallery；
  - 返回 `loss, loss_own, loss_global, cross_losses, diag`；
  - 不调用 `bank.update`；不读 `GPD_SCENE_WEIGHT`。
- [ ] 7 处分派全接 `'tpd'`（P0-1）；诊断新函数（P1-2）；meter/日志标签 HPD→TPD；epoch counts/geometry 用 hierarchical 分支。
- [ ] 默认 `single`/`subcenter`/`hierarchical` 路径逐行不变；bank 不入 `state_dict`；推理不变。

### 5.2 1ep smoke（必须）

- [ ] 类型断言：`HierarchicalPrototypeBank`；`GPD_TPD` 被调用（monkeypatch 计数）；`GPD_HPD` 不被调用；warm start 日志/counts；`single`/`subcenter` 回归。
- [ ] 聚合单测：per-modality CE `[1,1,3]`、权重 `[0.6,0.6,0.8]` → 2.8（不是 2.111/6.333）；TIR 的 w_own 改变必须真的影响 TIR 梯度；cross 的 w_cross 改变必须影响对应 gallery 的 logits/梯度。
- [ ] 锚点：`own=1,cross=0,global=0` = `GPD_SC`；`cross=0,own=.75,global=.25` = `GPD_HPD`；`global=1,warmstart=False` = UAD（目标）；`w_global=0` 两层。
- [ ] 三项 CE/梯度有限；无 NaN；pre-clip/clip 记录；bank 无梯度；counts；warm start；zero-column/missing；DDP all-gather；AMP；eval 前后模型/BN/RNG/`.grad` 不被诊断污染。
- [ ] `TEST.NECK_FEAT='before'`、`FEAT_NORM='yes'`；1000/GD 配置；`transformer_120.pth` 推理。

### 5.3 诊断（必须）

- [ ] TPD 专用 `diagnose_gpd_tpd_gradients`，在训练 forward 之前（`optimizer.zero_grad` 后、`model(imgs,...)` 前），last-block+norm、eval、rank0 代理。
- [ ] per-term CE：own、global、6 个 cross；per-modality own_ratio/usage；per-gallery cross CE。
- [ ] per-term 加权梯度 norm、两两 cos、有效份额（per-modality 权重乘在项内，不是标量）；pre-clip/clip。
- [ ] P_id/P_scene pair cos、per-column norm、zero-column/missing count、`g_cent_norm`、P_id norm。
- [ ] 指标记录：9 对 R1/mAP/mINP、TIR 四向、AG A→G/G→A、GD 四项、same/cross 平均；标注诊断是代理，不用 diag loss 代替训练 loss。

### 5.4 实验

- [ ] 1000 AS，4 卡 DDP，global batch 256（每卡 192 图），LR 0.032，120ep，seed 1234；与 V1/V2 同配方；只改 GPD。
- [ ] 第一轮只跑主方案（RGB/NIR 0.60/0.20/0.10；TIR 0.80/0.10/0.05）；加法 control 只在预算允许时跑，且按 P1-6 标注/修正。
- [ ] 测试 `transformer_120.pth`，`NECK_FEAT='before'`，`FEAT_NORM='yes'`；1000/GD/AG/detailed-domain；GD `FILTER_GD=True`。
- [ ] 单 seed screening；最终 2–3 seed；`<0.5pp` 不裁决；2337 等 1000 gate 之后。
- [ ] 记录 `WEV3/logs/TPD_1000_DDP`、`CHAT_LOG.md`、`WEV3/README_TPD.md`、`WEV3/review_v3_code.md`；第一轮只写“package + 梯度诊断”，不写“cross 单独贡献 +X”。

---

## 6. 与论文的边界

**论文直接实现的**：
- `L_base = L_CE + L_WRT`（Eq.3）；CE 在 BNNeck 后 `z`，WRT 在 pre-BN `f`（`WHU.txt:463-471`）。
- ProCA Eq.4–6：K 张图求模态中心，模态中心求 identity anchor（stop-gradient），`L_ProCA = 1/P Σ_y 1/M Σ_m ||µ_y,m − µ_y||²`；repo `PCA` 与之一致（`WHU.txt:486-519`；`processor.py:13-32`）。
- GPD Eq.7–9：**单个** identity prototype `p_y`，EMA `p_y ← norm(α p_y + (1−α) µ_y)`，CE 正样本 `p_yi`、负样本所有其他 identity，`|B|` 平均；repo `PrototypeBank`/`GPD` 与之一致（`WHU.txt:536-569`；`processor.py:35-88`）。
- 总目标 Eq.10：`L_base + λ_ProCA L_ProCA + λ_GPD L_GPD`；repo `0.01`/`1.0`（`WHU.txt:575-580`）。
- 推理：`ℓ2-normalize f` 检索；repo pre-BN CLS + `FEAT_NORM=yes` + 欧氏（`WHU.txt:467`；`make_model.py:74-80`）。

**V3 的扩展（不是论文直接实现）**：
- `P_scene[C,K,D]` + `max_k` 是 WEV1 的 GPD-SC 扩展；论文 GPD 没有模态子中心。
- cross `CE(u · P_scene[:,m']/τ)` 对另外两个 gallery 模态是 V2/PGPD 的扩展；论文 Eq.7–9 只有一个单原型，没有 gallery cross。
- `P_id` 在 TPD 里是 warm-started `id_mean` + `g_cent` EMA；论文 GPD 是零初始化/无 warm start；功能相似，但不是逐位相同。
- 模态感知权重（length-3）是 V3 的新机制；论文没有。
- 所以：**不能把 cross 说成论文直接实现；也不能把 P_id 说成 UAD 原版**。文档可以写“UAD 全局锚点的思想 + WEV1 子中心 + V2 cross 的扩展”。

**论文配方 vs repo 配方**：
- 论文：P=16、M=3、K=4、每卡 192 图、LR **0.008/GPU**、SGD 120ep、5ep warmup+cosine、α=0.8、τ=0.03、λ 0.01/1.0（`WHU.txt:591-602`）。
- repo：4 卡 DDP、global batch 256、每卡 64 样本×3=**192 图**、LR **0.032/GPU**（`make_optimizer` 不做 world-size 缩放）、SGD 120ep、5ep warmup+cosine、`GPD_MOMENTUM=0.2`（new 权重 = 论文 α=0.8 old）、τ=0.03、λ 0.01/1.0（`configs/HPD_UAD.yml:49-63`；`solver/make_optimizer.py:4-29`）。
- 结论：每卡图数相同，LR 是论文的 4×（repo 是内部统一配方，不是论文原配方）。V1/V2/TPD 可比，但不要拿论文 Table 3/4/6 的绝对值当同配方；paper+1 是 Tier2。
- `GPD_MOMENTUM=0.2` 与论文 α=0.8 的等价性应在文档中写清（repo 的 `m` 是 new 权重，论文 α 是 old 权重），否则会被误读为 0.2 vs 0.8 不一致。

---

## 7. 证据路径

**代码**
- `CC/WHU/processor/processor.py:13-30,35-71,74-88,187-227`
- `CC/WEV1/WHU/processor/processor.py:76-162,182-231,234-251,317-319,323-348,349-371,413-432,446-460,499-511`
- `CC/WEV2/WHU/processor/processor.py:165-230,250-337,360-435,438-525,596-680,710-738,756-791,809-833,888-941`
- `CC/WEV2/WHU/model/make_model.py:67-80`
- `CC/WEV2/WHU/datasets/sampler.py:105-140`；`sampler_ddp.py:201-287`；`make_dataloader.py:17-23,66-98`
- `CC/WEV2/WHU/config/defaults.py:42-64`；`configs/HPD_UAD.yml:18-28,49-71`；`CC/WEV1/WHU/configs/SCGPD_UAD.yml:18-24,47-50`
- `CC/WEV2/WHU/solver/make_optimizer.py:4-29`；`train.py:94-118`

**论文**
- `CC/WEV1/WHU/tmp/pdfs/WHU.txt:453-580`（Sec.4.2–4.5、Eq.3–10）
- `:591-602`（Implementation details）、`:608-622`（Table 3）、`:661-706`（Table 4/5/6）、`:718-723`（Table 8）

**数据**
- V1 1000：`CC/WEV1/logs/SCGPD_1000/train_log_20261029_2209.txt:1223-1231,1248-1254`；`test_log_20261030_1035.txt`
- V1 GD：`CC/WEV1/logs/SCGPD_1000GD/test_log_20261030_0954.txt:173-181,199-202`
- V2 1000：`WEV2/logs/HPD_1000/train_log_20261031_0946.txt:1826-1834,1851-1857`；诊断 `:1814`；几何 `:1838-1850`
- V2 GD：`WEV2/logs/HPD_1000GD_test/test_log_20261031_1500.txt:180-188,206-209`
- V2 分析：`WEV2/analysis_hpd_mechanism.md:14-21,39-51,144-182,184-208,210-237`；`WEV2/analysis_hpd_system.md:14-17,69-117,154-240`

**V3 文档**
- `CC/WEV3/review_v3_tpd_final.md`（权威；§0.1、§2.2、§4、§5.2、§6、§7）
- `CC/WEV3/README.md:5,25-46,50-73,77-99,118-123`
- `CC/WEV3/V3_design.md:8,98-147,230-260,282-314,316-325,329-352,362-390,393-404,420-423`（旧正文，P1-4）
- `CC/WEV3/review_v3_final.md`（旧两层，仅参考）

---

## 8. 最终裁决与条件

- **主方案：approve。** RGB/NIR `0.60/0.20/0.10` + TIR `0.80/0.10/0.05` 的结构、数据流、梯度流、bank/update/DDP/AMP/warm start 没有新的 P0；V1/V2/UAD 锚点在正确聚合下成立。
- **可以进入实现，条件是：**
  1. **P0-1** 7 处分派（6 gpd_type + 诊断）全部接 `'tpd'`；V3_design §9.3 的“6 处”必须改为 7；smoke 断言。
  2. **P0-2** per-query/per-modality 聚合；`w_cross[m]` per gallery；不写死 sum=1；V3_design §9.2 的标量×length-3 伪代码必须替换。
  3. **P1-4** V3_design 正文更新到主方案/W+G/6+1；否则实现者可能复制旧 TIR 0.70/旧 Gate。
  4. **P1-3** cross 的 per-column fallback 必须独立于 `L_own` 的 P_scene，保持 V1/V2 锚点；2337 missing 记录。
  5. **P1-2** TPD 诊断按 per-modality 加权梯度算份额；forward 之前；不调 GPD_HPD。
  6. **P1-1/P1-5/P1-6** 用户明确接受 gate 修正（第一轮 screening 放宽 AG/R5/GD/same；补 NIR→TIR；near-threshold 2 seed；0.898/+0.27/+0.45；control 标注/改 TIR 1.0/0.10/0.05）。
- **不承诺** paper+1；第一轮是“继承正面 + 弱指标净改善”的可证伪测试。若 W 的 cross/TIR 不达标，说明 cross 机制本身不够；若 G 的 same/AG/R5 回退，第二轮先调 own（RGB/NIR 0.60→0.65 或 TIR 0.80→0.85），单变量；不要同时降 cross。
- **AG** 没有平台项，TPD 权重不能改善 AG；若 AG 是必须改善项，第二轮加平台机制，不在第一轮硬调。
