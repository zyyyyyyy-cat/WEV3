# WEV3 / TPD 代码审查（review_v3_tpd_code）

> 审查对象：`CC/WEV3/WHU`（代码基线 `CC/WEV2/WHU`/HPD）。
> 对照：`CC/WEV1/WHU`（GPD-SC）、`CC/WHU`（UAD）、WHU 论文 Eq.3–10 / Sec.4.2–4.5。
> 权威方案：`CC/WEV3/review_v3_tpd_final.md` §0.1/§0.2/§2.2/§4/§5.2/§6/§7 + `review_v3_tpd_agent_final.md`；`V3_design.md` 正文有历史残留，以 README/审查为准。
> 审查方式：源码走查 + `compileall` + CPU 单测（`%TEMP%\tpd_review_test.py`）+ YACS 解析测试 + 2-rank gloo 尝试。未修改项目代码；未跑 GPU/120ep。

## 0. 裁决

**approve with conditions**。**P0 = 0，P1 = 1**（修复前；README_TPD 的运行命令不可执行/不启用 DDP，见 §3 P1-1）。**复核后 P1-1 已关闭，当前 P0 = 0 / P1 = 0（见 §0.1）**。核心机制（逐 query/per-modality 聚合、per-gallery cross、独立 cross fallback、7 处分派、TPD 诊断、bank/DDP/AMP、推理不变）与设计一致，CPU 单测的 V1/V2/UAD 锚点、zero-column、诊断梯度分解、per-modality 归因全部通过。

进入 1ep smoke 的条件（最小）：
1. 先修 P1-1：把 README_TPD §运行 的命令改成 `torchrun ... MODEL.DIST_TRAIN True` + 预训练路径（唯一修法见 §3）。
2. 按 §4 的 1ep GPU/DDP smoke 跑一次：重点看 warm start、bank 跨 rank 一致、TPD diag、AMP 无 NaN、推理 pre-BN。
3. 读诊断份额时注意 P2-1：`share_*` 是“逐项梯度范数之和”，`g_*` 是“加权项梯度之和的范数”；两者口径不同，不能直接和 V2 的单组件 share 对比（训练本身不受影响）。

[代码事实] 源码 diff（V2→V3）只改了 `processor/processor.py`、`config/defaults.py`、新增 `configs/TPD_UAD.yml`/`README_TPD.md`；`model/`、`loss/`、`datasets/`、`utils/`、`solver/` 源文件未变。默认 `single/subcenter/hierarchical` 路径仅把 `'tpd'` 加进元组/分支，原分支逻辑不变（warm start 日志文案由 `HPD` 改为 `HPD/TPD` 属文案）。

## 0.1 修复复核（coordinator 修复后，2026-10-31）

- **P1-1 已关闭**：`README_TPD.md:38-51` 已改为 `torchrun --standalone --nproc_per_node=4` + `MODEL.DIST_TRAIN True`；opts 中的 `MODEL.DIST_TRAIN / DATASETS.ROOT_DIR / SOLVER.IMS_PER_BATCH / SOLVER.BASE_LR / OUTPUT_DIR` 均为有效 YACS key，按该命令的 opts 实测 `cfg.merge_from_list` 通过（`DDP=True, root=/project/code/data, batch=256, lr=0.032, out=./logs/TPD_1000_DDP`）；argparse 实测 `--config_file` 取值、其余 token 全部进入 `opts`。单卡说明与 `MODEL.PRETRAIN_PATH`/`DATALOADER.NUM_WORKERS` 覆盖说明正确（单卡 opts 实测 `batch=64, lr=0.008`）。
- **P2-5 的 §9.4 已关闭**：`V3_design.md:335` 已改为 `[0.6, 0.6, 0.8]`；global/cross 为 `[0.2,0.2,0.1]/[0.1,0.1,0.05]`，与 `configs/TPD_UAD.yml` 一致。（`WEV3/README.md:3` 的“代码尚未创建”仍是旧状态文案，P2 不阻塞。）
- **P2-1 文档已补**：`processor.py:678-683` docstring 已说明 `share_*`（逐项梯度范数之和）与 `g_*`（加权项梯度之和的范数）口径不同；代码未变，训练不受影响。
- **未引入新问题**：`compileall` exit 0；README 命令的 YACS/argparse 模拟通过；V3_design §9.4 与 YAML 一致。

## 1. 代码事实核对表

| # | 事实 | 证据 |
|---|---|---|
| 1 | `train_collate_fn` 把每个模态 stack 成 `imgs[m]`，`pids` 为 `[B]`；所有模态 B 相同 | `datasets/make_dataloader.py:17-23` |
| 2 | `make_model.forward(mode=0)` 用 `torch.cat(list(x),0)`，`global_feat` 为 modality-major `[RGB B; NIR B; TIR B]`；`target.repeat(M)` 同序 | `model/make_model.py:67-80`；`processor.py:485, 525-535` |
| 3 | `PKMSampler`/`PKMSampler_DDP` 按 P×K（K=NUM_INSTANCE=4）组织，每组 K 个同 pid；batch B 是该 rank 的样本数（每模态相同，64） | `datasets/sampler.py:105-140`；`datasets/sampler_ddp.py:254-287`；`datasets/make_dataloader.py:66-98` |
| 4 | `PCA` 返回 `ids = target.reshape(-1, group_size)[:,0]`、`g_cent [P,D]`、`centers [P,D] × M`；`g_cent` detach | `processor.py:13-32` |
| 5 | `GPD_TPD` 检查 `M == bank.K`、`feat.shape[0] % M == 0`、`feat.shape[0] == M*B` | `processor.py:455-468` |
| 6 | 权重转 float32；校验长度=M、非负、每 query `own+global+(M-1)*cross > 0`；**不写死 sum(w)=1** | `processor.py:470-483` |
| 7 | `P_id/P_scene` 在 `no_grad` 下 clone + `.float()` + row-level zero fallback（与 V1/V2 同） | `processor.py:488-502` |
| 8 | `P_cross = P_scene.clone()`；单列缺失（norm<1e-6）用 `P_id` 补齐；`missing_cross`/`zero_cross_after` 只读统计；`L_own/L_global` 的 `P_scene` 不被改 | `processor.py:504-514` |
| 9 | `logits_id [M·B,C]`、`logits_scene/logits_cross [M·B,C,K]`；`L_own = logits_scene.max(dim=2)`；cross 用 `logits_cross[sl,:,mp]` | `processor.py:516-519, 525-535` |
| 10 | 每 query 模态 m 的 own/global/6 个 cross 分别 CE；`loss += (w_own[m]·own + w_global[m]·global + w_cross[m]·Σ_{m'≠m}cross)/M`；`w_cross[m]` 是 per-gallery | `processor.py:521-540` |
| 11 | `GPD_TPD` 返回 `loss, own_losses, global_losses, cross_losses, diag`；`cross_losses[m][m]=None` | `processor.py:523-540, 581` |
| 12 | 训练分派：允许类型、K==M 守卫、bank 创建、warm start、loss、update、epoch counts/geometry、epoch log 全部接 `'tpd'`（7 处） | `processor.py:901-909, 913-933, 952-985, 1018-1090, 1124-1151, 1189-1213, 1251-1266, 1284-1339` |
| 13 | `'tpd'` → `HierarchicalPrototypeBank`；warm start → `warm_start_hierarchical`（rank0 + broadcast P_id/P_scene/counts） | `processor.py:922-933, 952-985`；bank 定义 `processor.py:165-230` |
| 14 | bank update 在 `scaler.step`/`optimizer.step` 之后；传 `ids, g_cent, ids.repeat(M), mod_ids, centers_all` | `processor.py:1189-1213`；`HierarchicalPrototypeBank.update` `processor.py:216-225` |
| 15 | TPD meters：own/global 为 per-modality CE 均值，cross 为 6 个 pair CE 均值；epoch 日志标签 TPD | `processor.py:881-883, 997-999, 1139-1151, 1251-1266` |
| 16 | 诊断新函数 `diagnose_gpd_tpd_gradients`：训练 forward 之前、rank0、unwrapped、eval；不调用 `GPD_HPD`、不读 `GPD_SCENE_WEIGHT` | `processor.py:674-827, 1018-1051` |
| 17 | 诊断 per-modality 加权梯度：`(w_own[m]/M)·own_m` 等逐项 autograd；合并得 g_own/g_global/g_cross；per-modality share 列表；cos；loss 项；missing/zero；id/scene stats | `processor.py:716-732, 760-825` |
| 18 | 诊断在 `finally` 里 `model.train(was_training)`；用 `autograd.grad` 不写 `.grad`；不改 bank | `processor.py:682-683, 826-827` |
| 19 | 训练总损失：`loss_fn(CE+WRT) + 0.01·L_PCA + 1.0·L_GPD`；`L_GPD` 主方案每 query 合计 1.0，control 1.40/1.20 不被断言拦截 | `processor.py:1174-1177`；`configs/TPD_UAD.yml:65` |
| 20 | 配置：defaults 加 3 个 TPD list key；YAML 主方案 `0.6/0.6/0.8`、`0.2/0.2/0.1`、`0.1/0.1/0.05`；`GPD_DIAG_PERIOD=100`；输出 `./logs/TPD_1000_DDP` | `config/defaults.py:66-71`；`configs/TPD_UAD.yml:19-30, 73` |
| 21 | `GPD_SCENE_WEIGHT` 只在 HPD 路径（`GPD_HPD`/HPD 诊断/ε 日志）使用，TPD 分支不读 | `processor.py:607, 638, 1117, 1245` |
| 22 | 推理：`do_inference` 未改；`NECK_FEAT='before'` 返回 pre-BN CLS；evaluator `FEAT_NORM=yes` 后 L2 + 欧氏；bank 不入 `state_dict` | `processor.py:1402-1453`；`make_model.py:74-80`；`utils/metrics.py:462-486`；`processor.py:1355-1361` |
| 23 | 论文边界：GPD 是单 identity prototype + 全局 CE（Eq.7–9），总目标 Eq.10；TPD 的 P_scene/max_k 与 gallery cross 是扩展，不是论文原式 | `WHU.txt:530-584` |

## 2. 逐项审查

### 2.1 输入/输出、形状、切片、标签

[代码事实] 数据流与 V1/V2 相同：`imgs` 是 M 个 `[B,C,H,W]`，模型 cat 成 `[M*B,...]`，因此 `global_feat [M*B,D]` 是 modality-major；`target [B]` 经 `repeat(M)` 成 `[M*B]`，与分段一致。`GPD_TPD` 的 `B=target.shape[0]`、`sl=slice(m*B,(m+1)*B)`、`t=target_rep[sl]`、`logits_cross[sl,:,mp]` 全部对齐。`ids [P]` 与 `g_cent/centers [P,D]` 对齐；`bank.K=M=3` 有显式检查。

[代码事实] 当前 PKM sampler 保证每个 rank 的 B 是 `NUM_INSTANCE` 的整数倍、每 K 个样本同 pid；因此 `target.reshape(-1,group_size)` 与 `B` 假设成立。若以后改 sampler/最后 batch 不齐，`B` 假设会失效（V1/V2 已有同类假设），但第一轮 1000 配方不变。

### 2.2 损失聚合与锚点

[代码事实 + CPU 单测] `GPD_TPD` 用 per-modality CE 再 `/M`，与“每 query 模态权重、按 query 展开”在 B 相同时数值等价。CPU 单测（`%TEMP%\tpd_review_test.py`）：
- 主权重与手工 `reduction='none'` + `repeat_interleave` 的 per-query 形式相等（PASS）；
- control 权重 `1.0/1.0/1.0, 0.2/0.2/0.1, 0.1/0.1/0.05`（每 query 1.40/1.20，总和>1）正常计算且与手工式相等（PASS），说明没有写死 `sum(w)=1`；
- `w_cross[m]` 是 per-gallery：`w_cross[0]` 的贡献恰为 `(CE_{0→1}+CE_{0→2})/M`，且改 `w_cross[0]` 只改变 RGB query 行的梯度（单测：RGB 0.0761，NIR/TIR 0.0）；
- 改 `w_own[2]`（TIR）只改变 TIR query 行梯度（单测：TIR 0.1305，RGB/NIR 0.0）。

[代码事实 + CPU 单测] 锚点（同一 bank、同一输入）：
| 配置 | 对照 | 结果 |
|---|---|---|
| `w_own=1, w_global=0, w_cross=0` | V1 `GPD_SC` | loss 差 **0.0**；feat 梯度差 **0.0** |
| `w_cross=0, own=.75, global=.25` | V2 `GPD_HPD(alpha=.75)` | loss 差 **0.0** |
| `w_global=1, own=cross=0`（zero bank，warmstart=False） | UAD `GPD` | loss 差 **1.9e-6**（float32 求和顺序；目标/结构等价） |

[机制假设] `w_global=1` 要与 UAD 原版“逐位”一致，必须 `GPD_WARMSTART=False`；主方案 `WARMSTART=True`，P_id 是 warm-started id_mean，不是 UAD 零初始化。文档已注明。

[代码事实] 权重和/非负/长度校验在 `GPD_TPD` 内（`processor.py:470-483`）；config 不校验，但训练时首次调用即校验。NaN 权重不会被 `(w<0)`/`(sum<=0)` 拦截（NaN 比较为 False），属配置卫生问题（P2），不是主路径。

### 2.3 cross fallback

[代码事实 + CPU 单测] `P_cross` 是 `P_scene` 在 no_grad 下的独立 clone；`missing_cross = P_cross.norm(dim=2) < 1e-6`；仅对单列缺失用 `P_id` 补齐；`L_own/L_global` 的 `P_scene` 未变。单测：
- 人为把某类某列置零：`missing_cross=1`、`zero_cross_after=0`；`GPD_TPD` 与手工 per-column P_id fallback 数值相等；
- 同一 bank 下 `L_own` 仍精确等于 `GPD_SC`，`cross=0, own=.75/global=.25` 仍精确等于 `GPD_HPD`；
- `GPD_TPD` 不修改 bank buffers；bank 无 `.grad`、无 NaN。

[代码事实/残余] 无 warm start 且 `P_id` 为零的非 batch 类，`P_cross` 该列仍为零（死列）；`zero_cross_after` 会计数，CE 仍有限（零 logit 作负类）。主方案 warm start 会把所有类的所有模态填成 `id_mean`，因此主 run 的 missing/zero 应为 0；2337 长尾/无 warmstart 时看这两个计数。另：`missing_cross` 统计的是**全 C 类**，不是本 batch 的 P 个 id；无 warmstart 首次迭代会包含全部未见类，这是诊断口径，不是错误。

### 2.4 7 处分派

[代码事实] V2 的 7 个决策点全部接 `'tpd'`：
1. 允许类型 + `LOSS_TYPE` 守卫：`processor.py:901-909`；
2. `GPD_SUB_CENTERS==len(MODALITIES)` 守卫：`processor.py:913-921`；
3. bank 创建 → `HierarchicalPrototypeBank`：`processor.py:922-933`；
4. warm start → `warm_start_hierarchical` + `proto_bank.broadcast`：`processor.py:952-985`；
5. 诊断 → `diagnose_gpd_tpd_gradients`（不是 `GPD_HPD`）：`processor.py:1018-1090`；
6. loss → `GPD_TPD`：`processor.py:1124-1151`；
7. bank update + epoch counts/geometry/日志标签：`processor.py:1189-1213, 1251-1266, 1284-1339`。

[代码事实] 没有静默落回 HPD/single 的分支；`'tpd'` 的 bank 类型、loss、update、诊断、日志均为 TPD 路径。`GPD_TYPE='tpd'` 但 `LOSS_TYPE` 不含 `gpd` 会抛错。

### 2.5 诊断

[代码事实 + CPU 单测] `diagnose_gpd_tpd_gradients`：
- 在训练 forward 之前、`optimizer.zero_grad()` 之后调用（`processor.py:1018-1051`），用 `θ_t/B_t`；
- `model.eval()` + `torch.enable_grad()` + autocast；last ViT block + final norm 参数；rank0；`finally` 恢复 train；
- 不调用 `GPD_HPD`（monkeypatch 测试 PASS）、不读 `GPD_SCENE_WEIGHT`（改 999.0 结果不变 PASS）；
- per-term CE：own/global/6 cross；per-modality 加权梯度（权重乘在项内）；g_own/g_global/g_cross；per-modality share；两两 cos；own_ratio/usage；missing/zero；id/scene stats；g_cent_norm；
- 单测：`autograd.grad(总 GPD loss)` 与诊断 3M 个加权项梯度之和的最大差 **9.5e-7**（/M 与训练损失一致）；keys 完整；shares 有限；模型留在 train；不写 `.grad`；不改 bank；
- `gpd_diag` 在 loss 分支每 iter 赋值，epoch 日志的 `if gpd_diag else` 分支是防御性的；`loss_gpd` 在诊断里未用不影响任何输出。

[代码事实/残余] `share_*` 的分母是 `eff_own=Σ_m||g_own_m||`、`eff_global=Σ_m||g_global_m||`、`eff_cross=Σ_m||g_cross_m||`（逐项范数之和），而同一个日志里的 `g_own/g_global/g_cross` 是“加权项梯度之和的范数”。两者口径不同；当各模态梯度不同向时，`share_*` 会相对 `share_base` 偏大。随机小模型上实测 `eff/g` 比约 1.55–1.63（base share 3.6% vs 用 norm-of-sum 重算约 5.5%）。这是诊断口径问题，不影响训练；见 P2-1。诊断也不返回 pre-clip/clip 与 P_id/P_scene 几何：pre-clip/clip 由训练循环的 LOG_PERIOD 日志给，几何由 epoch 末日志给（README_TPD:31 的措辞略宽）。

### 2.6 训练循环、bank、DDP、AMP

[代码事实] TPD meters 定义/重置/更新完整；loss unpack 与函数返回一致；`meter_ls[3]` 收 `loss_gpd`；总损失 `+ L_GPD`（权重 1.0）。bank update 在 `scaler.step(optimizer)` 之后、`no_grad` 下；`ids.repeat(M)`、`mod_ids=arange(M).repeat_interleave(P)`、`centers_all=cat(centers)` 与 V2/HPD 完全同序（单测验证了 modality-major 对齐与 count/EMA）。warm start rank0 + broadcast、RNG 保存/恢复未改。AMP：`GPD_TPD` 对 feat/P_id/P_scene/g_cent 显式 `.float()`；CE 在 fp32 计算。DDP：bank 不走 DDP wrapper，每 iter 用 all_gather 同步；`ddp=cfg.MODEL.DIST_TRAIN`；warm start 后 broadcast。诊断用 unwrapped `model_meta`，rank0 额外 eval forward，不触发 DDP collective；其他 rank 会在随后的 DDP forward/backward 等待 rank0，属基线已有的停顿，不是死锁。

[未执行] 本机 Windows CPU 的 torch 2.8 gloo `makeDeviceForHostname(): unsupported gloo device`，无法实跑 2-rank all-gather；DDP 路径与 V2 完全相同，且 V2 有 4-GPU 日志；仍需在 GPU/Linux 上按 §4 做 2/4-rank bank 一致性 smoke。

### 2.7 配置、默认路径、README

[代码事实] `config/defaults.py:66-71` 新 key 为 list，默认主方案；YACS 解析实测：`GPD_TPD_OWN/GLOBAL/CROSS = [0.6,0.6,0.8]/[0.2,0.2,0.1]/[0.1,0.1,0.05]`；`GPD_TYPE='tpd'`、`GPD_SUB_CENTERS=3`、`GPD_DIAG_PERIOD=100`、`NECK_FEAT='before'`、`OUTPUT_DIR='./logs/TPD_1000_DDP'`。`GPD_SCENE_WEIGHT` 不参与 TPD（grep 仅 HPD 路径）。

[代码事实] `TPD_UAD.yml` 的注释写“4-GPU DDP, per-GPU 192 images, global batch 256, LR 0.032”，与 V2 配方一致；但 YAML 本身不设 `MODEL.DIST_TRAIN`（V2 同样靠 CLI 覆盖）。**修复前** `README_TPD.md:36-41` 的运行命令用 `python train.py`、`--root_dir`、`DALOADER.NUM_WORKERS`：前两者不是 `train.py` 的 argparse/YAML key，`cfg.merge_from_list` 会 `AssertionError: Non-existent key: --root_dir`；即使删掉，也没有 `torchrun`/`DIST_TRAIN True`/预训练路径覆盖，实际会变成单卡 batch 256、LR 0.032，不是注释/README 的 4 卡 DDP。详见 P1-1；**复核后已改为 `README_TPD.md:38-51` 的 torchrun 命令并验证通过（§0.1）**。

### 2.8 推理

[代码事实] `do_inference` 与 V2 逐行一致；`NECK_FEAT='before'` 返回 pre-BN CLS，`FEAT_NORM=yes` 后 evaluator L2 归一化并算欧氏距离；checkpoint 只存 model，bank 不入 `state_dict`，推理不加载 bank。TPD 不改变模型结构/推理路径。

### 2.9 论文边界

[代码事实] 论文 Eq.7–9 是单个 identity prototype 的全局 CE，Eq.10 是 `L_base + λ_ProCA L_ProCA + λ_GPD L_GPD`（`WHU.txt:530-584`）。TPD 的 `P_scene[C,K,D]`、`max_k`、gallery cross、模态感知权重都是 WEV1/V2/V3 扩展；不能写成论文原式，但作为“UAD 全局锚点 + 模态子中心 + gallery cross”的扩展是成立的。本轮审查不因“不是论文原版”否定 TPD。

## 3. P0/P1 表

| 编号 | 级别 | 问题 | 证据 | 最小修法 | 为什么 |
|---|---|---|---|---|---|
| P0 | — | 未发现阻塞正确性/可运行性/机制的代码问题 | §1、§2 单测与走查 | — | — |
| **P1-1** | **P1** | `README_TPD.md` 的“运行”命令不可执行，且不启用 4-GPU DDP：`--root_dir`/`DALOADER.NUM_WORKERS` 不是 YACS key，会 AssertionError；`python train.py` + 无 `DIST_TRAIN True` 会单卡跑 global batch 256，且未覆盖相对预训练路径 | `README_TPD.md:36-41`；`train.py:75-88`（仅 `--config_file`、`opts`、`--local_rank`）；`configs/TPD_UAD.yml` 无 `DIST_TRAIN`；对照 `WEV2/WHU/README.md:4-11` | 把该段替换为 V2 风格命令：`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py --config_file configs/TPD_UAD.yml MODEL.DIST_TRAIN True MODEL.PRETRAIN_PATH /project/code/data/jx_vit_base_p16_224-80ecf9dd.pth DATASETS.ROOT_DIR /project/code/data SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032 OUTPUT_DIR logs/TPD_1000_DDP` | 按当前 README 操作会在参数解析阶段失败；即使绕过，跑出的也不是注释声明的 4 卡 DDP 配方，第一轮结果不可比/可能 OOM |

> 复核状态（2026-10-31）：P1-1 已关闭；`README_TPD.md:38-51` 的新命令已用 `cfg.merge_from_list`/argparse 模拟验证。当前无 P0/P1。

## 4. 1ep smoke 最小清单

- [ ] 用 P1-1 修正后的 `torchrun` 命令启动；`MAX_EPOCHS=1`、`CHECKPOINT_PERIOD=1`、`EVAL_PERIOD=1`、`LOG_PERIOD` 小、`GPD_DIAG_PERIOD` 设成能在 1ep 内触发的值（如 10）。
- [ ] 日志断言：`GPD type: tpd`；`HPD/TPD warm start: ... K=3`；TPD diag 行（loss own/global/cross、g own/global/cross、share、cos、missing/zero、per-modality shares）；`Epoch[1] TPD losses`；`TPD scene bank`；`TPD geometry`。不应出现 `GPD-HPD diag`/`HPD scene bank`（tpd 下）。
- [ ] 类型/分派断言：`type(proto_bank).__name__ == 'HierarchicalPrototypeBank'`；可用临时 monkeypatch 计数确认 `GPD_TPD` 被调用、`GPD_HPD` 不被调用（不改项目代码）。
- [ ] 聚合/锚点单测（可复用 `%TEMP%\tpd_review_test.py`）：V1/V2/UAD 锚点、per-query 形式、per-gallery cross、改 TIR weight 只影响 TIR 行、zero-column fallback 不改 L_own。
- [ ] DDP：2 或 4 rank 跑若干 iter，导出/比较各 rank 的 `id_bank.prototypes`、`scene_bank.sub_centers`、`scene_bank.counts`，断言 allclose；确认无 hang（诊断 extra forward 的 rank0 停顿可接受）。
- [ ] AMP：loss/三项 CE 有限，pre-clip grad norm 有限，clip_count 正常；无 NaN/inf。
- [ ] bank：warm start 后 counts=1；一次 update 后 counts=2（每个 (id,modality)）；P_id/P_scene 无 `.grad`、不在 `model.state_dict()`。
- [ ] 默认路径回归：同一 1ep/1-2 iter 用 `GPD_TYPE='single'` 和 `'subcenter'` 能跑通（loss/bank 类型正确），确认 `'tpd'` 的加入没有改变旧分支。
- [ ] 推理：用 1ep checkpoint（或已有 120ep）跑 `test.py`，`TEST.NECK_FEAT='before'`、`TEST.FEAT_NORM='yes'`；1000（可选 GD/AG/detailed-domain）能产出 mAP/CMC；确认 bank 不被加载。
- [ ] 日志卫生：TPD 日志不出现 `GPD_SCENE_WEIGHT`/`alpha`；`GPD_SCENE_WEIGHT` 只出现在 HPD 路径。

## 5. 残余风险（P2，不阻塞）

1. **P2-1 诊断份额口径**：`share_own/global/cross` 用“逐项梯度范数之和”，`g_own/global/cross` 用“逐项梯度之和的范数”；各模态项不同向时前者偏大（随机小模型 eff/g≈1.55–1.63），与 V2 单组件 share 不可直接比。若要用 share 判断“base vs GPD/global 独立份额”，建议同日志同时给 norm-of-sum 口径，或在报告里用 `g_*`/cos 重新核算；训练/门控数值不受影响。（复核：`processor.py:678-683` docstring 已补此口径说明，代码未变。）
2. **P2-2 `missing_cross` 统计范围**：统计全 C 类，不是本 batch 的 P 类；无 warmstart 首轮会包含全部未见类（`zero_cross_after` 也 >0）。主方案 warm start 下应为 0；2337/无 warmstart 时看这两个数，不要把 C·K 当作 batch 缺失。
3. **P2-3 M=1 边界**：TPD 的 `cross` 不存在，训练循环 `sum(tpd_cross_vals)/len(tpd_cross_vals)` 会 ZeroDivisionError，诊断 `sum(cross_losses[m][mp])` 为 float 也会让 `autograd.grad` 报错。WHU 是 M=3，不影响第一轮；若要防御，在 do_train 对 `tpd` 加 `M>=2` 守卫。
4. **P2-4 诊断字段/时序文档**：诊断返回 pre-clip/clip 与 P_id/P_scene 几何不在同一 dict；pre-clip/clip 在训练 LOG_PERIOD 日志，几何在 epoch 末日志。`README_TPD.md:31` 的“诊断 ... pre-clip/clip/几何”表述略宽，实际信号都在，只是不在同一行。
5. **P2-5 文档残留**：`V3_design.md:335` 的 §9.4 YAML 已改为 `GPD_TPD_OWN_WEIGHT: [0.6,0.6,0.8]`，与 `configs/TPD_UAD.yml` 一致（复核关闭）。`WEV3/README.md:3` 仍写“代码尚未创建”，仅是旧状态文案；实现/运行以 YAML/README_TPD 为准。
6. **P2-6 defaults 注释**：`GPD_SHARED_MOMENTUM`/`GPD_DIAG_PERIOD` 注释写“HPD only”，但 TPD 也使用；仅注释卫生。
7. **P2-7 2337 训练集缺失模态**：`PKMSampler` 对 `cur=0` 会 `divmod(need,0)`；TPD 的 per-column bank fallback 不能替代 sampler 的前置条件。2337 前必须先保证训练集每个 identity 三模态齐全（或改 sampler）。
8. **P2-8 诊断代理/性能**：诊断是 last-block+norm/eval/rank0 代理，不是全 ViT 梯度；TPD meters 每 iter 多做若干 `.item()` 同步（6 个 cross + own/global），比 HPD 略慢，不影响正确性。
9. **P2-9 bank 持久化**：bank 不在 `state_dict`，resume 不恢复 bank；与 V1/V2/UAD 相同，不是 TPD 回归。

## 6. 结论

- **机制**：TPD 的 `/M` per-modality 聚合、per-gallery cross、独立 `P_cross`/P_id 单列 fallback 实现正确；V1/V2/UAD 锚点数值成立（V1/V2 差 0.0，UAD 1.9e-6 为 float 顺序）；三项都是 pre-BN CLS 上的全局 CE，推理不变。
- **集成**：7 处分派全部接 `'tpd'`；bank/warm start/update/DDP/AMP/日志与 V2 一致；默认 `single/subcenter/hierarchical` 路径未变。
- **第一轮前提（复核后）**：P1-1 已关闭；按 §4 做 1ep smoke（尤其 DDP bank 一致性、AMP、推理）；读诊断份额时注意 P2-1。
- **唯一权威结论**：配置/权重以 `configs/TPD_UAD.yml` + `README_TPD.md` + `review_v3_tpd_final.md` 为准（RGB/NIR `0.60/0.20/0.10`，TIR `0.80/0.10/0.05`）；`V3_design.md` §8.2/§9.2 的旧 Gate/标量伪代码是历史残留；§9.4 已于复核中修正为 0.80/0.10/0.05。
