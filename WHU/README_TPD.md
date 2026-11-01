# WEV3 / TPD：三层原型判别

代码基线：`WEV2/WHU`（HPD）。保留 `P_id` 全局原型、`P_scene` 模态子中心、warm start、count EMA、DDP、诊断；新增 `GPD_TPD`。

## 损失

对 query 模态 m：

\[
L_m = w_{own}(m)\mathrm{CE}(\max_k u^\top P_{scene}[c,k]/\tau)
+ w_{global}(m)\mathrm{CE}(u^\top P_{id}[c]/\tau)
+ w_{cross}(m)\sum_{m'\ne m}\mathrm{CE}(u^\top P_{scene}^{cross}[c,m']/\tau)
\]

`L_GPD = (1/M) \sum_m L_m`；`w_cross(m)` 是每个 gallery 模态 m' 的权重，总 cross = (M-1)·w_cross(m)。

第一轮权重：

| query | w_own | w_global | w_cross（每个 m'） | 合计 |
|---|---:|---:|---:|---:|
| RGB | 0.60 | 0.20 | 0.10 | 1.00 |
| NIR | 0.60 | 0.20 | 0.10 | 1.00 |
| TIR | 0.80 | 0.10 | 0.05 | 1.00 |

## 关键实现

- `GPD_TPD`：per-modality CE 均值再 `/M`；`w_cross[m]` 是 per-gallery；不写死 `sum(w)=1`。
- `P_scene_cross` 是独立 clone；单列缺失用 `P_id` 补齐，不改变 `L_own/L_global` 的 `P_scene`。
- 训练总损失仍是 `CE+WRT+0.01·ProCA+1.0·L_GPD`；`L_GPD` 权重和 1.0。
- 7 处分派（6 处 `gpd_type` + 1 处诊断）全部接 `'tpd'`。
- 诊断 `diagnose_gpd_tpd_gradients`：per-term CE、per-modality 加权梯度、own_ratio/usage、missing/zero cross、P_id/P_scene 几何、pre-clip/clip；在训练 forward 之前。
- 推理不变：pre-BN CLS + L2 + 欧氏；不加载 bank。

## 运行

4 卡 DDP（每卡 64 样本 × 3 模态 = 192 图，global batch 256）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train.py \
  --config_file configs/TPD_UAD.yml \
  MODEL.DIST_TRAIN True \
  DATASETS.ROOT_DIR /project/code/data \
  SOLVER.IMS_PER_BATCH 256 \
  SOLVER.BASE_LR 0.032 \
  OUTPUT_DIR ./logs/TPD_1000_DDP
```

- `--root_dir` 不是 YACS key；数据根目录用 `DATASETS.ROOT_DIR`。
- `MODEL.PRETRAIN_PATH` 默认 `jx_vit_base_p16_224-80ecf9dd.pth`（相对工作目录）；若文件不在工作目录，追加 `MODEL.PRETRAIN_PATH /path/to/jx_vit_base_p16_224-80ecf9dd.pth`。
- 单卡：`python train.py --config_file configs/TPD_UAD.yml DATASETS.ROOT_DIR /project/code/data SOLVER.IMS_PER_BATCH 64 SOLVER.BASE_LR 0.008`。
- `DATALOADER.NUM_WORKERS` 已在配置中为 8；需要覆盖时用 `DATALOADER.NUM_WORKERS 8`。

测试：使用 `transformer_120.pth`，`TEST.NECK_FEAT='before'`、`TEST.FEAT_NORM='yes'`；1000/GD/AG/detailed-domain 按现有脚本。

## 第一轮 gate（W + G screening）

- W：overall/cross > V2；TIR→TIR ≥ V2，TIR 四向（含 NIR→TIR）≥ V2。
- G screening：same ≥ V1−0.5；AG/R5/R10/GD ≥ V2−0.5。
- 第二轮 strict：same ≥ V1−0.3；AG/R5/R10/GD ≥ V2−0.2；先调 own（RGB/NIR 0.60→0.65 或 TIR 0.80→0.85）。
- paper+1 是 Tier2。

## 注意

- 第一轮是 TPD 机制/Pareto 测试，不承诺 all-metric +1；AG 没有平台项，只做护栏；平台机制第二轮。
- 2337 等 1000 W/G 之后；第一轮不跑。
- `GPD_SCENE_WEIGHT` 不属于 TPD；HPD 老路径保留但 TPD 不读。
