# 开放集 SEI 项目说明

项目已经整理成少量入口加函数目录的结构，不再使用 YAML 和命令行参数。

更完整的方案背景、数据流和方法细节见 `项目详细说明.md`。

## 主要入口

- `run_oracle.py`：Oracle 拒识主入口，默认连带做未知类细分。
- `run_wisig.py`：WiSig 拒识主入口，默认连带做未知类细分。
- `run_oracle_subdivision.py`：基于已有 Oracle 闭集模型，单独刷新未知类细分。
- `run_wisig_subdivision.py`：基于已有 WiSig 闭集模型，单独刷新未知类细分。
- `explore_subdivision.py`：在 Oracle 或 WiSig 上批量对比 feature_mode + 聚类后端 + target_k 组合，结果写入 `_explore_subdivision_<dataset>.json`。

入口文件顶部放常用开关和常改参数，直接改变量即可。完整默认参数放在 `settings/`，函数实现放在 `functions/`。

## 目录说明

- `functions/`：数据处理、模型、拒识方法、细分方法、通用工具函数。
- `settings/`：不常改的完整默认参数。
- `data/`：数据集和预处理结果。
- `outputs/`：正式实验输出。
- `ablations/`：消融实验记录和历史结果。
- `third_party/`：外部依赖代码。

## 常用参数

- `RUN_DATA_PREPARE`：是否重新预处理数据。数据已经处理好时保持 `False`。
- `RUN_TRAINING`：是否重新训练闭集模型。平时调拒识和细分时保持 `False`，避免覆盖已有权重。
- `RUN_UNKNOWN_SUBDIVISION`：是否在拒识结束后继续做未知类细分。
- `CHECKPOINT_PATH`：指定已有权重路径；为 `None` 时使用对应输出目录里的 `best_closed_set.pt`。
- `EPOCHS`：训练轮数。
- `LEARNING_RATE`：学习率。
- `BATCH_SIZE`：批大小。
- `FUSION_LAMBDA_GRID`：OpenMax 分数和原型距离分数的融合权重。
- `THRESHOLD_GRID`：拒识阈值搜索范围。
- `MIN_KNOWN_ACCURACY`：阈值搜索时要求的已知类准确率下限。
- `SUBDIVISION_FEATURE_MODE`：未知类细分特征，可选 `score_distance` / `embedding` / `embedding_distance` / `prototype_residual` / `residual_distance`。
- `SUBDIVISION_PCA_DIM`：细分前的 PCA 维度。
- `SUBDIVISION_K_MIN` / `SUBDIVISION_K_MAX`：未知细分类数搜索范围；二者相同表示固定类别数。
- `SUBDIVISION_CLUSTERING_BACKEND`：细分后端，可选 `kmeans`（球面 KMeans）、`agglomerative_cosine`（余弦凝聚式）、`gmm` / `gmm_full`（高斯混合）、`gmm_full_direct`（GMM-full 直接出标签，不做原型再分配）。
- `SUBDIVISION_TARGET_NUM_CLUSTERS`：若设置则提示自动 K 搜索向该值靠拢；`SUBDIVISION_TARGET_K_STRENGTH=1.0` 时直接锁定到此值。

## 输出文件

每次实验输出在 `outputs/<实验名>/`：

- `final_report.md`：中文结果报告。
- `open_set_metrics.json`：开放集拒识完整指标。
- `open_set_predictions.csv`：逐样本预测结果。
- `figures/`：ROC、PR、混淆矩阵、分数直方图等图表。
- `unknown_subdivision/`：未知类细分结果，包含 `unknown_subdivision_report.md`、`unknown_subdivision_metrics.json`、`unknown_subdivision_labels.npy`、`unknown_subdivision_assignments.csv`、`unknown_subdivision_centers.npy`、`true_unknown_confusion.csv`、`k_search_history.json`。

## 指标说明

- `overall_accuracy`：总体准确率，所有已知和未知样本一起统计。
- `known_accuracy`：已知类准确率，只看真实已知类样本是否被正确分到对应已知类别。
- `unknown_precision`：未知类精确率，被拒识为未知的样本中真实未知样本的比例。
- `unknown_recall`：未知类召回率，真实未知样本被拒识出来的比例。
- `known_fpr_as_unknown`：已知类误拒率，真实已知样本被误拒为未知的比例。
- `unknown_false_accept_rate`：未知类误接收率，真实未知样本被误分到已知类别的比例。
- `macro_f1`：宏平均 F1，各类别 F1 的平均值，更关注类别均衡。
- `auroc`：已知/未知区分能力，越高越好。
- `fpr95`：未知召回约 95% 时的已知类误拒率，越低越好。
- `oscr`：开放集分类-拒识综合曲线面积，越高越好。
- `nmi`：归一化互信息，衡量未知细分聚类和真实未知类别的一致性。
- `ari`：调整兰德指数，衡量聚类划分和真实类别的一致性。
- `purity`：聚类纯度，每个簇中主导真实类别所占比例。
- `hungarian_accuracy`：匈牙利匹配后的聚类准确率。
