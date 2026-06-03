# 开放集 SEI 项目说明

项目已经整理成少量入口加函数目录的结构，不再使用 YAML 和命令行参数。

更完整的方案背景、数据流和方法细节见 `项目详细说明.md`。当前 WiSig 与 Oracle 的拒识、细分结果汇总见 `当前方法结果汇总.md`。

## 主要入口

- `run_oracle.py`：Oracle 拒识主入口，默认连带做未知类细分。
- `run_wisig.py`：WiSig 拒识主入口，默认连带做未知类细分。
- `run_oracle_subdivision.py`：基于已有 Oracle 闭集模型，刷新开放集拒识中间结果后单独刷新未知类细分。
- `run_wisig_subdivision.py`：基于已有 WiSig 闭集模型，刷新开放集拒识中间结果后单独刷新未知类细分。

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
- `RUN_UNKNOWN_SUBDIVISION`：是否在拒识结束后继续做未知类细分；为 `True` 时运行主入口就会自动完成细分，通常不需要再单独运行 `run_oracle_subdivision.py` 或 `run_wisig_subdivision.py`。
- `CHECKPOINT_PATH`：指定已有权重路径；为 `None` 时使用对应输出目录里的 `best_closed_set.pt`。
- `EPOCHS`：训练轮数。
- `LEARNING_RATE`：学习率。
- `BATCH_SIZE`：批大小。
- `FUSION_LAMBDA_GRID`：OpenMax 分数和原型距离分数的融合权重。
- `THRESHOLD_GRID`：拒识阈值搜索范围。
- `MANUAL_GLOBAL_THRESHOLD`：手动全局拒识阈值；WiSig 当前使用 `0.94` 来提高未知类召回率。
- `MIN_KNOWN_ACCURACY`：阈值搜索时要求的已知类准确率下限。
- Oracle 当前在 `settings/oracle_settings.py` 中启用了 `score_calibration = "classwise_z"`，用于按已知预测类别校准 unknown score。
- Oracle 当前在 `settings/oracle_settings.py` 中启用了 `known_rescue`，用于把少量距离已知原型很近、原型间隔较大的误拒样本救回已知类。
- `SUBDIVISION_FEATURE_MODE`：未知类细分特征，可选 `embedding` / `embedding_stats` / `embedding_distance` / `embedding_score_distance` / `prototype_residual` / `residual_distance` / `score_distance`。
- `SUBDIVISION_PCA_DIM`：细分前的 PCA 维度。
- `SUBDIVISION_K_MIN` / `SUBDIVISION_K_MAX`：未知细分类数搜索范围；二者相同表示固定类别数。
- `SUBDIVISION_CLUSTERING_BACKEND`：细分后端，可选 `kmeans`（球面 KMeans）、`agglomerative_cosine`（余弦凝聚式）、`gmm` / `gmm_full`（高斯混合）、`gmm_full_direct`（GMM-full 直接出标签，不做原型再分配）。
- `SUBDIVISION_TARGET_NUM_CLUSTERS`：若设置则提示自动 K 搜索向该值靠拢；`SUBDIVISION_TARGET_K_STRENGTH=1.0` 时直接锁定到此值。
- `SUBDIVISION_OVERCLUSTER_EXTRA`：细分阶段的冗余候选分量数，记为 `m`。实际拟合候选数为 `K + m`，其中 `K` 是协议给定的真实未知类数量；当前在 `m ∈ {0, 1, 2}` 的小范围内固定主实验设置。Oracle 当前为 `m=2`，表示先拟合 8 个候选分量，最终只保留 6 个稳定未知细分类；WiSig 当前为 `m=0`，直接使用协议目标 12 类，避免把干净未知类拆成类内子模态。
- `SUBDIVISION_DIRECT_CONFIDENCE_QUANTILE`：`gmm_full_direct` 下按 GMM 后验置信度过滤低置信样本的分位数。
- `SUBDIVISION_DIRECT_MIN_CLUSTER_SIZE`：`gmm_full_direct` 下候选簇低于该样本数时视为不稳定小簇，标为不确定样本。

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
- `score_calibration_mode`：unknown score 的校准方式；`classwise_z` 表示按已知预测类别做 z-score 校准。
- `known_rescue_enabled`：是否启用已知类救回规则，用于降低已知类误拒。
- `nmi`：归一化互信息，衡量未知细分聚类和真实未知类别的一致性。
- `ari`：调整兰德指数，衡量聚类划分和真实类别的一致性。
- `purity`：聚类纯度，每个簇中主导真实类别所占比例。
- `hungarian_accuracy`：匈牙利匹配后的聚类准确率。
- `target_num_clusters`：协议目标未知细分类数，例如 Oracle 为 6，WiSig 为 12。
- `fit_num_clusters`：实际拟合的候选簇数；Oracle 当前先拟合 8 个候选簇，再剔除低置信或不稳定小簇；WiSig 当前直接拟合 12 个候选簇。
- `resolved_num_clusters`：剔除不确定样本后实际保留的有效细分类数。
- `uncertain_size`：未强行分入细分类的样本数，包含低置信、小簇或疑似已知污染样本。
- `coverage_of_total_test_unknown`：真实未知测试样本中最终参与细分评估的比例。
