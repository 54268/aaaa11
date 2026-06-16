# Wisig 未知类细分结果

当前协议：unknown cache 细分聚类（支持 KMeans / Agglomerative / GMM 后端；GMM-full-direct 直接由 GMM 输出候选标签，再通过 GMM 后验置信度和不稳定小簇规则标记不确定样本）。

| 指标键 | 中文说明 | 数值 |
| --- | --- | ---: |
| method | 聚类协议 | unknown_only_gmm_full_direct |
| clustering_backend | 聚类后端 | gmm_full_direct |
| feature_mode | 聚类特征 | embedding_stats |
| use_known_prototype_anchors | 是否启用已知原型锚点 | False |
| resolved_num_clusters | 自动确定的未知细分类数 | 12 |
| target_num_clusters | 协议目标未知细分类数 | 12 |
| fit_num_clusters | 实际拟合候选细分类数 | 12 |
| overcluster_extra_clusters | 冗余候选细分类数 | 0 |
| overcluster_extra_candidates | 参与自动选择的冗余候选列表 | [0] |
| auto_selected_overcluster_extra_clusters | 自动选择的冗余候选数 | 0 |
| m_selection_mode | m 选择模式 | unsupervised |
| m_selection_min_quality_gain | 增加冗余分量所需最小质量增益 | 0.010000 |
| m_selection_score | m 无监督诊断评分 | 1.310311 |
| m_selection_offline_quality | 离线细分质量均值 | 0.998878 |
| m_selection_offline_adjusted_quality | 覆盖率修正后的离线细分质量 | 0.998878 |
| direct_confidence_quantile | GMM低置信过滤分位数 | 0.000000 |
| direct_min_cluster_size | GMM不稳定小簇最小样本数 | 0 |
| selected_unknown_cache_size | 进入 unknown cache 的样本数 | 9634 |
| uncertain_size | 未分配/不确定样本数 | 0 |
| uncertain_ratio | 未分配/不确定样本比例 | 0.000000 |
| cluster_size_min | 最小细分类样本数 | 798 |
| cluster_size_max | 最大细分类样本数 | 807 |
| cluster_size_mean | 平均细分类样本数 | 802.833333 |
| nearest_known_proto_distance_mean | 到最近已知原型的平均距离 | 0.377671 |
| nearest_known_proto_distance_min | 到最近已知原型的最小距离 | 0.040442 |
| nmi | 归一化互信息，越高表示聚类与真实未知类越一致 | 0.998125 |
| ari | 调整兰德指数，越高表示聚类与真实未知类越一致 | 0.998637 |
| purity | 纯度，每个聚类中主导真实类的占比 | 0.999375 |
| hungarian_accuracy | 匈牙利匹配后的聚类准确率 | 0.999375 |
| unknown_cache_precision | unknown cache 中真实未知样本占比 | 0.996471 |
| unknown_cache_recall | 真实未知样本进入 unknown cache 的比例 | 1.000000 |
| coverage_of_total_test_unknown | 完成细分的真实未知样本覆盖率 | 1.000000 |

真实未知标签只用于离线 NMI、ARI、纯度、匈牙利准确率和混淆分析，不参与训练或在线判别。
