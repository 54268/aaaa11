# Wisig 未知类细分结果

当前协议：unknown cache 细分聚类（支持 KMeans / Agglomerative / GMM 后端；GMM-full-direct 直接由 GMM 输出标签，可结合已知原型距离、GMM 后验置信度和不稳定小簇规则标记不确定样本）。

| 指标键 | 中文说明 | 数值 |
| --- | --- | ---: |
| method | 聚类协议 | prototype_guided_gmm_full_direct |
| clustering_backend | 聚类后端 | gmm_full_direct |
| feature_mode | 聚类特征 | embedding |
| resolved_num_clusters | 自动确定的未知细分类数 | 8 |
| target_num_clusters | 协议目标未知细分类数 | 12 |
| fit_num_clusters | 实际拟合候选细分类数 | 13 |
| direct_confidence_quantile | GMM低置信过滤分位数 | 0.400000 |
| direct_min_cluster_size | GMM不稳定小簇最小样本数 | 50 |
| selected_unknown_cache_size | 进入 unknown cache 的样本数 | 9634 |
| uncertain_size | 未分配/不确定样本数 | 5808 |
| uncertain_ratio | 未分配/不确定样本比例 | 0.602865 |
| cluster_size_min | 最小细分类样本数 | 206 |
| cluster_size_max | 最大细分类样本数 | 801 |
| cluster_size_mean | 平均细分类样本数 | 478.250000 |
| nearest_known_proto_distance_mean | 到最近已知原型的平均距离 | 0.476658 |
| nearest_known_proto_distance_min | 到最近已知原型的最小距离 | 0.325592 |
| nmi | 归一化互信息，越高表示聚类与真实未知类越一致 | 1.000000 |
| ari | 调整兰德指数，越高表示聚类与真实未知类越一致 | 1.000000 |
| purity | 纯度，每个聚类中主导真实类的占比 | 1.000000 |
| hungarian_accuracy | 匈牙利匹配后的聚类准确率 | 1.000000 |
| unknown_cache_precision | unknown cache 中真实未知样本占比 | 0.996471 |
| unknown_cache_recall | 真实未知样本进入 unknown cache 的比例 | 1.000000 |
| coverage_of_total_test_unknown | 完成细分的真实未知样本覆盖率 | 0.398438 |

真实未知标签只用于离线 NMI、ARI、纯度、匈牙利准确率和混淆分析，不参与训练或在线判别。
