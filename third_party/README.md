# third_party

这里用于放可复用的成熟开源模块。当前保留：

- `pytorch-ood`
- `openmax`
- `torchcvnn`
- `prototypical-networks`

建议做法：
1. 外部仓库保留原目录，便于回溯来源。
2. 不要在训练脚本里直接调用第三方接口。
3. 通过 `sei_osr/` 下的本地 wrapper 暴露统一输入输出。
4. 你的创新模块仍然保留在本项目里自己实现，不被第三方替代。

这样后续替换 backbone、OpenMax 或数据集时，只需要改 wrapper，不会破坏主流程。
