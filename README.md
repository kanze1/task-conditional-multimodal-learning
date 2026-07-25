# 多模态学习中的任务条件信息价值与结构表征

**Task-Conditional Information Value in Multimodal Representation Learning**

本项目研究一个可证伪的问题：

> 多模态训练相对单模态训练的表征收益，是否由附加模态对目标任务的条件信息价值决定？

项目首先通过可控潜在场景图，分别操纵 `redundant`、`complementary`、`irrelevant` 和 `conflict` 四类信息条件，考察冻结表示中的结构可解码性、组合迁移和模态扰动鲁棒性。

## 当前状态

**两周 Go/No-Go 立项与协议冻结阶段。**

- 尚无正式实验结果；
- 尚未提出新 loss 或新 fusion architecture；
- GCL 不再作为中心贡献或默认主方法；
- 在预注册阈值通过前，不进入完整论文写作或自然数据大实验。

## 权威文档

- [研究计划](新研究计划_多模态条件信息价值_20260726.md)
- [研究需求](docs/requirements/新研究需求_多模态条件信息价值_20260726.md)
- [旧研究缺陷与迁移边界](docs/defects/旧研究缺陷与归档说明_20260726.md)
- [Go/No-Go 冻结实验协议草案](docs/protocol/go_no_go_冻结实验协议_20260726.md)
- [项目身份决策](docs/decisions/0001_项目身份与仓库边界.md)

## 第一阶段资源边界

- 单张 RTX 4090；
- 最多 48 GPU 小时；
- 两种轻量架构；
- 三个 seed 的最小主矩阵；
- 数据生成和评估可由 CPU 复核。

## 证据状态规则

`smoke`、`pilot` 与 `formal` 必须明确区分。任何主结论都必须能追溯到具体结果文件、完整运行配置、配置哈希、数据 manifest 和 seed。缺少冻结协议、生成器审计、两架构三 seed 主矩阵、公平预算核对、预注册裁决或缺陷记录中的任一项，项目状态只能标记为“实验进行中”。
