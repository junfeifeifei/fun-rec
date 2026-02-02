# 新闻推荐系统增强版（Chapter 5 Project Plus）

这个文件夹是 `news_recommendation_system/` 的**增强版项目入口**，用于把项目补齐成更贴近真实工业推荐（实习/面试更有竞争力）的形态：

- **双塔召回训练**（in-batch negatives / 对比学习）+ **FAISS 向量检索**
- **深度排序**（DeepFM / DIN 等）对比传统 `LGBMRanker`
- **序列建模**（SASRec / DIEN 等）覆盖“兴趣演化”
- **多任务学习**（MMoE / ESMM / PLE 等）覆盖多目标/去偏
- **统一评估**（Recall/Ranking/多样性等）+ 复现实验报告
- **导出与部署**（ONNX/向量服务化思路）

## 运行依赖

1. 根目录 `.env` 配置（建议直接按 `config.md` 设置），确保：
   - `FUNREC_RAW_DATA_PATH` 指向 `data/dataset`
   - `FUNREC_PROCESSED_DATA_PATH` 指向 `tmp`
2. 先跑基础版（生成中间产物）：
   - `notebooks/fun-rec/chapter_5_projects/news_recommendation_system/1-6`
3. 再跑增强版（本文件夹内的 7-12）。

## Notebooks（7-12）

- 7. `7.two_tower_recall.ipynb`：双塔召回训练（InfoNCE / in-batch negatives）+ FAISS
- 8. `8.deep_ranking.ipynb`：DeepFM / DIN 排序（与 LGBMRanker 对比）
- 9. `9.sequence_modeling.ipynb`：SASRec 召回 + DIEN-like 排序
- 10. `10.multi_task.ipynb`：SharedBottom / MMoE / ESMM / PLE 多任务
- 11. `11.evaluation.ipynb`：统一评估（Recall/Ranking/多样性/GAUC）
- 12. `12.model_serving.ipynb`：本地推理链路 + 模型导出（SavedModel；ONNX 可选）

## 说明

- 该增强版默认复用基础版产物目录：`tmp/projects/news_recommendation_system/`
- 新增产物会统一写到：`tmp/projects/news_recommendation_system/artifacts/`
- ONNX 导出是可选项：本仓库默认 `requirements.txt` 未强依赖 `tf2onnx/onnxruntime`

## 关于 `src/funrec/projects/newsrec`

你在仓库里看到的 `src/funrec/projects/newsrec/` 是一个“可复用的 Python 包”雏形，用来把数据准备、编码、训练、召回等逻辑做成可脚本化/可复用模块（更像真实工程）。

但为了符合你希望的形态（**notebook 内包含完整模块代码 + 笔记**），Plus 版本的 7-12 选择了“代码内联”，不强依赖该包；两者会共享同一份数据与产物目录。
