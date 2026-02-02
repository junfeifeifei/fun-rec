# 新闻推荐系统增强实现计划（Plus）

## 概述

为你的新闻推荐系统项目添加 6 个新的 Jupyter Notebook，覆盖面试高频的深度学习推荐算法知识点，并把工程形态做得更接近真实工业推荐。

**实现位置（已落地）：**

- 增强版项目代码全部放在：`notebooks/fun-rec/chapter_5_projects/news_recommendation_system_plus/`
- 默认复用基础版产物目录：`tmp/projects/news_recommendation_system/`
- 新增训练/评估/部署产物统一写到：`tmp/projects/news_recommendation_system/artifacts/`

## 新增文件

| 序号 | 文件名 | 主题 | 核心模型 |
|------|--------|------|----------|
| 7 | `7.two_tower_recall.ipynb` | 双塔模型召回 | DSSM（in-batch negatives / InfoNCE） |
| 8 | `8.deep_ranking.ipynb` | 深度排序模型 | DeepFM, DIN |
| 9 | `9.sequence_modeling.ipynb` | 序列推荐 | SASRec, DIEN |
| 10 | `10.multi_task.ipynb` | 多任务学习 | MMoE, ESMM, PLE |
| 11 | `11.evaluation.ipynb` | 综合评估体系 | GAUC, NDCG, 多样性指标 |
| 12 | `12.model_serving.ipynb` | 模型部署 | ONNX, FAISS服务化 |

## 详细内容

### Notebook 7: 双塔模型召回 (`7.two_tower_recall.ipynb`)

**核心知识点：**
- 双塔架构原理（用户塔 + 物品塔）
- 负采样策略（随机、热门惩罚、困难负样本）
- 训练过程（Sampled Softmax Loss / BPR Loss）
- FAISS 向量索引构建与检索
- 在线/离线一致性

**主要章节：**
1. 双塔模型原理介绍
2. 数据准备与负采样实现
3. 特征工程（用户侧/物品侧特征）
4. DSSM 模型构建与训练
5. 物品向量离线计算
6. FAISS 索引构建（Flat / IVF）
7. 召回效果评估（对比 ItemCF）
8. 面试要点总结

**产出文件（默认路径）：**
- `tmp/projects/news_recommendation_system/artifacts/two_tower/dssm_inbatch/`
  - `two_tower_model.keras` / `user_tower.keras` / `item_tower.keras`
  - `faiss_index.bin` / `item_embeddings.npy`
  - `id_maps.pkl`（user/item/cat 的编码器）
  - `metrics.pkl`（离线 HR/NDCG 等）
- `tmp/projects/news_recommendation_system/recall_candidates_two_tower.pkl`（召回候选集）

---

### Notebook 8: 深度排序模型 (`8.deep_ranking.ipynb`)

**核心知识点：**
- FM 特征交叉原理
- DeepFM = FM + DNN
- DIN 注意力机制（Attention Pooling）
- 序列特征处理

**主要章节：**
1. CTR 预估任务定义
2. 排序样本构建（利用召回结果）
3. DeepFM 模型实现与训练
4. DIN 模型实现（带序列特征）
5. 注意力权重可视化
6. 模型对比评估（LGBMRanker vs DeepFM vs DIN）
7. 面试要点总结

**产出文件：**
- `tmp/projects/news_recommendation_system/artifacts/ranking/deep_models/`
  - `deepfm.keras` / `din.keras`
  - `scaler.pkl`（dense 标准化）
  - `deepfm_factorizers.pkl`（稀疏特征编码表 + 列定义）
  - `din_encoders.pkl`（raw→enc 的 user/item 编码表）

---

### Notebook 9: 序列建模 (`9.sequence_modeling.ipynb`)

**核心知识点：**
- 序列推荐动机（用户兴趣演化）
- Self-Attention 序列建模（SASRec）
- 兴趣抽取与演化（DIEN）
- GRU with Attention Unit (AUGRU)

**主要章节：**
1. 序列推荐问题定义
2. 序列样本构建（Next-Item Prediction）
3. SASRec 模型（Transformer-based）
4. DIEN 模型（Interest Extractor + Evolution）
5. 序列长度与效率分析
6. 评估：HR@K, NDCG@K
7. 面试要点总结

**产出文件：**
- `tmp/projects/news_recommendation_system/artifacts/sequence/sasrec_inbatch/`
  - `sasrec_inbatch.keras` / `user_tower.keras` / `item_tower.keras`
  - `faiss_index.bin` / `item_embeddings.npy` / `item_id_map.pkl`
- `tmp/projects/news_recommendation_system/recall_candidates_sasrec.pkl`
- `tmp/projects/news_recommendation_system/artifacts/ranking/dien_like/dien_like.keras`

---

### Notebook 10: 多任务学习 (`10.multi_task.ipynb`)

**核心知识点：**
- 多任务学习动机（数据稀疏、任务关联）
- 样本空间选择偏差（SSB）
- MMoE Gate 机制
- ESMM 解决 CVR 估计偏差
- PLE 渐进式分层提取

**主要章节：**
1. 多任务学习动机与场景
2. 多目标标签构造（点击 + 阅读深度）
3. Shared-Bottom 基线
4. MMoE 模型实现
5. ESMM 模型实现
6. PLE 模型实现
7. 多任务评估
8. 面试要点总结

**产出文件：**
- `tmp/projects/news_recommendation_system/artifacts/multitask/`
  - `shared_bottom.keras` / `mmoe.keras` / `esmm.keras` / `ple.keras`
  - `scaler.pkl` / `preprocess.pkl`
  - `multi_task_report.csv`

---

### Notebook 11: 综合评估体系 (`11.evaluation.ipynb`)

**核心知识点：**
- 离线评估 vs 在线评估
- AUC vs GAUC（分组 AUC）
- 排序指标：Precision, Recall, NDCG, MAP, MRR
- 多样性指标：Coverage, ILD, Category Entropy
- 新颖性指标

**主要章节：**
1. 评估指标体系概述
2. 精度指标实现（Precision, Recall, HR, NDCG, MAP）
3. AUC 与 GAUC 实现
4. 多样性指标（覆盖率、列表内多样性）
5. 新颖性指标
6. 综合评估报告生成
7. 模型对比可视化（雷达图）
8. 面试要点总结

**产出文件：**
- `tmp/projects/news_recommendation_system/artifacts/evaluation/`
  - `recall_report.csv` / `ranking_report.csv` / `multitask_report.csv`

---

### Notebook 12: 模型部署 (`12.model_serving.ipynb`)

**核心知识点：**
- 模型格式转换（TensorFlow → ONNX）
- ONNX Runtime 推理
- 双塔模型的向量服务架构
- 在线/离线一致性验证
- 推理性能优化（批处理、量化）

**主要章节：**
1. 模型部署概述
2. ONNX 导出（排序模型）
3. ONNX Runtime 推理测试
4. 双塔模型向量服务设计
5. 在线/离线一致性验证
6. 推理服务架构设计图
7. 性能优化技巧（批处理、模型量化）
8. 面试要点总结

**产出文件：**
- `tmp/projects/news_recommendation_system/artifacts/serving/`
  - `*_savedmodel/`（SavedModel 导出）
  - `*.onnx`（可选：依赖 tf2onnx 时才会导出）

---

## README 更新

更新项目 README，添加：
1. 进阶内容章节（Plus：Notebooks 7-12）
2. 新增中间产物列表与产物路径
3. 技术栈更新（TensorFlow, FAISS；ONNX 可选）
4. 面试知识点索引（双塔/序列/多任务/评估/部署）

---

## 实现顺序

```
Phase 1: Notebook 7 (双塔召回) - 召回训练 + 向量检索
    ↓
Phase 2: Notebook 8 (深度排序) - 依赖召回结果
    ↓
Phase 3: Notebook 9 (序列建模) - 可并行开发
    ↓
Phase 4: Notebook 10 (多任务) - 可并行开发
    ↓
Phase 5: Notebook 11 (评估) - 需要所有模型产出
    ↓
Phase 6: Notebook 12 (部署) - 需要训练好的模型
    ↓
最后: 更新 README.md / Plus README
```

---

## 技术依赖

- **TensorFlow 2.x**：深度学习框架
- **FAISS**：向量检索
- **LightGBM**：排序基线
- **ONNX / ONNX Runtime（可选）**：部署格式（本仓库默认 requirements 未强依赖）
- **现有数据**：复用 `train_hist.pkl`, `recall_candidates.pkl` 等

> 说明：为了方便复现与阅读，Plus 版本 notebooks 采用“代码内联”的形式（每个 notebook 都是完整模块 + 笔记），不强依赖 `src/` 下的可复用包。

---

## 预期成果

完成后，项目将覆盖以下面试高频知识点：

| 知识点 | 对应 Notebook |
|--------|---------------|
| 双塔模型/DSSM | 7 |
| 负采样策略 | 7 |
| FM/DeepFM 特征交叉 | 8 |
| DIN 注意力机制 | 8 |
| Transformer 序列推荐 | 9 |
| DIEN 兴趣演化 | 9 |
| 多任务学习 MMoE | 10 |
| ESMM CVR 预估 | 10 |
| GAUC 评估 | 11 |
| NDCG/Coverage/Diversity | 11 |
| ONNX 模型部署 | 12 |
| 向量检索服务 | 12 |
