# 新闻推荐系统项目（Chapter 5）

本项目基于新闻点击数据集，实现了一个完整的推荐系统流程，涵盖数据理解、召回、特征工程到排序模型的全链路。

## 项目概览

```
数据理解 → 离线划分 → 数据分析 → 多路召回 → 特征工程 → 排序模型
```

## 增强版（Plus）

如果你希望把项目做得更贴近工业推荐（更适合找推荐算法实习/面试），请继续看：

- `notebooks/fun-rec/chapter_5_projects/news_recommendation_system_plus/`：Notebooks 7-12
  - 双塔召回训练 + FAISS（Two-Tower / DSSM in-batch）
  - 深度排序（DeepFM / DIN）
  - 序列建模（SASRec / DIEN-like）
  - 多任务学习（MMoE / ESMM / PLE）
  - 统一评估与报告
  - 本地推理链路与导出（Serving Demo）

## 数据集

| 数据文件 | 规模 | 说明 |
|---------|------|------|
| train_click_log.csv | 111万条 | 训练集用户点击日志 |
| testA_click_log.csv | 51万条 | 测试集用户点击日志 |
| articles.csv | 36万篇 | 文章元信息（类别、创建时间、字数） |
| articles_emb.csv | 36万篇 × 250维 | 文章预训练向量 |

用户规模：20万用户，人均点击约 5.6 次

## 流程详解

### 1. 数据理解 (`1.understanding.ipynb`)

- 加载并预览数据集结构
- 统计用户与物品规模
- 分析用户点击分布（长尾分布，中位数 3 次）
- 可视化热门类别分布

### 2. 离线划分与基线 (`2.baseline.ipynb`)

- **离线划分策略**：每个用户的最后一次点击作为验证目标，其余作为历史
- **热门召回基线**：全局热门物品召回
- **基线效果**：Hit Rate@20 ≈ 14.5%

### 3. 数据分析 (`3.analysis.ipynb`)

- 点击行为与文章内容关联分析
- 文章新鲜度分析：75% 点击发生在文章发布后 17 小时内
- 类别点击分布：Top 类别（375、281、250）占据主要点击量
- 用户活跃度 ECDF 分析

### 4. 多路召回 (`4.recall.ipynb`)

实现三种召回策略并进行融合：

| 召回策略 | 方法 | Hit Rate@50 |
|---------|------|-------------|
| 热门召回 | 全局点击热度排序 | 25.8% |
| ItemCF | 基于物品的协同过滤（IUF 加权） | **42.6%** |
| Embedding | FAISS 向量相似度检索 | 2.7% |
| 融合召回 | 加权倒数排名融合 | 38.9% |

- ItemCF 使用 IUF（Inverse User Frequency）加权
- Embedding 召回基于用户最后点击物品检索相似文章
- 融合权重：ItemCF=1.0, Embedding=0.8, 热门=0.2

### 5. 特征工程 (`5.feature_engineering.ipynb`)

构建四类特征用于排序模型：

**用户特征**
- `user_click_count`：历史点击次数
- `user_unique_items`：历史点击去重物品数
- `user_last_click_ts`：最后点击时间戳
- `user_top_category`：用户偏好类别（众数）

**物品特征**
- `item_click_count`：物品被点击次数
- `item_last_click_ts`：物品最后被点击时间
- `category_id`：文章类别
- `words_count`：文章字数
- `created_at_ts`：文章创建时间

**交互特征**
- `is_same_category`：候选物品是否与用户偏好类别相同
- `item_age_hours`：物品年龄（小时）
- `time_gap_hours`：距离物品上次被点击的时间差
- `emb_sim_last`：候选物品与用户最后点击物品的向量相似度

**召回特征**
- `recall_score`：召回融合分数
- `recall_rank`：召回排名

### 6. 排序模型 (`6.ranking.ipynb`)

- **模型**：LGBMRanker（LambdaRank 目标函数）
- **训练集划分**：80% 用户训练，20% 用户验证
- **评估指标**：Hit Rate@5 ≈ **32.9%**
- **可选**：测试集推断与提交文件生成

## 目录结构

```
├── index.ipynb                 # 项目概览与目录
├── 1.understanding.ipynb       # 数据理解与基础统计
├── 2.baseline.ipynb            # 离线划分 + 热门召回基线
├── 3.analysis.ipynb            # 行为与内容数据分析
├── 4.recall.ipynb              # 多路召回（热门/ItemCF/向量）
├── 5.feature_engineering.ipynb # 特征构建与训练样本生成
├── 6.ranking.ipynb             # LGBMRanker 排序与离线评估
└── README.md                   # 项目说明
```

## 中间产物

运行过程中生成的文件保存在 `tmp/projects/news_recommendation_system/`：

| 文件 | 说明 |
|------|------|
| train_hist.pkl | 训练历史点击数据 |
| valid_last.pkl | 验证集（用户最后一次点击） |
| user_hist.pkl | 用户历史点击序列 |
| popular_items.pkl | 热门物品列表 |
| itemcf_i2i.pkl | ItemCF 相似度矩阵 |
| recall_candidates.pkl | 召回候选集 |
| rank_train.pkl | 排序训练样本（含特征和标签） |
| user_features.pkl | 用户特征表 |
| item_features.pkl | 物品特征表 |
| lgb_ranker.txt | LGBMRanker 模型文件 |
| feature_cols.pkl | 特征列名列表 |

## 运行说明

1. **环境配置**：设置 `FUNREC_RAW_DATA_PATH` 环境变量指向数据目录
2. **运行顺序**：按编号 1 → 6 依次运行
3. **调试模式**：`2.baseline.ipynb` 和 `4.recall.ipynb` 提供 `DEBUG` 开关，可快速调试
4. **推断模式**：`6.ranking.ipynb` 中设置 `RUN_INFERENCE=True` 可生成测试集推荐结果

## 技术栈

- **数据处理**：Pandas, NumPy
- **可视化**：Matplotlib, Seaborn
- **向量检索**：FAISS
- **排序模型**：LightGBM (LGBMRanker)

## 效果总结

| 阶段 | 指标 | 效果 |
|------|------|------|
| 热门召回基线 | Hit Rate@20 | 14.5% |
| ItemCF 召回 | Hit Rate@50 | 42.6% |
| 多路融合召回 | Hit Rate@50 | 38.9% |
| LGBMRanker 排序 | Hit Rate@5 | **32.9%** |
