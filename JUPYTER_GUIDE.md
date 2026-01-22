# 📚 Jupyter Notebook 快速入门指南

## 🚀 启动 Jupyter Notebook

### 方法1️⃣：使用启动脚本（推荐）
```bash
cd /Users/wangjunfei/Desktop/fun-rec
./start_jupyter.sh
```

### 方法2️⃣：手动启动
```bash
cd /Users/wangjunfei/Desktop/fun-rec
jupyter notebook
```

或使用经典界面：
```bash
jupyter nbclassic --NotebookApp.use_redirect_file=False
```

---

## 📖 学习路径

### 新闻推荐系统 - 6步完整流程

1. **Step 1: 赛题理解** (`1.understanding.ipynb`)
   - ⏱️ 预计时间：10分钟
   - 📝 类型：理论阅读
   - ✅ 操作：无需运行代码，阅读理解即可

2. **Step 2: Baseline构建** (`2.baseline.ipynb`)
   - ⏱️ 预计时间：15分钟
   - 📝 类型：代码实践
   - ✅ 操作：按顺序运行所有 Cell

3. **Step 3: 数据分析** (`3.analysis.ipynb`)
   - ⏱️ 预计时间：20分钟
   - 📝 类型：数据可视化
   - ✅ 操作：运行代码，观察图表

4. **Step 4: 召回策略** (`4.recall.ipynb`)
   - ⏱️ 预计时间：30分钟
   - 📝 类型：多路召回实现
   - ✅ 操作：理解不同召回算法

5. **Step 5: 特征工程** (`5.feature_engineering.ipynb`)
   - ⏱️ 预计时间：25分钟
   - 📝 类型：特征构造
   - ✅ 操作：学习特征设计思路

6. **Step 6: 排序模型** (`6.ranking.ipynb`)
   - ⏱️ 预计时间：30分钟
   - 📝 类型：模型训练
   - ✅ 操作：训练LightGBM和DIN模型

---

## ⌨️ Jupyter Notebook 快捷键

### 命令模式（按 `Esc` 进入）
| 快捷键 | 功能 |
|--------|------|
| `Enter` | 进入编辑模式 |
| `Shift + Enter` | 运行当前 Cell，移到下一个 |
| `Ctrl + Enter` | 运行当前 Cell，停留在当前 |
| `A` | 在上方插入 Cell |
| `B` | 在下方插入 Cell |
| `DD` | 删除当前 Cell |
| `M` | 切换为 Markdown Cell |
| `Y` | 切换为 Code Cell |
| `Z` | 撤销删除 |
| `L` | 显示/隐藏行号 |

### 编辑模式（按 `Enter` 进入）
| 快捷键 | 功能 |
|--------|------|
| `Esc` | 退出编辑模式 |
| `Cmd/Ctrl + /` | 注释/取消注释 |
| `Tab` | 代码补全 |
| `Shift + Tab` | 查看函数帮助 |

### 通用快捷键
| 快捷键 | 功能 |
|--------|------|
| `Cmd/Ctrl + S` | 保存 Notebook |
| `Cmd/Ctrl + Shift + P` | 打开命令面板 |

---

## 🛠️ 常见问题解决

### 问题1：端口已被占用
```bash
# 查找占用端口的进程
lsof -ti:8888

# 杀死进程
kill -9 $(lsof -ti:8888)

# 重新启动
./start_jupyter.sh
```

### 问题2：Kernel 连接失败
**解决方案：**
1. 菜单栏：`Kernel` → `Restart Kernel`
2. 如果还不行：关闭 Notebook，重启 Jupyter 服务器

### 问题3：代码运行报错
**常见原因：**
- ❌ 没有按顺序运行 Cell
- ❌ 缺少依赖包
- ❌ 数据路径不对

**解决方案：**
1. 从头开始运行：`Kernel` → `Restart & Run All`
2. 检查 `.env` 文件配置
3. 查看错误信息，搜索解决方案

### 问题4：浏览器没有自动打开
**解决方案：**
1. 手动复制终端中显示的 URL
2. 粘贴到浏览器打开
3. URL 格式：`http://localhost:8888/?token=xxxxx`

---

## 💡 使用技巧

### 1. 查看变量
```python
# 在 Cell 中直接输入变量名
df

# 或者使用 print
print(df.head())
```

### 2. 显示所有输出
```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

### 3. 显示进度条
```python
from tqdm import tqdm
for i in tqdm(range(100)):
    # your code
    pass
```

### 4. 魔法命令
```python
# 显示代码运行时间
%time your_function()

# 显示详细的运行时间
%%timeit
your_code()

# 显示当前目录
%pwd

# 列出当前目录文件
%ls
```

---

## 📝 最佳实践

### ✅ 推荐做法
1. **按顺序运行**：从第一个 Cell 开始，依次执行
2. **经常保存**：`Cmd/Ctrl + S` 定期保存
3. **重启测试**：完成后用 `Restart & Run All` 验证
4. **添加注释**：修改代码时添加注释说明
5. **实验备份**：重要修改前复制一份 Notebook

### ❌ 避免做法
1. 不要跳着运行 Cell
2. 不要在多个 Notebook 中同时运行耗时代码
3. 不要在运行中修改正在执行的 Cell
4. 不要关闭浏览器就认为 Jupyter 已停止

---

## 🔧 配置与环境

### 检查环境变量
```python
import os
print(os.getenv('FUNREC_RAW_DATA_PATH'))
print(os.getenv('FUNREC_PROCESSED_DATA_PATH'))
```

应该输出：
```
/Users/wangjunfei/Desktop/fun-rec/data/dataset
/Users/wangjunfei/Desktop/fun-rec/tmp
```

### 检查依赖包
```python
import pandas as pd
import numpy as np
import lightgbm as lgb
print("✅ 所有依赖包已安装")
```

---

## 📞 获取帮助

### 在 Notebook 中查看帮助
```python
# 查看函数文档
?pd.DataFrame

# 查看详细文档
??pd.DataFrame

# 查看对象的所有方法
dir(pd.DataFrame)
```

### 在线资源
- [Jupyter 官方文档](https://jupyter-notebook.readthedocs.io/)
- [Pandas 文档](https://pandas.pydata.org/docs/)
- [FunRec 项目 GitHub](https://github.com/datawhalechina/fun-rec)

---

## 🎯 下一步

现在你已经准备好了！

1. ✅ 启动 Jupyter Notebook
2. ✅ 打开 `1.understanding.ipynb`
3. ✅ 开始学习新闻推荐系统

**祝学习愉快！** 🚀
