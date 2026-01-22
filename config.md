# FunRec Notebook 配置说明（本地）

这份说明记录了本地运行 FunRec notebooks 的完整流程，方便下次快速恢复环境。

## 1) 目录结构

- 项目根目录：`/Users/wangjunfei/Desktop/fun-rec`
- Notebook：`notebooks/fun-rec/`
- 原始数据：`data/dataset/`
- 处理后数据：`tmp/`

## 2) .env（数据路径）

在项目根目录执行：

```bash
cp .env.example .env
cat > .env <<'EOF'
FUNREC_RAW_DATA_PATH=/Users/wangjunfei/Desktop/fun-rec/data/dataset
FUNREC_PROCESSED_DATA_PATH=/Users/wangjunfei/Desktop/fun-rec/tmp
EOF
```

验证：

```bash
python -c "import os; from funrec.utils import load_env_with_fallback; load_env_with_fallback(); print('RAW:', os.getenv('FUNREC_RAW_DATA_PATH')); print('PROCESSED:', os.getenv('FUNREC_PROCESSED_DATA_PATH'))"
```

## 3) Jupyter + Kernel

确认已进入 conda 环境：

```bash
conda activate funrec
which python
python -V
```

在该环境内安装 kernel 与 notebook 依赖：

```bash
python -m pip install jupyter ipykernel nbclassic notebook
python -m ipykernel install --user --name funrec --display-name "funrec (py3.10)"
```

## 4) 启动 Notebook（经典界面）

使用 nbclassic 可以避免空白页：

```bash
cd /Users/wangjunfei/Desktop/fun-rec
python -m nbclassic --NotebookApp.use_redirect_file=False
```

打开终端里带 token 的链接，例如：

```
http://127.0.0.1:8888/tree?token=...
```

## 5) 选择 Kernel

打开任意 `.ipynb` 后：

- 菜单 `Kernel` -> `Change Kernel` -> `funrec (py3.10)`

## 6) 跑 ItemCF（真实数据）

打开 `notebooks/fun-rec/chapter_1_retrieval/1.cf/1.itemcf.ipynb`，
在文末插入新的代码单元并运行：

```python
from funrec import run_experiment
run_experiment("item_cf")
```

第一次运行会做数据预处理并写入 `tmp/`。

## 常见问题

- 浏览器空白页：
  - 停止服务（`Ctrl+C`），重新用经典界面启动：
    ```bash
    python -m nbclassic --NotebookApp.use_redirect_file=False
    ```
- 找不到 kernel：
  - 重新安装 kernel：
    ```bash
    python -m ipykernel install --user --name funrec --display-name "funrec (py3.10)"
    ```
- 数据路径报错：
  - 检查 `.env`，确保 `data/dataset/` 下有 `ml-latest-small`、`ml-1m`、`kuairand`、`e_commerce` 等目录。
