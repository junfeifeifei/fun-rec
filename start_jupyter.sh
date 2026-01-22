#!/bin/bash

echo "=================================================="
echo "🚀 启动 FunRec 新闻推荐系统 Jupyter Notebook"
echo "=================================================="

# 进入项目目录
cd /Users/wangjunfei/Desktop/fun-rec

# 检查是否已有 Jupyter 在运行
if lsof -Pi :8888 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  检测到 Jupyter 已经在运行 (端口 8888)"
    echo "请在浏览器中访问: http://localhost:8888"
    echo ""
    echo "如果需要重启，请先关闭现有的 Jupyter 进程"
    echo "（在运行 Jupyter 的终端按 Ctrl+C）"
    exit 1
fi

echo ""
echo "📂 当前目录: $(pwd)"
echo ""
echo "📚 可用的 Notebooks:"
echo "  1. 赛题理解          - notebooks/fun-rec/chapter_5_projects/1.understanding.ipynb"
echo "  2. Baseline构建      - notebooks/fun-rec/chapter_5_projects/2.baseline.ipynb"
echo "  3. 数据分析          - notebooks/fun-rec/chapter_5_projects/3.analysis.ipynb"
echo "  4. 召回策略          - notebooks/fun-rec/chapter_5_projects/4.recall.ipynb"
echo "  5. 特征工程          - notebooks/fun-rec/chapter_5_projects/5.feature_engineering.ipynb"
echo "  6. 排序模型          - notebooks/fun-rec/chapter_5_projects/6.ranking.ipynb"
echo ""
echo "=================================================="
echo "🌐 正在启动 Jupyter Notebook..."
echo "=================================================="
echo ""
echo "💡 提示："
echo "  - 浏览器会自动打开"
echo "  - 按 Ctrl+C 停止服务器"
echo "  - 服务器地址: http://localhost:8888"
echo ""

# 启动 Jupyter Notebook
jupyter notebook
