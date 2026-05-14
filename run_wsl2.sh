#!/bin/bash
# WSL2 一键运行脚本
# 用法：在 WSL2 Ubuntu 里运行 bash run_wsl2.sh [part1|part2|all]
#
# 前提：已按 README 完成 uv + pytorch3d 安装，
#       虚拟环境默认位于 ~/lab6_env（可通过 LAB6_VENV 环境变量覆盖）

VENV=${LAB6_VENV:-"$HOME/lab6_env"}
PY="$VENV/bin/python"

# 切换到脚本所在目录（即项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || { echo "[ERROR] 无法进入项目目录 $SCRIPT_DIR"; exit 1; }

echo "[INFO] 项目目录: $SCRIPT_DIR"
echo "[INFO] Python:   $PY"

# 验证环境
"$PY" -c "import pytorch3d, torch; print('[OK] pytorch3d', pytorch3d.__version__, '| torch', torch.__version__)" 2>&1 || {
    echo ""
    echo "[ERROR] pytorch3d 未安装，请先完成环境配置（见 README 方法二）。"
    exit 1
}

MODE=${1:-all}

case $MODE in
    part1)
        echo ""
        echo "=== Part 1：剪影优化（CPU 约 30-60 分钟）==="
        "$PY" main_silhouette.py
        ;;
    part2)
        echo ""
        echo "=== Part 2：联合纹理优化（CPU 约 60-90 分钟）==="
        "$PY" main_texture.py
        ;;
    all|*)
        echo ""
        echo "=== Part 1：剪影优化 ==="
        "$PY" main_silhouette.py
        echo ""
        echo "=== Part 2：联合纹理优化 ==="
        "$PY" main_texture.py
        ;;
esac

echo ""
echo "=== 输出文件 ==="
echo "Part 1: $(ls output/silhouette/*.png 2>/dev/null | wc -l) 帧, GIF: $(ls output/silhouette/*.gif 2>/dev/null | head -1)"
echo "Part 2: $(ls output/texture/*.png 2>/dev/null | wc -l) 帧, GIF: $(ls output/texture/*.gif 2>/dev/null | head -1)"
