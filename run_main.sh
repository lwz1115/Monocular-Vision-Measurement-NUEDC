#!/bin/bash
cd ~/Desktop/202508
source ~/.bashrc
export OPENBLAS_CORETYPE=ARMV8
# 确保使用系统 Python3 或你终端里用的 Python 路径
python3 main.py
# 保持终端打开，查看输出
read -p "程序结束，按 Enter 关闭..."
