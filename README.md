# 2025全国大学生电子设计大赛C题 - 视觉测量系统

![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.0+-lightgrey.svg)

## 项目简介

本项目是2025年全国大学生电子设计大赛C题的参赛作品，实现了一个基于计算机视觉的实时测量系统。该系统能够通过摄像头捕获图像，识别目标物体的形状和尺寸，并计算距离等参数。

![项目截图](images/screenshot.png)
*系统主界面*

## 功能模块

### 1. JC1 - 形状测量
- 识别并测量圆形、等边三角形、正方形的尺寸
- 计算目标距离
- 使用滑动窗口和众数滤波提高测量稳定性

![JC1功能演示](images/jc1_demo.png)
*JC1形状测量功能*

### 2. FH1 - 最小正方形测量
- 检测参考矩形内的正方形
- 计算最小正方形的边长
- 实时显示距离和边长数据

![FH1功能演示](images/fh1_demo.png)
*FH1最小正方形测量*

### 3. FH2 - 透视矫正测量
- 对参考矩形进行透视矫正
- 检测并测量最短延伸边
- 实时显示距离和边长数据

![FH2功能演示](images/fh2_demo.png)
*FH2透视矫正测量*

### 4. FH3A/FH3B - 数字识别与高亮
- 使用OCR技术识别数字
- 高亮显示目标数字对应的正方形
- 测量并显示目标正方形的边长

![FH3功能演示](images/fh3_demo.png)
*FH3数字识别与高亮*

### 5. FH4 - 面积比例测量
- 检测参考矩形内的正方形
- 计算正方形面积与参考矩形面积的比例
- 计算正方形的实际边长

![FH4功能演示](images/fh4_demo.png)
*FH4面积比例测量*

## 硬件要求

- 摄像头（支持USB或其他接口）
- 计算机（支持Python 3.6+）
- INA226电流传感器（用于功率监测）

![硬件连接图](images/hardware_connection.png)
*硬件连接示意图*

## 软件依赖

```
python>=3.6
opencv-python
numpy
PyQt5
ddddocr
```

## 安装与运行

1. 克隆项目到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行主程序：
   ```bash
   python main.py
   ```

## 使用说明

1. 启动程序后，会显示主界面，包含多个功能按钮
2. 点击相应的功能按钮（如JC1、FH1等）开始测量
3. 对于FH3A和FH3B功能，需要先选择目标数字（0-9）
4. 测量过程中，实时显示图像和测量结果
5. 点击"停止"按钮结束测量

## 标定方法

1. 首次运行时，需要输入A4纸到摄像机的参考距离（单位：cm）
2. 系统会自动保存参考参数到`ref_params.json`文件
3. 后续运行时会自动加载这些参数

## 技术实现

- **图像处理**：使用OpenCV进行边缘检测、轮廓提取和形状识别
- **GUI界面**：使用PyQt5构建交互式界面
- **多线程**：使用QThread实现实时图像采集和处理
- **OCR识别**：使用ddddocr库进行数字识别
- **传感器数据**：使用INA226传感器监测电流和功率

![系统架构图](images/system_architecture.png)
*系统架构图*

![算法流程图](images/algorithm_flow.png)
*算法流程图*

## 项目结构

```
├── main.py          # 主程序，包含UI和线程实现
├── core.py          # 核心功能模块
├── utils.py         # 工具函数
├── utils1.py        # 工具函数
├── measurement_utils.py  # 测量相关工具
├── sensor_interface.py   # 传感器接口
├── winUI.py         # UI界面定义
├── ref_params.json  # 参考参数文件
├── FH1.py           # FH1功能实现
├── FH2.py           # FH2功能实现
├── FH3.py           # FH3功能实现
├── FH3A.py          # FH3A功能实现
├── FH3B.py          # FH3B功能实现
├── FH4.py           # FH4功能实现
├── JC1.py           # JC1功能实现
└── README.md        # 项目说明文档
```

## 项目展示

![项目展示图1](images/showcase1.png)
![项目展示图2](images/showcase2.png)
![项目展示图3](images/showcase3.png)

## 注意事项

1. 确保摄像头能够清晰拍摄到目标物体
2. 首次使用时需要正确标定参考距离
3. 测量环境应保持光线充足且稳定
4. 参考矩形（如A4纸）应放置在摄像头视野中央

## 团队信息

- **团队名称**：[团队名称]
- **参赛成员**：[李文卓], [李慧慧], [刘佳慧]
- **指导老师**：[李镇]

![团队徽章](images/team_badge.png)

---

![GitHub stars](https://img.shields.io/github/stars/yourusername/yourrepo.svg?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/yourrepo.svg?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/yourrepo)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/yourrepo)

© 2025 全国大学生电子设计大赛参赛作品