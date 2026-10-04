# YOLOv5-KITTI

这是一个面向 KITTI 派生四分类车辆检测的可复现实验代码仓库。项目基于 **Ultralytics YOLOv5**，加入了 Coordinate Attention、C3Ghost、Deep-C3Ghost、KITTI 数据转换、训练、验证和复杂度分析代码。

原始 YOLOv5 提供基础检测框架。本项目新增内容包括：

- **Coordinate Attention（CA）**：插入 PANet 特征融合路径的注意力模块。
- **C3Ghost**：用于消融实验的 Ghost 结构 C3 替换模块。
- **Deep-C3Ghost**：只在深层 P4/P5 backbone 阶段选择性替换 C3。
- **KITTI 适配与实验代码**：四分类标签转换、数据检查、模型 YAML、训练/验证命令和实验指标记录。

仓库保留当前项目使用的 YOLOv5 上游 LICENSE 文件不变，并注明基于 Ultralytics YOLOv5 修改；源代码继续遵循上游 AGPL-3.0 条款。

## 模型方法

最终模型将 Coordinate Attention 与 Deep-C3Ghost 结合：

- CA 放置在 P3/8 和 P4/16 的 PANet 融合输出之后。
- Deep-C3Ghost 在深层 backbone 的 P4/16 和 P5/32 阶段替换 C3。
- 检测头继续使用标准 YOLOv5s C3 结构。
- Detect 层接收 P3、P4、P5 三个尺度的特征。

模型配置位于 yolov5/models/：

- yolov5s.yaml：原始 YOLOv5s baseline。
- yolov5s_ca.yaml：加入 CA。
- yolov5s_c3ghost.yaml：C3Ghost 消融模型。
- yolov5s_ca_c3ghost.yaml：CA + 广泛 C3Ghost 替换。
- yolov5s_ca_backbone_c3ghost.yaml：CA + backbone C3Ghost。
- yolov5s_ca_deep_c3ghost.yaml：最终 CA + Deep-C3Ghost 模型。

## KITTI 数据

仓库不包含 KITTI 数据集。请用户自行从官方渠道下载并在本地转换。四个类别为：

| ID | 类别 |
|---:|---|
| 0 | Car |
| 1 | Van |
| 2 | Truck |
| 3 | Tram |

准备后的目录格式：

    datasets/kitti-yolo/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

转换和检查：

    python scripts/prepare_kitti_yolo.py --images-dir path/to/training/image_2 --labels-dir path/to/training/label_2 --output datasets/kitti-yolo
    python scripts/check_kitti_yolo.py --data datasets/kitti-yolo

数据集和生成的 cache 均由 .gitignore 排除，不会上传到 GitHub。

## 环境安装

依赖版本固定为当前实验环境使用的版本；实验使用 CUDA 12.6 版本的 PyTorch。

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt

仓库不包含预训练权重，请自行下载或提供本地 checkpoint。

## 训练命令

在仓库根目录执行。

YOLOv5s baseline：

    python yolov5/train.py --weights yolov5s.pt --cfg yolov5/models/yolov5s.yaml --data data/kitti.yaml --img 640 --batch-size 8 --epochs 100 --project runs --name baseline-100

最终模型：

    python yolov5/train.py --weights "" --cfg yolov5/models/yolov5s_ca_deep_c3ghost.yaml --data data/kitti.yaml --img 640 --batch-size 8 --epochs 100 --project runs --name ca-deep-c3ghost-100

验证：

    python yolov5/val.py --weights path/to/best.pt --data data/kitti.yaml --img 640 --batch-size 8 --device 0

模型结构和复杂度检查：

    python scripts/check_ca_model.py

## 最终实验结果

验证设置为 640x640 输入、batch size 8、100 epochs、seed 42 和固定 KITTI 派生数据划分。下表来自已记录的项目汇总结果。

| 模型 | P | R | mAP50 | mAP50-95 | 参数量 | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s Baseline | 0.9250 | 0.8850 | 0.9560 | 0.6940 | 7,030,417 | 15.9691 |
| CA | 0.9351 | 0.9015 | 0.9654 | 0.7340 | 7,040,449 | 15.9729 |
| C3Ghost | 0.9373 | 0.8726 | 0.9536 | 0.7016 | 4,904,953 | 10.6025 |
| CA + C3Ghost | 0.9301 | 0.8750 | 0.9530 | 0.6950 | 4,914,985 | 10.6038 |
| CA + Backbone-C3Ghost | 0.9206 | 0.8910 | 0.9589 | 0.7132 | 5,877,561 | 12.5652 |
| **CA + Deep-C3Ghost** | **0.9503** | 0.8993 | **0.9651** | **0.7244** | **5,962,273** | **14.0046** |

RTX 4060 Laptop GPU 上的 batch=1 FP32 forward 测试：

| 模型 | 参数量 | GFLOPs | 延迟 | FPS |
|---|---:|---:|---:|---:|
| YOLOv5s Baseline | 7,030,417 | 15.9691 | 5.9062 ms | 169.31 |
| CA + Deep-C3Ghost | 5,962,273 | 14.0021 | 6.2604 ms | 159.73 |

测试包含 50 次 warmup 和 500 次计时 forward，不包含数据读取、传输、NMS 和绘图。

## 项目目录

    data/       KITTI YAML 模板
    results/    清理后的指标和实验协议摘要
    scripts/    KITTI 转换、检查、模型 smoke test、benchmark 脚本
    yolov5/     YOLOv5 源码及项目模型扩展

以下内容不会上传：KITTI 数据集、runs/、虚拟环境、IDE 配置、cache、日志、临时文件和模型权重。

## 归属说明

本项目基于并修改 Ultralytics YOLOv5。上游许可证和原作者归属请参见仓库中的 LICENSE 文件。
