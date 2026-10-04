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
    python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
    python -m pip install -r requirements.txt

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

下表严格按 2026-10-04 提交的最终论文消融表取值：640x640 输入、batch size 8、100 epochs、seed 42 和固定 KITTI 派生数据划分。最终模型的选择依据是相对于 YOLOv5s baseline 的 accuracy-complexity trade-off；不声称它在所有消融模型中精度最高，因为 CA 单独模型的 mAP50-95 为 0.734。

| 模型 | P | R | mAP50 | mAP50-95 | 参数量 | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv5s Baseline | 0.925 | 0.885 | 0.956 | 0.694 | 7.03M | 15.97 |
| CA | 0.935 | 0.901 | 0.965 | 0.734 | 7.04M | 15.97 |
| C3Ghost | 0.937 | 0.873 | 0.954 | 0.702 | 4.90M | 10.60 |
| CA + Full-C3Ghost | 0.930 | 0.875 | 0.953 | 0.695 | 4.91M | 10.60 |
| CA + Backbone-C3Ghost | 0.921 | 0.891 | 0.959 | 0.713 | 5.88M | 12.57 |
| **CA + Deep-C3Ghost** | 0.950 | 0.899 | 0.965 | 0.724 | 5.96M | 14.00 |

下面的详细 forward benchmark 遵循论文中报告的相同协议；额外小数位是测量细节，与四舍五入后的主结果表分开保存。

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
