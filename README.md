# YOLO Models for Autonomous Driving

This project is a comprehensive toolkit for researchers and developers working on autonomous driving datasets. It automates the conversion of different file formats into the YOLO format and facilitates training of various YOLO model versions.

## Features

- **Multi-format Converter:**  
  Convert common autonomous driving dataset annotations (e.g., Pascal VOC, COCO, [KITTI](https://www.cvlibs.net/datasets/kitti/eval_tracking.php), [BDD100K](https://arxiv.org/abs/1805.04687) into the YOLO format.
  
- **YOLO Trainer:**  
  Seamlessly train different versions of YOLO (v10, [v11](https://github.com/ultralytics/ultralytics), ·[v12](https://arxiv.org/abs/2502.12524), etc.) using pre-configured training pipelines.

- **Modular and Extensible:**  
  Designed with modularity in mind, allowing you to add new conversion modules or support additional YOLO versions as needed.

## Getting Started

### Prerequisites

- Python 3.10
- Git
- [uv](https://github.com/astral-sh/uv)

### Installation

```bash
git clone git@github.com:vs-uulm/yolo-for-autonomous-driving.git
cd yolo-for-autonomous-driving
uv sync
```

## Usage Example

### Download and Convert KITTI Dataset to YOLO Format

```bash
uv run dataset_to_yolo_converter.py ./config/converter/kitti.yaml

```

### Train YOLOv11s from Scretch and Pretrained from COCO

```bash
uv run train_yolos.py ./config/trainer/yolov11_kitti.yaml
```

### Check The Model Quality

```bash
cd runs
```
