---
license: cc-by-4.0
task_categories:
- object-detection
tags:
- yolo
- yolo-obb
- yolo11
- robotics
- parallel-gripper
- grasping
- garbage-classification
- realsense
size_categories:
- n<1K
---

# Robotic Garbage OBB Dataset (`robotic-garbage-obb`)

![Demo](demo.gif)

This dataset is designed for **robotic parallel gripper grasping** and **top-down garbage classification** using **Oriented Bounding Boxes (OBB)**.

Images are captured in a real robotic bird's-eye view workspace (pure black background tabletop) using Intel RealSense cameras, labeled with oriented bounding boxes to provide both classification and exact planar yaw ($\theta$) orientation angles for robotic parallel grippers.

---

## 📸 Data Sources & Composition

The dataset contains **472 top-down images** curated specifically for robotic tabletop grasping:
- **Real-world Robotic Captures**:
  - Captured directly from the bird's-eye view camera of a robotic manipulation platform using **Intel RealSense** cameras against a matte black background.
  - Paired aligned depth maps (`.npy`, uint16 mm) and JET heatmaps are provided in `demo/captures/`.
- **AI-Synthesized Augmentation (Google Nano Banana)**:
  - Part of the dataset images were generated using **Google's Nano Banana** AI model to enrich visual diversity, augment object orientation distributions, and balance class frequencies.
  - All AI-generated images underwent strict manual screening to ensure visual fidelity, realistic tabletop shadows, and consistent perspective alignment with physical robot workspace conditions.
- **OBB Manual Annotation**:
  - All images were verified and annotated with oriented bounding boxes (OBB) providing 4-corner coordinates and yaw grasping angles.

---

## 🏷️ Category Definitions & Gripper Physics Constraints

The dataset is categorized into 4 actionable robotic grasping classes:

| Class ID | Class Name | Description | Robotic Grasping Strategy |
|:---|:---|:---|:---|
| `0` | `plastic` | Recyclable plastics (PET bottles, beverage cups, pudding cups, etc.) | Gripper aligns with the major/minor axis to enclose volume. |
| `1` | `metal` | Recyclable metal containers (aluminum cans, tin cans, coffee cans, etc.) | Aligns with cylindrical symmetry axis. |
| `2` | `paper` | **Rigid structural paper containers** (Tetra Pak foil cartons, fresh house milk cartons, instant noodle paper bowls, paper bento boxes) | Has structural rigidity and height for the gripper to close firmly. |
| `3` | `general_waste` | Non-recyclable or non-graspable waste (crumpled tissue paper, straws, plastic wrap, **flat paper sheets / paper scraps**) | Parallel grippers cannot grasp flat thin sheets resting directly on table surfaces; therefore, flat paper is classified as `general_waste`. |

---

## 📁 Dataset Structure

```text
robotic-garbage-obb/
├── dataset.yaml            # YOLO dataset configuration
├── train/                  # 330 images (70%)
│   ├── images/
│   └── labels/
├── val/                    # 71 images (15%)
│   ├── images/
│   └── labels/
├── test/                   # 71 images (15%)
│   ├── images/
│   └── labels/
├── demo/
│   └── captures/           # RealSense pairs: {stem}_color.png, {stem}_depth.npy (uint16 mm), {stem}_depth_jet.png
└── demo.gif                # Inference preview GIF
```

### 📐 Annotation Format (YOLO-OBB)

Each label file contains 9 normalized space-separated values per object:
```text
<class_index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```
where $(x_1, y_1)$ through $(x_4, y_4)$ are the 4 clockwise corner coordinates normalized to $[0, 1]$.

---

## 🚀 Usage with Ultralytics YOLO11-OBB

```bash
from ultralytics import YOLO

# Train YOLO11-OBB model
model = YOLO("yolo11x-obb.pt")
model.train(data="dataset.yaml", epochs=300, imgsz=1024, batch=16)
```

---

## 📄 License
- Dataset annotations and images are released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
