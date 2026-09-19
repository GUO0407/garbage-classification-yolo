import os
import glob
import numpy as np
import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"
DATASET_ROOT = "custom_dataset"
DATASETS = ["pool", "train", "val", "test"]

def obb_iou(pts1, pts2):
    p1 = np.ascontiguousarray(pts1, dtype=np.float32)
    p2 = np.ascontiguousarray(pts2, dtype=np.float32)
    area1 = cv2.contourArea(p1)
    area2 = cv2.contourArea(p2)
    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def apply_agnostic_nms(boxes_list, iou_thresh=0.40):
    if len(boxes_list) <= 1:
        return boxes_list
    boxes_list = sorted(boxes_list, key=lambda x: x['conf'], reverse=True)
    kept = []
    while boxes_list:
        best = boxes_list.pop(0)
        kept.append(best)
        boxes_list = [b for b in boxes_list if obb_iou(best['pts'], b['pts']) < iou_thresh]
    return kept

def auto_label_dataset(conf_thresh=0.25, iou_thresh=0.40):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型權重: {MODEL_PATH}")
        return

    print(f"🚀 正在載入模型: {MODEL_PATH} 進行自動預標註...")
    model = YOLO(MODEL_PATH)

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    total_auto_labeled = 0
    total_boxes_generated = 0

    for split in DATASETS:
        images_dir = os.path.join(DATASET_ROOT, split, "images")
        labels_dir = os.path.join(DATASET_ROOT, split, "labels")

        if not os.path.exists(images_dir):
            continue
        os.makedirs(labels_dir, exist_ok=True)

        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        image_paths = sorted(image_paths)

        unlabeled_images = []
        for p in image_paths:
            name, _ = os.path.splitext(os.path.basename(p))
            txt_path = os.path.join(labels_dir, name + ".txt")
            if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
                unlabeled_images.append(p)

        if not unlabeled_images:
            print(f"✅ [{split.upper()}] 資料夾所有圖片皆已有標籤，無需重複預標註。")
            continue

        print(f"\n🔍 [{split.upper()}] 發現 {len(unlabeled_images)} 張尚未標註的照片，開始 AI 自動預測標註...")

        for img_path in unlabeled_images:
            filename = os.path.basename(img_path)
            name, _ = os.path.splitext(filename)
            txt_path = os.path.join(labels_dir, name + ".txt")

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            # 1. 執行推論
            results = model(img_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

            boxes_list = []
            if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
                xyxyxyxy = results.obb.xyxyxyxy.cpu().numpy()
                confs = results.obb.conf.cpu().numpy()
                clss = results.obb.cls.cpu().numpy().astype(int)

                for i in range(len(confs)):
                    boxes_list.append({
                        'cls': clss[i],
                        'name': model.names[clss[i]],
                        'conf': float(confs[i]),
                        'pts': xyxyxyxy[i]
                    })

            # 2. 跨類別 NMS (消除重複預測)
            boxes_list = apply_agnostic_nms(boxes_list, iou_thresh=iou_thresh)

            # 3. 儲存為 9 欄位標準 YOLO-OBB 格式
            if boxes_list:
                with open(txt_path, "w") as fp:
                    for b in boxes_list:
                        # Normalize 4 corners coordinates
                        norm_pts = []
                        for px, py in b['pts']:
                            nx = max(0.0, min(1.0, float(px) / float(w)))
                            ny = max(0.0, min(1.0, float(py) / float(h)))
                            norm_pts.append(f"{nx:.6f} {ny:.6f}")
                        pts_str = " ".join(norm_pts)
                        fp.write(f"{b['cls']} {pts_str}\n")

                print(f"  • {filename} -> 自動標註 {len(boxes_list)} 個物件 (已儲存)")
                total_auto_labeled += 1
                total_boxes_generated += len(boxes_list)
            else:
                print(f"  • {filename} -> 信心度閾值內未檢測到物件 (保留空白)")

    print("\n" + "="*55)
    print(f"🎉 自動預標註完成！")
    print(f"  • 處理新照片: {total_auto_labeled} 張")
    print(f"  • 產生 OBB 旋轉標籤: {total_boxes_generated} 個")
    print("="*55)
    print("👉 現在你可以直接執行 `python label_tool.py` 進行快速檢視與微調！\n")

if __name__ == '__main__':
    auto_label_dataset()
