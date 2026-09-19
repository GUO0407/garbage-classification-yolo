import os
import glob
import argparse
import random
import shutil
from collections import defaultdict

DATASET_ROOT = "custom_dataset"
OLD_SPLITS = ["train", "val", "test"]
NEW_SPLITS = ["train", "val", "test"]
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
NUM_CLASSES = 4
NEG_KEY = -1
EXTENSIONS = ["jpg", "jpeg", "png", "webp"]
TMP_DIR = "custom_dataset/.split_tmp"

def list_images(split_dir):
    paths = []
    for ext in EXTENSIONS:
        paths.extend(glob.glob(os.path.join(split_dir, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(split_dir, f"*.{ext.upper()}")))
    return sorted(paths)

def label_class_counts(label_path):
    counts = defaultdict(int)
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if parts:
                    counts[int(float(parts[0]))] += 1
    return counts

def largest_remainder(total, ratios):
    raw = [total * r for r in ratios]
    floor = [int(x) for x in raw]
    remainder = total - sum(floor)
    fracs = sorted(range(len(raw)), key=lambda i: raw[i] - floor[i], reverse=True)
    for i in range(remainder):
        floor[fracs[i]] += 1
    return floor

def present_classes(counts):
    has_any = any(counts.get(c, 0) > 0 for c in range(NUM_CLASSES))
    if not has_any:
        return [NEG_KEY]
    return [c for c in range(NUM_CLASSES) if counts.get(c, 0) > 0]

def summarize(image_items):
    summary = defaultdict(lambda: defaultdict(int))
    for item in image_items:
        split = item["split"]
        summary[split]["images"] += 1
        counts = item["counts"]
        if present_classes(counts) == [NEG_KEY]:
            summary[split]["negative"] += 1
        else:
            for c in range(NUM_CLASSES):
                summary[split][c] += counts.get(c, 0)
    return summary

def print_summary(summary):
    header = f"{'split':<8}{'images':<8}" + "".join(f"{'cls'+str(c):<8}" for c in range(NUM_CLASSES)) + "negative"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for split in NEW_SPLITS:
        s = summary[split]
        row = f"{split:<8}{s['images']:<8}"
        for c in range(NUM_CLASSES):
            row += f"{s[c]:<8}"
        row += f"{s['negative']}"
        print(row)
    print("-" * len(header))

def main():
    parser = argparse.ArgumentParser(description="將 custom_dataset 依類別平衡分層抽樣重切為 train/val/test")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子 (預設: 42)")
    parser.add_argument("--source", choices=["pool", "splits"], default="pool",
                        help="圖片來源：pool（預設，保留 master）或 splits（舊行為：吃掉 train/val/test）")
    args = parser.parse_args()
    random.seed(args.seed)

    source_splits = ["pool"] if args.source == "pool" else OLD_SPLITS
    items = []
    for split in source_splits:
        images_dir = os.path.join(DATASET_ROOT, split, "images")
        labels_dir = os.path.join(DATASET_ROOT, split, "labels")
        for img_path in list_images(images_dir):
            name, ext = os.path.splitext(os.path.basename(img_path))
            label_path = os.path.join(labels_dir, name + ".txt")
            items.append({
                "old_split": split,
                "img_path": img_path,
                "label_path": label_path if os.path.exists(label_path) else None,
                "ext": ext.lower(),
                "counts": label_class_counts(label_path),
            })

    total = len(items)
    print(f"📊 總共掃描到 {total} 張圖片")
    if total == 0:
        print("❌ 沒有找到任何圖片，請確認資料夾結構")
        return

    ratio_list = [RATIOS[s] for s in NEW_SPLITS]
    target_img = largest_remainder(total, ratio_list)

    class_totals = defaultdict(int)
    for item in items:
        for c in present_classes(item["counts"]):
            class_totals[c] += 1

    target = {}
    for c in list(class_totals.keys()):
        target[c] = largest_remainder(class_totals[c], ratio_list)

    current_img = defaultdict(int)
    current_cls = defaultdict(int)

    shuffled = items[:]
    random.shuffle(shuffled)

    for item in shuffled:
        present = present_classes(item["counts"])
        best_split = None
        best_need = None
        for i, split in enumerate(NEW_SPLITS):
            img_need = target_img[i] - current_img[split]
            if img_need <= 0:
                continue
            cls_need = 0.0
            for c in present:
                cls_need += max(0, target[c][i] - current_cls[(split, c)])
            need = img_need * 1.0 + cls_need * 2.0
            if best_need is None or need > best_need:
                best_need = need
                best_split = split

        if best_split is None:
            for i, split in enumerate(NEW_SPLITS):
                if best_split is None or current_img[split] < current_img[best_split]:
                    best_split = split

        item["split"] = best_split
        current_img[best_split] += 1
        for c in present:
            current_cls[(best_split, c)] += 1

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    counter = defaultdict(int)
    for item in items:
        split = item["split"]
        counter[split] += 1
        new_name = f"{split}_{counter[split]:03d}{item['ext']}"
        tmp_img_dir = os.path.join(TMP_DIR, split, "images")
        tmp_lbl_dir = os.path.join(TMP_DIR, split, "labels")
        os.makedirs(tmp_img_dir, exist_ok=True)
        os.makedirs(tmp_lbl_dir, exist_ok=True)
        tmp_img = os.path.join(tmp_img_dir, new_name)
        if args.source == "pool":
            shutil.copy2(item["img_path"], tmp_img)
        else:
            shutil.move(item["img_path"], tmp_img)
        if item["label_path"] and os.path.exists(item["label_path"]):
            new_lbl = os.path.join(tmp_lbl_dir, f"{split}_{counter[split]:03d}.txt")
            if args.source == "pool":
                shutil.copy2(item["label_path"], new_lbl)
            else:
                shutil.move(item["label_path"], new_lbl)

    for split in OLD_SPLITS:
        old_dir = os.path.join(DATASET_ROOT, split)
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir)

    for split in NEW_SPLITS:
        tmp_split = os.path.join(TMP_DIR, split)
        if os.path.isdir(tmp_split):
            shutil.move(tmp_split, os.path.join(DATASET_ROOT, split))

    print("\n✅ 資料集重新切分完成！各分群統計：")
    print_summary(summarize(items))
    print("📁 新結構:")
    for split in NEW_SPLITS:
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        n = len(list_images(img_dir)) if os.path.isdir(img_dir) else 0
        print(f"  • {split}/images: {n} 張")

if __name__ == '__main__':
    main()