import os
import glob
import re
import math
import argparse
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

CLASS_COLORS = {
    0: (46, 204, 113),   # plastic - 綠色
    1: (235, 152, 52),   # metal - 藍色
    2: (34, 126, 230),   # paper - 橘色
    3: (255, 0, 255)     # general_waste - 螢光洋紅/亮紫紅
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate demo GIF with side-by-side original and annotated comparison.")
    parser.add_argument("--path", type=str, default="custom_dataset/demo", help="Path to input images directory")
    parser.add_argument("--output", type=str, default="demo.gif", help="Path to output GIF file")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to trained YOLO model")
    parser.add_argument("--height", type=int, default=480, help="Target height for each image panel in pixels (width dynamically follows aspect ratio)")
    parser.add_argument("--ref-image", type=str, default=None, help="Optional specific image path to use as aspect ratio reference")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="NMS IoU threshold")
    parser.add_argument("--duration", type=int, default=1000, help="Duration per frame in milliseconds")
    parser.add_argument("--no-grasp", action="store_true", help="Disable grasp pose overlay")
    return parser.parse_args()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

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

def depth_to_jet_display(depth_mm: np.ndarray):
    """uint16 mm -> JET BGR uint8，0=黑色無效區；percentile + 2m cap 自適應刻度（與 test_gui 一致）"""
    if depth_mm.dtype != np.uint16 or depth_mm.size == 0:
        return None
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, (2, 98))
    hi = max(float(hi), 2000.0)
    clipped = np.clip(depth_mm, lo, hi).astype(np.float32)
    norm8 = ((clipped - lo) / (hi - lo + 1e-6)).astype(np.float32) * 255.0
    jet = cv2.applyColorMap(norm8.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(jet, cv2.COLOR_RGB2BGR)

def find_depth_jet(image_path: str):
    """找同 stem 的 depth（與 test_gui/capture_dataset.py 命名一致），回傳 JET BGR uint8（原圖解析度）或 None。
    優先 {stem}[_color-stripped]_depth.npy（uint16 mm），回退 {stem}_depth_jet.png。"""
    d = os.path.dirname(os.path.abspath(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if stem.endswith("_color"):
        stem = stem[: -len("_color")]
    for base in (stem,):
        npy_path = os.path.join(d, f"{base}_depth.npy")
        if os.path.exists(npy_path):
            depth_mm = np.load(npy_path)
            if isinstance(depth_mm, np.ndarray) and depth_mm.dtype == np.uint16:
                jet = depth_to_jet_display(depth_mm)
                if jet is not None:
                    return jet
            continue
        jet_path = os.path.join(d, f"{base}_depth_jet.png")
        if os.path.exists(jet_path):
            depth_jet = cv2.imread(jet_path, cv2.IMREAD_COLOR)
            if depth_jet is not None:
                return depth_jet
    return None

def _tag(canvas: np.ndarray, text: str, color: tuple) -> None:
    """於面板左上角畫標題（白色粗描邊 + 專屬色文字），比例隨解析度自適應（與 test_gui 一致）"""
    w = canvas.shape[1]
    tag_scale = max(0.7, min(1.2, w / 1000.0))
    tag_thick = max(2, int(round(tag_scale * 2.0)))
    pos = (max(6, int(w * 0.014)), int(35 * tag_scale + 5))
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (0, 0, 0), tag_thick + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, color, tag_thick, cv2.LINE_AA)

def draw_boxes(img: np.ndarray, boxes_list, show_grasp: bool) -> np.ndarray:
    """在給定圖片上畫 OBB 框 + badge（+ Grasp Pose）；RGB 與 JET 深度共用（與 test_gui._draw_boxes 一致）。傳入 copy，會就地修改並回傳。"""
    img_h, img_w = img.shape[:2]
    font_scale = max(0.85, min(1.4, img_w / 900.0))
    font_thick = max(2, int(round(font_scale * 2.0)))
    line_thick = max(2, int(round(img_w / 450.0)))

    for b in boxes_list:
        cls_id = b['cls']
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        pts = b['pts'].astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_thick, lineType=cv2.LINE_AA)

        top_pt = b['pts'][np.argmin(b['pts'][:, 1])]
        bx, by = int(top_pt[0]), int(top_pt[1])

        badge_text = f" {b['name']} {b['conf']:.2f} "
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

        pad_y = int(th * 0.35)
        badge_y1 = max(0, by - th - pad_y * 2)
        badge_y2 = badge_y1 + th + pad_y * 2
        badge_x1 = max(0, bx)

        text_x = max(4, badge_x1)
        text_y = badge_y2 - pad_y
        outline_radius = max(2, int(round(font_scale * 2.5)))

        for dx in range(-outline_radius, outline_radius + 1):
            for dy in range(-outline_radius, outline_radius + 1):
                if dx != 0 or dy != 0:
                    cv2.putText(img, badge_text, (text_x + dx, text_y + dy),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick + 1, cv2.LINE_AA)

        cv2.putText(img, badge_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

        if show_grasp:
            cx, cy, bw, bh, rad = b['xywhr']
            grasp_len = max(bw, bh) * 0.5
            dx = (grasp_len / 2.0) * math.cos(rad + math.pi / 2.0)
            dy = (grasp_len / 2.0) * math.sin(rad + math.pi / 2.0)
            p1 = (int(cx - dx), int(cy - dy))
            p2 = (int(cx + dx), int(cy + dy))
            cv2.line(img, p1, p2, (255, 255, 0), max(2, line_thick), cv2.LINE_AA)
            dx2 = (grasp_len / 3.0) * math.cos(rad)
            dy2 = (grasp_len / 3.0) * math.sin(rad)
            p3 = (int(cx - dx2), int(cy - dy2))
            p4 = (int(cx + dx2), int(cy + dy2))
            cv2.line(img, p3, p4, (0, 255, 255), max(2, line_thick), cv2.LINE_AA)
            cv2.circle(img, (int(cx), int(cy)), max(4, line_thick + 2), (0, 0, 255), -1, cv2.LINE_AA)

    return img

def main():
    args = parse_args()
    
    # 支援替代模型路徑
    model_path = args.model
    if not os.path.exists(model_path):
        alt_weights = [
            "best.pt",
            "garbage_classification_runs/yolo11x_obb_model/weights/best.pt",
            "yolo11x-obb.pt"
        ]
        for alt in alt_weights:
            if os.path.exists(alt):
                model_path = alt
                break

    print(f"🚀 正在載入模型: {model_path} ...")
    model = YOLO(model_path)
    
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.path, ext)))
        image_paths.extend(glob.glob(os.path.join(args.path, ext.upper())))

    # 深度預覽圖（*_depth_jet.png）不是資料圖，排除掉（與 test_gui.load_random_image 一致）
    image_paths = [p for p in image_paths if not os.path.basename(p).endswith("_depth_jet.png")]

    image_paths = sorted(image_paths, key=natural_sort_key)
    total_imgs = len(image_paths)
    if total_imgs == 0:
        print(f"❌ 在目錄 '{args.path}' 中找不到任何圖片！")
        return

    print(f"📸 找到 {total_imgs} 張測試圖片，開始計算畫布比例與繪製標註影像...")
    
    # 以參考圖的原生解析度作為「工作畫布」尺寸；整張組合影像最後再做一次性等比例縮放到 --height。
    ref_img_path = args.ref_image if (args.ref_image and os.path.exists(args.ref_image)) else image_paths[0]
    ref_bgr = cv2.imread(ref_img_path)
    if ref_bgr is not None:
        panel_h = ref_bgr.shape[0]
        panel_w = ref_bgr.shape[1]
    else:
        _, panel_w, panel_h = 0, int(480 * 16 / 9), 480

    gif_frames = []
    show_grasp = not args.no_grasp
    print(f"📐 工作畫布: 單面板 ({panel_w}x{panel_h})，輸出依 --height={args.height} 等比例縮放 (參考圖: {os.path.basename(ref_img_path)})")

    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img_h, img_w = img_bgr.shape[:2]

        # 1. 執行推論
        results = model(img_bgr, conf=args.conf, iou=args.iou, verbose=False)[0]

        boxes_list = []
        if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
            xyxyxyxy = results.obb.xyxyxyxy.cpu().numpy()
            xywhr = results.obb.xywhr.cpu().numpy()
            confs = results.obb.conf.cpu().numpy()
            clss = results.obb.cls.cpu().numpy().astype(int)

            for i in range(len(confs)):
                boxes_list.append({
                    'cls': clss[i],
                    'name': model.names[clss[i]],
                    'conf': float(confs[i]),
                    'pts': xyxyxyxy[i],
                    'xywhr': xywhr[i]
                })

        # 2. 跨類別 NMS
        boxes_list = apply_agnostic_nms(boxes_list, iou_thresh=args.iou)

        # 3. 深度（與 test_gui.find_depth_for 同規則）
        depth_jet = find_depth_jet(img_path)
        depth_ok = depth_jet is not None and depth_jet.shape[:2] == (img_h, img_w)
        if not depth_ok:
            depth_jet = None

        # 4. 原生解析度渲染（與 test_gui.render_panels 一致的 2x2 佈局）
        raw_display = img_bgr.copy()
        annotated = draw_boxes(img_bgr.copy(), boxes_list, show_grasp)
        _tag(raw_display, "Original", (255, 255, 255))
        _tag(annotated, f"OBB Detection ({len(boxes_list)} objects)", (0, 255, 255))

        col = np.hstack([raw_display, annotated])
        if depth_jet is not None:
            depth_panel = draw_boxes(depth_jet.copy(), boxes_list, show_grasp)
            _tag(depth_jet, "Depth", (255, 255, 0))
            _tag(depth_panel, "Depth + OBB", (255, 255, 0))
            col_depth = np.hstack([depth_jet, depth_panel])
            divider_h = max(3, int(img_h / 300.0))
            divider = np.zeros((divider_h, img_w * 2, 3), dtype=np.uint8)
            divider[:] = (45, 45, 45)
            combined = np.vstack([col, divider, col_depth])
        else:
            combined = col

        # 5. HUD 資訊列（依組合影像寬度）
        canvas_w = combined.shape[1]
        hud_h = max(32, int(canvas_w * 0.04))
        hud_bar = np.full((hud_h, canvas_w, 3), 25, dtype=np.uint8)
        hud_font_scale = max(0.5, min(0.9, canvas_w / 1600.0))
        hud_thick = max(1, int(round(hud_font_scale * 2)))
        hud_text_y = int(hud_h * 0.70)

        label_filename = filename if len(filename) <= 48 else "..." + filename[-45:]
        cv2.putText(hud_bar, f"YOLOv11-OBB Inference Demo [{idx+1}/{total_imgs}] {label_filename}", (16, hud_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (255, 255, 255), hud_thick, cv2.LINE_AA)

        status = f"Detections: {len(boxes_list)}" + ("  |  Depth: loaded" if depth_jet is not None else "  |  Depth: n/a")
        (cw2, ch2), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, hud_thick)
        cv2.putText(hud_bar, status, (canvas_w - cw2 - 16, hud_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, hud_font_scale, (0, 255, 255), hud_thick, cv2.LINE_AA)

        frame = np.vstack([combined, hud_bar])

        # 6. 一次性等比例縮放到 --height（GIF 尺寸穩定、解析度不爆掉）
        out_h = args.height
        scale = out_h / float(frame.shape[0])
        out_w = max(1, int(round(frame.shape[1] * scale)))
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # 轉換為 RGB PIL Image 並做調色盤量化
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).convert('P', palette=Image.ADAPTIVE, colors=256)
        gif_frames.append(pil_frame)
        depth_tag = "有depth" if depth_jet is not None else "無depth"
        print(f"  [{idx+1}/{total_imgs}] {filename} -> 完成 (偵測到 {len(boxes_list)} 個垃圾, {depth_tag})")

    if gif_frames:
        print(f"\n🎬 正在產生最佳化 GIF 動圖（每張 {args.duration/1000.0:.1f} 秒，共 {len(gif_frames)} 幀）...")
        gif_frames[0].save(
            args.output,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=args.duration,
            loop=0
        )
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"🎉 {args.output} 製作完成！(大小: {file_size_mb:.2f} MB)")
    else:
        print("❌ 沒有產生任何幀。")

if __name__ == '__main__':
    main()
