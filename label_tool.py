import os
import glob
import re
import math
import numpy as np
import cv2

DATASET_ROOT = "custom_dataset"
DATASETS = ["pool", "train", "val", "test"]

def auto_standardize_filenames(dataset_name):
    """
    自動掃描 dataset_name (如 train / test) 資料夾。
    若發現未符合 {prefix}_{idx:03d}.ext 格式的新圖片，
    會自動接續當前最大編號，同步將圖片與對應 txt 標籤重命名為標準格式。
    """
    images_dir = os.path.join(DATASET_ROOT, dataset_name, "images")
    labels_dir = os.path.join(DATASET_ROOT, dataset_name, "labels")
    if not os.path.exists(images_dir):
        return

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    all_imgs = []
    for ext in extensions:
        all_imgs.extend(glob.glob(os.path.join(images_dir, ext)))
        all_imgs.extend(glob.glob(os.path.join(images_dir, ext.upper())))

    prefix = dataset_name
    pattern = re.compile(rf"^{prefix}_(\d+)\.(jpg|jpeg|png|webp)$", re.IGNORECASE)

    standard_indices = []
    non_standard_files = []

    for img_path in all_imgs:
        filename = os.path.basename(img_path)
        match = pattern.match(filename)
        if match:
            standard_indices.append(int(match.group(1)))
        else:
            non_standard_files.append(img_path)

    if not non_standard_files:
        return

    max_idx = max(standard_indices) if standard_indices else 0
    non_standard_files = sorted(non_standard_files)

    print(f"\n✨ 自動偵測到 {len(non_standard_files)} 張新加入的圖片，正在自動標準化檔名編號...")
    for old_img_path in non_standard_files:
        max_idx += 1
        old_filename = os.path.basename(old_img_path)
        name, ext = os.path.splitext(old_filename)
        ext = ext.lower()

        new_base = f"{prefix}_{max_idx:03d}"
        new_img_path = os.path.join(images_dir, f"{new_base}{ext}")
        old_lbl_path = os.path.join(labels_dir, name + ".txt")
        new_lbl_path = os.path.join(labels_dir, f"{new_base}.txt")

        os.rename(old_img_path, new_img_path)
        if os.path.exists(old_lbl_path):
            os.rename(old_lbl_path, new_lbl_path)

        print(f"  • {old_filename} -> {new_base}{ext}")

    print(f"🎉 檔名標準化完成！{dataset_name} 當前總圖片數已擴充至: {max_idx} 張\n")

CLASSES = {
    ord('0'): (0, "Plastic"),
    ord('1'): (1, "Metal"),
    ord('2'): (2, "Paper"),
    ord('3'): (3, "General Waste")
}

CLASS_NAMES = ["Plastic", "Metal", "Paper", "General Waste"]

COLORS = [
    (0, 255, 0),     # 0: Plastic - 綠色 (BGR)
    (255, 230, 0),   # 1: Metal - 亮青黃色
    (0, 140, 255),   # 2: Paper - 鮮橘色
    (255, 0, 255)    # 3: General Waste - 螢光洋紅/亮紫紅 (超高對比，一眼看清)
]

class OBBBox:
    def __init__(self, cls_id, cx, cy, w, h, angle_deg=0.0):
        self.cls_id = int(cls_id)
        self.cx = float(cx)
        self.cy = float(cy)
        self.w = max(5.0, float(w))
        self.h = max(5.0, float(h))
        self.angle_deg = float(angle_deg) % 360.0

    def get_corners(self):
        rad = math.radians(self.angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        hw, hh = self.w / 2.0, self.h / 2.0
        
        # Local corners: top-left (p0), top-right (p1), bottom-right (p2), bottom-left (p3)
        local_pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        world_pts = []
        for lx, ly in local_pts:
            wx = lx * cos_a - ly * sin_a + self.cx
            wy = lx * sin_a + ly * cos_a + self.cy
            world_pts.append((wx, wy))
        return world_pts

    def get_rotation_handle(self, offset=30.0):
        rad = math.radians(self.angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        hh = self.h / 2.0
        # Local position along negative Y axis (top direction)
        lx, ly = 0.0, -hh - offset
        wx = lx * cos_a - ly * sin_a + self.cx
        wy = lx * sin_a + ly * cos_a + self.cy
        # Top edge midpoint
        tm_x = 0.0 * cos_a - (-hh) * sin_a + self.cx
        tm_y = 0.0 * sin_a + (-hh) * cos_a + self.cy
        return (wx, wy), (tm_x, tm_y)

    def get_edge_midpoints(self):
        rad = math.radians(self.angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        hw, hh = self.w / 2.0, self.h / 2.0
        # Top, Right, Bottom, Left midpoints
        local_mids = [(0, -hh), (hw, 0), (0, hh), (-hw, 0)]
        world_mids = []
        for lx, ly in local_mids:
            wx = lx * cos_a - ly * sin_a + self.cx
            wy = lx * sin_a + ly * cos_a + self.cy
            world_mids.append((wx, wy))
        return world_mids

    def to_local(self, px, py):
        dx = px - self.cx
        dy = py - self.cy
        rad = math.radians(-self.angle_deg)
        lx = dx * math.cos(rad) - dy * math.sin(rad)
        ly = dx * math.sin(rad) + dy * math.cos(rad)
        return lx, ly

    def contains_point(self, px, py):
        lx, ly = self.to_local(px, py)
        return abs(lx) <= (self.w / 2.0) and abs(ly) <= (self.h / 2.0)

    def copy(self):
        return OBBBox(self.cls_id, self.cx, self.cy, self.w, self.h, self.angle_deg)


# Global State
current_dataset_idx = 0
boxes = []
selected_box_idx = -1
history_stack = []

# Interaction Modes: 'IDLE', 'DRAWING_NEW', 'MOVING', 'ROTATING', 'RESIZING_HANDLE'
interaction_mode = 'IDLE'
drag_start_pos = (0, 0)
drag_start_box = None
active_handle_idx = -1  # 0~3 for corners, 4~7 for edges, 8 for rotation handle

def push_history():
    global history_stack
    history_stack.append([b.copy() for b in boxes])
    if len(history_stack) > 20:
        history_stack.pop(0)

def load_labels(txt_path, img_w, img_h):
    loaded = []
    if not os.path.exists(txt_path):
        return loaded

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # Standard HBB: cls cx cy w h (normalized)
                cls_id = int(parts[0])
                cx = float(parts[1]) * img_w
                cy = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                loaded.append(OBBBox(cls_id, cx, cy, w, h, 0.0))
            elif len(parts) == 9:
                # Standard YOLO OBB: cls x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                cls_id = int(parts[0])
                x1, y1 = float(parts[1]) * img_w, float(parts[2]) * img_h
                x2, y2 = float(parts[3]) * img_w, float(parts[4]) * img_h
                x3, y3 = float(parts[5]) * img_w, float(parts[6]) * img_h
                x4, y4 = float(parts[7]) * img_w, float(parts[8]) * img_h
                
                # Compute center, width, height, and orientation angle
                pts = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))
                
                # Vector along top edge p0 -> p1
                vx = x2 - x1
                vy = y2 - y1
                w = float(math.hypot(vx, vy))
                
                # Vector along side edge p1 -> p2
                ux = x3 - x2
                uy = y3 - y2
                h = float(math.hypot(ux, uy))
                
                angle_deg = float(math.degrees(math.atan2(vy, vx))) % 360.0
                loaded.append(OBBBox(cls_id, cx, cy, w, h, angle_deg))
    return loaded

def save_labels(txt_path, img_w, img_h):
    if len(boxes) == 0:
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return

    with open(txt_path, "w") as f:
        for b in boxes:
            corners = b.get_corners()
            # Normalize 4 corners coordinates
            norm_pts = []
            for px, py in corners:
                nx = max(0.0, min(1.0, px / float(img_w)))
                ny = max(0.0, min(1.0, py / float(img_h)))
                norm_pts.append(f"{nx:.6f} {ny:.6f}")
            pts_str = " ".join(norm_pts)
            f.write(f"{b.cls_id} {pts_str}\n")

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def on_mouse(event, x, y, flags, param):
    global interaction_mode, drag_start_pos, drag_start_box, active_handle_idx
    global selected_box_idx, boxes

    HANDLE_RADIUS = 10
    # Compensate for 42px header banner offset
    y_adj = y - 42

    # 1. MOUSE WHEEL FOR ROTATION
    if event == cv2.EVENT_MOUSEWHEEL:
        if selected_box_idx >= 0 and selected_box_idx < len(boxes):
            push_history()
            delta = 2.0
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                delta = 10.0
            elif flags & cv2.EVENT_FLAG_CTRLKEY:
                delta = 0.5
            
            # Check wheel direction
            if flags > 0:
                boxes[selected_box_idx].angle_deg = (boxes[selected_box_idx].angle_deg + delta) % 360.0
            else:
                boxes[selected_box_idx].angle_deg = (boxes[selected_box_idx].angle_deg - delta) % 360.0
        return True   # 吃掉滾輪事件，避免後端預設行為把圖片放大縮小

    # 2. RIGHT CLICK TO DELETE OR DESELECT
    if event == cv2.EVENT_RBUTTONDOWN:
        if y_adj < 0:
            return
        for i in range(len(boxes) - 1, -1, -1):
            if boxes[i].contains_point(x, y_adj):
                push_history()
                print(f"🗑️ 刪除框: {CLASS_NAMES[boxes[i].cls_id]}")
                boxes.pop(i)
                selected_box_idx = -1
                return
        selected_box_idx = -1
        return

    # 3. LEFT BUTTON DOWN
    if event == cv2.EVENT_LBUTTONDOWN:
        if y_adj < 0:
            return
        drag_start_pos = (x, y_adj)

        # Check if clicking on active handle of currently selected box
        if selected_box_idx >= 0 and selected_box_idx < len(boxes):
            cur_box = boxes[selected_box_idx]
            rot_handle, _ = cur_box.get_rotation_handle()
            corners = cur_box.get_corners()
            edge_mids = cur_box.get_edge_midpoints()

            # A. Rotation Handle Check
            if dist((x, y_adj), rot_handle) <= HANDLE_RADIUS + 5:
                push_history()
                interaction_mode = 'ROTATING'
                drag_start_box = cur_box.copy()
                return

            # B. Corner Handles Check (0: TL, 1: TR, 2: BR, 3: BL)
            for c_idx, cp in enumerate(corners):
                if dist((x, y_adj), cp) <= HANDLE_RADIUS + 5:
                    push_history()
                    interaction_mode = 'RESIZING_HANDLE'
                    active_handle_idx = c_idx
                    drag_start_box = cur_box.copy()
                    return

            # C. Edge Handles Check (4: Top, 5: Right, 6: Bottom, 7: Left)
            for e_idx, ep in enumerate(edge_mids):
                if dist((x, y_adj), ep) <= HANDLE_RADIUS + 5:
                    push_history()
                    interaction_mode = 'RESIZING_HANDLE'
                    active_handle_idx = 4 + e_idx
                    drag_start_box = cur_box.copy()
                    return

        # Check if clicking inside any box (Select & Move)
        clicked_idx = -1
        for i in range(len(boxes) - 1, -1, -1):
            if boxes[i].contains_point(x, y_adj):
                clicked_idx = i
                break

        if clicked_idx != -1:
            push_history()
            selected_box_idx = clicked_idx
            interaction_mode = 'MOVING'
            drag_start_box = boxes[clicked_idx].copy()
        else:
            # Click on empty space -> Start drawing a new box
            interaction_mode = 'DRAWING_NEW'
            selected_box_idx = -1

    # 4. MOUSE MOVE
    elif event == cv2.EVENT_MOUSEMOVE:
        if interaction_mode == 'MOVING' and selected_box_idx >= 0:
            cur_box = boxes[selected_box_idx]
            dx = x - drag_start_pos[0]
            dy = y_adj - drag_start_pos[1]
            cur_box.cx = drag_start_box.cx + dx
            cur_box.cy = drag_start_box.cy + dy

        elif interaction_mode == 'ROTATING' and selected_box_idx >= 0:
            cur_box = boxes[selected_box_idx]
            angle_rad = math.atan2(y_adj - cur_box.cy, x - cur_box.cx)
            cur_box.angle_deg = (math.degrees(angle_rad) + 90.0) % 360.0

        elif interaction_mode == 'RESIZING_HANDLE' and selected_box_idx >= 0:
            cur_box = boxes[selected_box_idx]
            orig_box = drag_start_box
            lx, ly = orig_box.to_local(x, y_adj)

            if active_handle_idx in [0, 1, 2, 3]:
                # Corner resize
                cur_box.w = max(10.0, abs(lx) * 2.0)
                cur_box.h = max(10.0, abs(ly) * 2.0)
            elif active_handle_idx in [4, 6]:
                # Top / Bottom edge: adjust height only
                cur_box.h = max(10.0, abs(ly) * 2.0)
            elif active_handle_idx in [5, 7]:
                # Left / Right edge: adjust width only
                cur_box.w = max(10.0, abs(lx) * 2.0)

    # 5. LEFT BUTTON UP
    elif event == cv2.EVENT_LBUTTONUP:
        if interaction_mode == 'DRAWING_NEW':
            x1, y1 = drag_start_pos
            x2, y2 = x, y_adj
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            bw = max_x - min_x
            bh = max_y - min_y
            if bw >= 10 and bh >= 10:
                push_history()
                new_box = OBBBox(cls_id=0, cx=(min_x + max_x)/2.0, cy=(min_y + max_y)/2.0, w=bw, h=bh, angle_deg=0.0)
                boxes.append(new_box)
                selected_box_idx = len(boxes) - 1
                print(f"✨ 建立新標籤框 (預設 0:Plastic)，按 0~3 切換類別，拖曳頂部黃點可旋轉")

        interaction_mode = 'IDLE'
        drag_start_box = None
        active_handle_idx = -1

def render_ui(display_img, img_w, img_h, img_idx, total_imgs, filename, dataset_name):
    # 1. Draw All Bounding Boxes
    for i, b in enumerate(boxes):
        corners = b.get_corners()
        pts_np = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        
        is_selected = (i == selected_box_idx)
        color = COLORS[b.cls_id]
        
        # Draw Box Polygon
        cv2.polylines(display_img, [pts_np], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
        # Semi-transparent overlay
        overlay = display_img.copy()
        cv2.fillPoly(overlay, [pts_np], color=color)
        alpha = 0.15 if not is_selected else 0.28
        cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)

        # Center Point
        cx_i, cy_i = int(round(b.cx)), int(round(b.cy))
        cv2.circle(display_img, (cx_i, cy_i), 3, color, -1, cv2.LINE_AA)

        # Forward Direction Pointer (Top edge midpoint)
        _, tm = b.get_rotation_handle(offset=0)
        cv2.line(display_img, (cx_i, cy_i), (int(round(tm[0])), int(round(tm[1]))), (255, 255, 255), 1, cv2.LINE_AA)

        # Class and Angle Label
        label_text = f"{CLASS_NAMES[b.cls_id]} ({int(b.angle_deg)}°)"
        txt_pos_x = int(round(corners[0][0]))
        txt_pos_y = max(20, int(round(corners[0][1])) - 6)
        cv2.putText(display_img, label_text, (txt_pos_x, txt_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Draw Handles for Selected Box
        if is_selected:
            # Highlight border
            cv2.polylines(display_img, [pts_np], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # Rotation Handle (Yellow circle connected with line)
            rot_handle, top_mid = b.get_rotation_handle(offset=25)
            cv2.line(display_img, (int(round(top_mid[0])), int(round(top_mid[1]))),
                     (int(round(rot_handle[0])), int(round(rot_handle[1]))), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(display_img, (int(round(rot_handle[0])), int(round(rot_handle[1]))), 7, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(display_img, (int(round(rot_handle[0])), int(round(rot_handle[1]))), 8, (0, 0, 0), 1, cv2.LINE_AA)

            # Corner Handles (Squares)
            for cp in corners:
                cpx, cpy = int(round(cp[0])), int(round(cp[1]))
                cv2.rectangle(display_img, (cpx-4, cpy-4), (cpx+4, cpy+4), (255, 255, 255), -1)
                cv2.rectangle(display_img, (cpx-4, cpy-4), (cpx+4, cpy+4), (0, 0, 0), 1)

            # Edge Midpoint Handles (Circles)
            edge_mids = b.get_edge_midpoints()
            for ep in edge_mids:
                epx, epy = int(round(ep[0])), int(round(ep[1]))
                cv2.circle(display_img, (epx, epy), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(display_img, (epx, epy), 4, (0, 0, 0), 1, cv2.LINE_AA)

    # 2. Top Banner & HUD Information
    header_h = 42
    banner = np.zeros((header_h, img_w, 3), dtype=np.uint8)
    banner[:] = (35, 35, 35)

    # Dataset & Progress Info
    ds_badge = f"[{dataset_name.upper()}] [{img_idx+1}/{total_imgs}] {filename}"
    cv2.putText(banner, ds_badge, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Selected Box Info / Quick Help
    if selected_box_idx >= 0 and selected_box_idx < len(boxes):
        sel_b = boxes[selected_box_idx]
        sel_info = f"Selected: {CLASS_NAMES[sel_b.cls_id]} | Angle: {int(sel_b.angle_deg)}° | [0-3]:Change [Wheel]:Rotate [Del]:Remove"
        cv2.putText(banner, sel_info, (max(10, img_w - 700), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    else:
        hint_text = "Drag:Draw New | Click:Select | A:Prev D:Next T:Switch-Dataset Q:Quit"
        cv2.putText(banner, hint_text, (max(10, img_w - 630), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Combine Banner and Image
    final_canvas = np.vstack([banner, display_img])
    return final_canvas

def _suppress_win32_context_menu(window_name):
    """Win32：移除視窗右鍵系統選單（Restore/Move/Size...）；非 Windows 或找不到視窗就跳過。"""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            user32.GetSystemMenu(hwnd, False)
    except Exception:
        pass


def main():
    global current_dataset_idx, boxes, selected_box_idx, history_stack

    window_name = "YOLO OBB Smart Annotation & Review Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    _suppress_win32_context_menu(window_name)

    print("\n" + "="*55)
    print(" 🚀 YOLO OBB (Oriented Bounding Box) 旋轉標註工具已啟動")
    print("="*55)
    print("🎮 操作說明：")
    print("  • 左鍵點擊框     : 選取該框（顯示旋轉與縮放把手）")
    print("  • 拖曳黃色把手   : 自由旋轉角度")
    print("  • 滑鼠滾輪       : 旋轉角度微調（Shift=10°、Ctrl=0.5°）")
    print("  • 拖曳角落/邊緣  : 等比 / 長寬拉伸（保持 90 度矩形）")
    print("  • 拖曳框中心     : 移動框位置")
    print("  • 空白處拖曳     : 畫出新框")
    print("  • 數字鍵 0~3     : 快速切換選取框的類別 (0:塑膠, 1:金屬, 2:紙類, 3:一般)")
    print("  • Delete / X / 右鍵: 刪除選取框")
    print("  • A / D / Space  : 上一張 / 下一張（自動存檔）")
    print("  • T 鍵           : 切換 pool / train / val / test 資料集")
    print("  • Q / Esc        : 儲存並離開")
    print("="*55)

    img_idx = 0
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    is_first_scan = True

    while True:
        dataset_name = DATASETS[current_dataset_idx]
        images_dir = os.path.join(DATASET_ROOT, dataset_name, "images")
        labels_dir = os.path.join(DATASET_ROOT, dataset_name, "labels")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # 自動檢測並將新圖片標準化命名為 train_xxx / test_xxx
        auto_standardize_filenames(dataset_name)

        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        image_paths = sorted(image_paths)

        if not image_paths:
            print(f"⚠️ {dataset_name} 資料夾中沒有找到圖片。按 T 可切換資料夾。")
            blank = np.zeros((400, 700, 3), dtype=np.uint8)
            cv2.putText(blank, f"No images found in {images_dir}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(blank, "Press T to switch dataset, or Q to quit", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(window_name, blank)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('t'):
                current_dataset_idx = (current_dataset_idx + 1) % len(DATASETS)
                img_idx = 0
                is_first_scan = True
                continue
            else:
                break

        # 首次啟動或切換資料集時，自動跳轉至第一張「尚未標註」的新照片
        if is_first_scan:
            for idx_check, p_check in enumerate(image_paths):
                chk_name, _ = os.path.splitext(os.path.basename(p_check))
                chk_txt = os.path.join(labels_dir, chk_name + ".txt")
                if not os.path.exists(chk_txt) or os.path.getsize(chk_txt) == 0:
                    img_idx = idx_check
                    print(f"📍 自動定位至尚未標註的第一張新圖片: [第 {img_idx+1} / {len(image_paths)} 張] {os.path.basename(p_check)}")
                    break
            is_first_scan = False

        img_idx = max(0, min(img_idx, len(image_paths) - 1))
        img_path = image_paths[img_idx]
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        txt_path = os.path.join(labels_dir, name + ".txt")

        raw_img = cv2.imread(img_path)
        if raw_img is None:
            img_idx += 1
            continue

        h_raw, w_raw = raw_img.shape[:2]
        max_dim = 1000
        scale = 1.0
        if max(h_raw, w_raw) > max_dim:
            scale = max_dim / float(max(h_raw, w_raw))
            raw_img = cv2.resize(raw_img, (int(w_raw * scale), int(h_raw * scale)))

        h, w = raw_img.shape[:2]
        
        # Load Labels (Supports both HBB and OBB)
        boxes = load_labels(txt_path, w, h)
        selected_box_idx = -1
        history_stack.clear()

        while True:
            display_img = raw_img.copy()
            canvas = render_ui(display_img, w, h, img_idx, len(image_paths), filename, dataset_name)
            cv2.imshow(window_name, canvas)

            raw_key = cv2.waitKey(25)
            if raw_key != -1:
                key = raw_key & 0xFF

                # Key Actions
                if key in CLASSES:
                    if selected_box_idx >= 0 and selected_box_idx < len(boxes):
                        push_history()
                        boxes[selected_box_idx].cls_id = CLASSES[key][0]
                        print(f"🏷️ 標籤已修改為: {CLASSES[key][1]}")

                elif key in [ord('x'), ord('X'), 8, 127] or raw_key in [65535, 0xFFFF]: # 'x', Backspace, Delete
                    if selected_box_idx >= 0 and selected_box_idx < len(boxes):
                        push_history()
                        print(f"🗑️ 刪除標籤框: {CLASS_NAMES[boxes[selected_box_idx].cls_id]}")
                        boxes.pop(selected_box_idx)
                        selected_box_idx = -1

                elif key in [ord('t'), ord('T')]: # Switch Dataset between train and test
                    save_labels(txt_path, w, h)
                    current_dataset_idx = (current_dataset_idx + 1) % len(DATASETS)
                    img_idx = 0
                    print(f"📂 切換資料夾至: {DATASETS[current_dataset_idx]}")
                    break

                elif key in [ord('a'), ord('A'), 81]: # 'a' or Left Arrow -> Prev Image
                    save_labels(txt_path, w, h)
                    img_idx = max(0, img_idx - 1)
                    break

                elif key in [ord('d'), ord('D'), 13, 32, 83]: # 'd', Enter, Space, Right Arrow -> Next
                    save_labels(txt_path, w, h)
                    img_idx += 1
                    if img_idx >= len(image_paths):
                        print("🎉 當前資料夾所有圖片已審查完畢！")
                        img_idx = len(image_paths) - 1
                    break

                elif key in [ord('q'), ord('Q'), 27]: # 'q' or Esc -> Quit
                    save_labels(txt_path, w, h)
                    print("💾 所有標籤已儲存，退出標註工具。")
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
