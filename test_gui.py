import os
import random
import glob
import math
import tkinter as tk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Garbage Classification Inference Engine")
    parser.add_argument("--path", type=str, default="custom_dataset/test/images", help="Path to the test images directory")
    return parser.parse_args()

args = parse_args()

# Setup CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Config Paths
MODEL_PATH = "best.pt"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "custom_dataset")


def list_image_folders(dataset_dir: str) -> list[str]:
    """列出資料夾中所有「含有圖片」的目錄，依路徑排序（含根目錄本身若含圖）"""
    folders = []
    if os.path.isdir(dataset_dir):
        root_has_img = any(glob.glob(os.path.join(dataset_dir, ext)) for ext in IMAGE_EXTS)
        if root_has_img:
            folders.append(dataset_dir)
        for dirpath, _dirnames, filenames in os.walk(dataset_dir):
            rel = os.path.relpath(dirpath, dataset_dir)
            if rel == "." or "demo_backup" in rel.split(os.sep):
                continue
            if any(glob.glob(os.path.join(dirpath, ext)) for ext in IMAGE_EXTS):
                folders.append(dirpath)
    folders.sort()
    return folders


IMAGE_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.webp"]

# BGR Colors for OpenCV rendering
CLASS_COLORS = {
    0: (46, 204, 113),   # plastic - 翠綠色
    1: (235, 152, 52),   # metal - 藍青色
    2: (34, 126, 230),   # paper - 亮橘色
    3: (255, 0, 255)     # general_waste - 螢光洋紅/亮紫紅 (超高對比，極清晰)
}

def obb_iou(pts1, pts2):
    """計算兩個任意四邊形/旋轉邊界框之間的精確 IoU"""
    p1 = np.ascontiguousarray(pts1, dtype=np.float32)
    p2 = np.ascontiguousarray(pts2, dtype=np.float32)
    area1 = cv2.contourArea(p1)
    area2 = cv2.contourArea(p2)
    
    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def depth_to_jet_display(depth_mm: np.ndarray) -> np.ndarray | None:
    """uint16 mm -> JET BGR uint8，0=黑色無效區；percentile + 2m cap 自適應刻度"""
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


def apply_agnostic_nms(boxes_list, iou_thresh=0.40):
    """跨類別 NMS：若多個不同類別的框重疊，只保留最高信心度的那一個"""
    if len(boxes_list) <= 1:
        return boxes_list
        
    boxes_list = sorted(boxes_list, key=lambda x: x['conf'], reverse=True)
    kept = []
    
    while boxes_list:
        best = boxes_list.pop(0)
        kept.append(best)
        boxes_list = [b for b in boxes_list if obb_iou(best['pts'], b['pts']) < iou_thresh]
        
    return kept

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Garbage Classification Inference Engine (YOLOv11-OBB)")
        self.geometry("1420x780")
        
        # Configure Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(12, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Garbage Vision", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 5))

        self.model_status_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Loading Model...", 
            text_color="gray70",
            font=ctk.CTkFont(size=12)
        )
        self.model_status_label.grid(row=1, column=0, padx=20, pady=(0, 15))

        # Dataset folder dropdown (相對路徑顯示，abspath 存在 dict)
        self._folder_display_to_path: dict[str, str] = {
            os.path.relpath(f, REPO_ROOT): f for f in list_image_folders(DEFAULT_DATASET_DIR)
        }
        if not any(os.path.realpath(p) == os.path.realpath(args.path) for p in self._folder_display_to_path.values()):
            if os.path.isdir(args.path):
                self._folder_display_to_path[os.path.relpath(args.path, REPO_ROOT)] = args.path
        self.folder_dropdown = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=sorted(self._folder_display_to_path.keys()),
            command=self.on_folder_change,
            font=ctk.CTkFont(size=12),
        )
        # 預設選 --path 指向的資料夾；否則選 custom_dataset 根
        default_display = (
            os.path.relpath(args.path, REPO_ROOT)
            if os.path.isdir(args.path) and os.path.relpath(args.path, REPO_ROOT) in self._folder_display_to_path
            else "custom_dataset"
        )
        self.folder_dropdown.set(default_display)
        self.current_image_dir: str = self._folder_display_to_path.get(
            default_display, DEFAULT_DATASET_DIR
        )
        self.folder_dropdown.grid(row=2, column=0, padx=20, pady=(0, 12), sticky="ew")

        # Confidence Slider
        self.conf_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Confidence: 0.60", 
            anchor="w"
        )
        self.conf_label.grid(row=3, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.conf_slider = ctk.CTkSlider(
            self.sidebar_frame, 
            from_=0.05, 
            to=0.95, 
            number_of_steps=90, 
            command=self.update_sliders
        )
        self.conf_slider.set(0.60)
        self.conf_slider.grid(row=4, column=0, padx=20, pady=(5, 10))

        # NMS IoU Slider
        self.iou_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="Agnostic NMS IoU: 0.40", 
            anchor="w"
        )
        self.iou_label.grid(row=5, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.iou_slider = ctk.CTkSlider(
            self.sidebar_frame, 
            from_=0.10, 
            to=0.90, 
            number_of_steps=80, 
            command=self.update_sliders
        )
        self.iou_slider.set(0.40)
        self.iou_slider.grid(row=6, column=0, padx=20, pady=(5, 15))

        # Agnostic NMS Switch (跨類別抑制)
        self.agnostic_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="Agnostic NMS",
            command=self.run_inference
        )
        self.agnostic_switch.select()
        self.agnostic_switch.grid(row=7, column=0, padx=20, pady=(5, 10), sticky="w")

        # Grasp Pose Switch (預設開啟)
        self.grasp_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="Grasp Pose",
            command=self.run_inference
        )
        self.grasp_switch.select()
        self.grasp_switch.grid(row=8, column=0, padx=20, pady=(5, 15), sticky="w")

        self.load_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Load Random Image", 
            command=self.load_random_image,
            height=36,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.load_btn.grid(row=9, column=0, padx=20, pady=(5, 5))

        self.select_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="Select Image File", 
            command=self.select_image_file,
            height=36,
            fg_color="#3a7ebf",
            hover_color="#326ba3",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.select_btn.grid(row=10, column=0, padx=20, pady=(5, 10))

        # Legend Box
        self.legend_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.legend_frame.grid(row=11, column=0, padx=20, pady=10, sticky="w")
        
        legend_items = [
            ("Plastic", "#2ecc71"),
            ("Metal", "#3498db"),
            ("Paper", "#e67e22"),
            ("General Waste", "#ff00ff")
        ]
        for idx, (c_name, c_hex) in enumerate(legend_items):
            lbl = ctk.CTkLabel(
                self.legend_frame, 
                text=f"■ {c_name}", 
                text_color=c_hex, 
                font=ctk.CTkFont(size=12, weight="bold")
            )
            lbl.grid(row=idx, column=0, sticky="w", pady=2)

        # --- Main View Frame ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(
            self.main_frame, 
            bg="#1a1a1a", 
            highlightthickness=0
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.image_canvas.bind("<Configure>", self.on_canvas_resize)

        self.current_tk_image = None
        self.current_image_path = None
        self.raw_image_bgr = None
        self.current_depth_jet: np.ndarray | None = None

        # Load the custom trained model
        self.update()
        try:
            model_to_load = MODEL_PATH
            if not os.path.exists(model_to_load):
                alt_weights = [
                    "garbage_classification_runs/yolo11x_obb_model/weights/best.pt",
                    "yolo11x-obb.pt"
                ]
                for alt in alt_weights:
                    if os.path.exists(alt):
                        model_to_load = alt
                        break

            self.model = YOLO(model_to_load)
            self.model_status_label.configure(
                text=f"Ready: {os.path.basename(model_to_load)}", 
                text_color="#28a745"
            )
            self.load_random_image()
        except Exception as e:
            self.model_status_label.configure(text=f"Load Failed: {str(e)[:20]}", text_color="#dc3545")

    def update_sliders(self, value=None):
        conf_val = self.conf_slider.get()
        iou_val = self.iou_slider.get()
        self.conf_label.configure(text=f"Confidence: {conf_val:.2f}")
        self.iou_label.configure(text=f"Agnostic NMS IoU: {iou_val:.2f}")
        
        if self.current_image_path and self.raw_image_bgr is not None:
            self.run_inference()

    def on_folder_change(self, display_name):
        self.current_image_dir = self._folder_display_to_path.get(display_name, DEFAULT_DATASET_DIR)
        self.load_random_image()

    def find_depth_for(self, image_path: str) -> np.ndarray | None:
        """找同 stem 的 depth：優先 {stem}_depth.npy（uint16 mm），回退 {stem}_depth_jet.png。
        支援 capture_dataset.py 的命名（NNNN_color.png 配 NNNN_depth.{npy,png}_jet）"""
        d = os.path.dirname(os.path.abspath(image_path))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        candidates = [stem]
        if stem.endswith("_color"):
            candidates.insert(0, stem[: -len("_color")])
        for base in candidates:
            npy_path = os.path.join(d, f"{base}_depth.npy")
            if os.path.exists(npy_path):
                depth_mm = np.load(npy_path)
                if isinstance(depth_mm, np.ndarray) and depth_mm.dtype == np.uint16:
                    return depth_to_jet_display(depth_mm)
                continue
            jet_path = os.path.join(d, f"{base}_depth_jet.png")
            if os.path.exists(jet_path):
                depth_jet = cv2.imread(jet_path, cv2.IMREAD_COLOR)
                if depth_jet is not None:
                    return depth_jet
        return None

    def load_image_from_path(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return
            
        self.current_image_path = file_path
        filename = os.path.basename(self.current_image_path)
        
        if len(filename) > 22:
            display_name = filename[:19] + "..."
        else:
            display_name = filename
            
        self.model_status_label.configure(text=f"Image: {display_name}", text_color="gray70")
        
        # Read image supporting UTF-8 / non-ASCII paths
        try:
            img_array = np.fromfile(self.current_image_path, dtype=np.uint8)
            self.raw_image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            self.raw_image_bgr = None

        if self.raw_image_bgr is None:
            self.raw_image_bgr = cv2.imread(self.current_image_path)

        if self.raw_image_bgr is None:
            self.model_status_label.configure(text="Decode Error", text_color="#dc3545")
            return

        self.current_depth_jet = self.find_depth_for(file_path)
        self.run_inference()

    def load_random_image(self):
        images = []
        for ext in IMAGE_EXTS:
            images.extend(glob.glob(os.path.join(self.current_image_dir, ext)))
            images.extend(glob.glob(os.path.join(self.current_image_dir, ext.upper())))
        # 深度預覽圖不是資料圖，排除掉
        images = [p for p in images if not os.path.basename(p).endswith("_depth_jet.png")]

        if not images:
            self.model_status_label.configure(
                text=f"No images in {os.path.relpath(self.current_image_dir, REPO_ROOT)}",
                text_color="#dc3545",
            )
            return

        selected = random.choice(images)
        self.load_image_from_path(selected)

    def select_image_file(self):
        initial_dir = self.current_image_dir if os.path.isdir(self.current_image_dir) else os.getcwd()
        filetypes = [
            ("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp *.JPG *.JPEG *.PNG *.WEBP *.BMP"),
            ("All Files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="選擇要推論的圖片檔案",
            initialdir=initial_dir,
            filetypes=filetypes
        )
        if file_path:
            self.load_image_from_path(file_path)

    def run_inference(self):
        if not hasattr(self, 'model') or self.raw_image_bgr is None:
            return
            
        conf_thresh = self.conf_slider.get()
        iou_thresh = self.iou_slider.get()
        use_agnostic_nms = self.agnostic_switch.get() == 1
        show_grasp = self.grasp_switch.get() == 1
        
        # 1. Model inference
        results = self.model(self.raw_image_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]
        
        # 2. Extract OBB detections
        boxes_list = []
        if hasattr(results, 'obb') and results.obb is not None and len(results.obb) > 0:
            xyxyxyxy = results.obb.xyxyxyxy.cpu().numpy()
            xywhr = results.obb.xywhr.cpu().numpy()
            confs = results.obb.conf.cpu().numpy()
            clss = results.obb.cls.cpu().numpy().astype(int)
            
            for i in range(len(confs)):
                boxes_list.append({
                    'cls': clss[i],
                    'name': self.model.names[clss[i]],
                    'conf': float(confs[i]),
                    'pts': xyxyxyxy[i],
                    'xywhr': xywhr[i]
                })

        # 3. Apply Class-Agnostic NMS (消除跨類別重複框)
        if use_agnostic_nms and len(boxes_list) > 1:
            boxes_list = apply_agnostic_nms(boxes_list, iou_thresh=iou_thresh)

        # 4. Clean & Minimalist Rendering
        img_h, img_w = self.raw_image_bgr.shape[:2]
        annotated = self._draw_boxes(self.raw_image_bgr.copy(), boxes_list, show_grasp)
        depth_panel = None
        if (
            self.current_depth_jet is not None
            and self.current_depth_jet.shape[:2] == (img_h, img_w)
        ):
            depth_panel = self._draw_boxes(self.current_depth_jet.copy(), boxes_list, show_grasp)

        self.render_panels(annotated, boxes_list, depth_panel)

    def _draw_boxes(self, annotated: np.ndarray, boxes_list, show_grasp: bool) -> np.ndarray:
        """在給定圖片上畫 OBB 框 + badge（+Grasp Pose）；回傳同一張圖。RGB 與 JET 深度共用"""
        img_h, img_w = annotated.shape[:2]

        # Adaptive font scale and line thickness based on image resolution
        font_scale = max(0.85, min(1.4, img_w / 900.0))
        font_thick = max(2, int(round(font_scale * 2.0)))
        line_thick = max(2, int(round(img_w / 450.0)))

        for b in boxes_list:
            cls_id = b['cls']
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))
            pts = b['pts'].astype(np.int32).reshape((-1, 1, 2))
            
            # Clean oriented bounding box border
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=line_thick, lineType=cv2.LINE_AA)
            
            # Top-left corner point for badge
            top_pt = b['pts'][np.argmin(b['pts'][:, 1])]
            bx, by = int(top_pt[0]), int(top_pt[1])
            
            # Badge text: e.g. "plastic 0.91"
            badge_text = f" {b['name']} {b['conf']:.2f} "
            (tw, th), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            
            # Draw badge background with neat padding
            pad_y = int(th * 0.35)
            badge_y1 = max(0, by - th - pad_y * 2)
            badge_y2 = badge_y1 + th + pad_y * 2
            badge_x1 = max(0, bx)
            badge_x2 = min(annotated.shape[1], badge_x1 + tw)
            
            # Draw solid badge background with slightly darker inner border
            # cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
            # cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 0, 0), 1)
            
            # 繪製醒目白色描邊（8 方向擴展純白外框，確保在任何背景皆有清晰白邊）
            text_x = max(4, badge_x1)
            text_y = badge_y2 - pad_y
            outline_radius = max(2, int(round(font_scale * 2.5)))
            
            # 先以多重偏移繪製純白實心描邊
            for dx in range(-outline_radius, outline_radius + 1):
                for dy in range(-outline_radius, outline_radius + 1):
                    if dx != 0 or dy != 0:
                        cv2.putText(annotated, badge_text, (text_x + dx, text_y + dy),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick + 1, cv2.LINE_AA)
            
            # 再於中心繪製類別專屬顏色文字
            cv2.putText(annotated, badge_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick, cv2.LINE_AA)

            # Optional Grasp Pose Overlay
            if show_grasp:
                cx, cy, bw, bh, rad = b['xywhr']
                grasp_len = max(bw, bh) * 0.5
                dx = (grasp_len / 2.0) * math.cos(rad + math.pi / 2.0)
                dy = (grasp_len / 2.0) * math.sin(rad + math.pi / 2.0)
                p1 = (int(cx - dx), int(cy - dy))
                p2 = (int(cx + dx), int(cy + dy))
                cv2.line(annotated, p1, p2, (255, 255, 0), max(2, line_thick), cv2.LINE_AA)
                dx2 = (grasp_len / 3.0) * math.cos(rad)
                dy2 = (grasp_len / 3.0) * math.sin(rad)
                p3 = (int(cx - dx2), int(cy - dy2))
                p4 = (int(cx + dx2), int(cy + dy2))
                cv2.line(annotated, p3, p4, (0, 255, 255), max(2, line_thick), cv2.LINE_AA)
                cv2.circle(annotated, (int(cx), int(cy)), max(4, line_thick + 2), (0, 0, 255), -1, cv2.LINE_AA)

        return annotated

    def render_panels(self, annotated: np.ndarray, boxes_list, depth_panel: np.ndarray | None):
        # 5. Panels (2x2): 左上 Original | 右上 OBB Detection
        #                  左下 Depth (raw) | 右下 Depth + OBB（無 depth 時退化成上方 1x2）
        img_h, img_w = annotated.shape[:2]
        tag_scale = max(0.7, min(1.2, img_w / 1000.0))
        tag_thick = max(2, int(round(tag_scale * 2.0)))

        def _tag(canvas: np.ndarray, text: str, color: tuple[int, int, int]):
            pos = (18, int(35 * tag_scale + 5))
            cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (0, 0, 0), tag_thick + 3, cv2.LINE_AA)
            cv2.putText(canvas, text, pos, cv2.FONT_HERSHEY_SIMPLEX, tag_scale, color, tag_thick, cv2.LINE_AA)

        raw_display = self.raw_image_bgr.copy()
        _tag(raw_display, "Original", (255, 255, 255))
        _tag(annotated, f"OBB Detection ({len(boxes_list)} objects)", (0, 255, 255))

        col = np.hstack([raw_display, annotated])
        if depth_panel is not None:
            depth_raw = self.current_depth_jet.copy()
            _tag(depth_raw, "Depth", (255, 255, 0))
            _tag(depth_panel, "Depth + OBB", (255, 255, 0))
            col_depth = np.hstack([depth_raw, depth_panel])
            divider_h = max(3, int(img_h / 300.0))
            divider = np.zeros((divider_h, img_w * 2, 3), dtype=np.uint8)
            divider[:] = (45, 45, 45)
            combined = np.vstack([col, divider, col_depth])
        else:
            combined = col
        self.show_image(combined)

    def on_canvas_resize(self, event):
        if self.current_tk_image is not None and hasattr(self, 'last_annotated_img_rgb'):
            self.draw_on_canvas(self.last_annotated_img_rgb, event.width, event.height)

    def show_image(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.last_annotated_img_rgb = img_rgb
        
        self.update_idletasks()
        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()
        
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600
            
        self.draw_on_canvas(img_rgb, canvas_w, canvas_h)

    def draw_on_canvas(self, img_rgb, canvas_w, canvas_h):
        img_h, img_w = img_rgb.shape[:2]
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(int(img_w * ratio), 1)
        new_h = max(int(img_h * ratio), 1)
        
        resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(resized)
        
        self.current_tk_image = ImageTk.PhotoImage(pil_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_w // 2, 
            canvas_h // 2, 
            anchor="center", 
            image=self.current_tk_image
        )

if __name__ == "__main__":
    app = App()
    app.mainloop()
