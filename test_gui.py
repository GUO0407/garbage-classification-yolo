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
    # D415 相機內參預設值（1280x720）；要精確時以 ROS camera_info 為准，用下面幾支覆蓋
    parser.add_argument("--fx", type=float, default=920.0, help="相機內參 fx（像素）")
    parser.add_argument("--fy", type=float, default=920.0, help="相機內參 fy（像素）")
    parser.add_argument("--cx", type=float, default=None, help="主點 cx（像素，預設=影像寬/2）")
    parser.add_argument("--cy", type=float, default=None, help="主點 cy（像素，預設=影像高/2）")
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

def load_depth_mm(image_path: str) -> np.ndarray | None:
    """找同 stem 的原始 uint16 depth (mm)，供 hover 反投影使用。找法與 find_depth_for 一致（只取 .npy）。"""
    d = os.path.dirname(os.path.abspath(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if stem.endswith("_color"):
        stem = stem[: -len("_color")]
    npy_path = os.path.join(d, f"{stem}_depth.npy")
    if not os.path.exists(npy_path):
        return None
    depth_mm = np.load(npy_path)
    if isinstance(depth_mm, np.ndarray) and depth_mm.dtype == np.uint16:
        return depth_mm
    return None

def backproject_px(u: float, v: float, depth_mm: int | float, fx: float, fy: float, cx: float, cy: float) -> tuple[float, float, float]:
    """針孔反投影：像素 + 深度(mm) → 相機座標 (x, y, z)，单位 mm。公式與 robot_utils.pose_estimator 一致。"""
    z = float(depth_mm)
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)

def _median_patch_depth(depth_mm: np.ndarray, u: float, v: float, half: int = 8) -> int | None:
    """以 (u,v) 為中心、half 像素的方形 patch 內取中值深度；無有效點回傳 None。"""
    h, w = depth_mm.shape[:2]
    x0, y0 = max(int(u) - half, 0), max(int(v) - half, 0)
    x1, y1 = min(int(u) + half, w), min(int(v) + half, h)
    if x0 >= x1 or y0 >= y1:
        return None
    sub = depth_mm[y0:y1, x0:x1].ravel()
    sub = sub[sub > 0]
    if len(sub) == 0:
        return None
    return int(np.median(sub))

def _edge_between(pts: np.ndarray, i: int, j: int) -> tuple[float, float, float]:
    """回傳 OBB 相鄰角點 pts[i]→pts[j] 的像素向量 (dx, dy, 長度)。"""
    dx = float(pts[j][0] - pts[i][0])
    dy = float(pts[j][1] - pts[i][1])
    return (dx, dy, math.hypot(dx, dy))

def _short_edge_endpoints(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳 OBB 四邊形較短那一側「邊」的兩個端點（抓取側）。cv2.boxPoints 順序下相對邊為 (0,1)&(2,3)、(1,2)&(3,0)。"""
    (dx1, dy1, l1), (dx2, dy2, l2) = _edge_between(pts, 0, 1), _edge_between(pts, 1, 2)
    if l1 <= l2:
        return pts[0], pts[1]
    return pts[1], pts[2]

def box_metric_info(b: dict, depth_mm: np.ndarray | None, fx: float, fy: float, cx: float, cy: float) -> dict:
    """由一個 OBB detection 算出 metric 資訊：中心 3D 位置(mm)、角度(deg)、爪開尺寸(mm=短邊長)。
    無 depth 或全零時回傳 pixel domain 並標 units='px'。"""
    cx_px, cy_px, bw, bh, rad = b['xywhr']
    angle_deg = float(np.degrees(math.atan2(math.sin(rad), math.cos(rad))))
    short, long_ = min(bw, bh), max(bw, bh)
    if depth_mm is None:
        return {"units": "px", "center": (float(cx_px), float(cy_px)), "z": 0.0,
                "angle_deg": angle_deg, "grip_mm": None, "size_px": (float(short), float(long_))}

    h, w = depth_mm.shape[:2]
    if cx_px < 0 or cy_px < 0 or cx_px >= w or cy_px >= h:
        return {"units": "px", "center": (float(cx_px), float(cy_px)), "z": 0.0,
                "angle_deg": angle_deg, "grip_mm": None, "size_px": (float(short), float(long_))}

    d_center = _median_patch_depth(depth_mm, cx_px, cy_px)
    if d_center is None:
        return {"units": "px", "center": (float(cx_px), float(cy_px)), "z": 0.0,
                "angle_deg": angle_deg, "grip_mm": None, "size_px": (float(short), float(long_))}

    x_c, y_c, z_c = backproject_px(cx_px, cy_px, d_center, fx, fy, cx, cy)

    grip_mm = None
    grab_a, grab_b = _short_edge_endpoints(b['pts'])
    da = _median_patch_depth(depth_mm, float(grab_a[0]), float(grab_a[1]))
    db = _median_patch_depth(depth_mm, float(grab_b[0]), float(grab_b[1]))
    if da is not None and db is not None:
        p1 = backproject_px(float(grab_a[0]), float(grab_a[1]), da, fx, fy, cx, cy)
        p2 = backproject_px(float(grab_b[0]), float(grab_b[1]), db, fx, fy, cx, cy)
        grip_mm = float(math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]))

    return {"units": "mm", "center": (float(x_c), float(y_c)), "z": float(z_c),
            "angle_deg": angle_deg, "grip_mm": grip_mm, "size_px": (float(short), float(long_))}

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
        self.image_canvas.bind("<Motion>", self._on_image_motion)
        self.image_canvas.bind("<Leave>", lambda e: self._clear_hover())

        self.current_tk_image = None
        self.current_image_path = None
        self.raw_image_bgr = None
        self.current_depth_jet: np.ndarray | None = None

        # --- Hover-to-inspect 狀態 ---
        self._boxes_list: list = []
        self._metric_info: dict = {}
        self._canvas_geometry: dict | None = None    # scale/offset + combined dims
        self._hover_idx: int | None = None
        self._label_ids: tuple = ()                  # hover 標籤 (rect, text) canvas item ids
        self._label_size: tuple[int, int] = (0, 0)   # 標籤 rect (w, h)，供位移計算
        self._last_pointer: tuple | None = None      # 最後 cursor (canvas 座標)
        self._tip_ids: tuple = ()                    # depth probe 提示 (rect, text) ids
        self._tip_size: tuple[int, int] = (0, 0)
        self._base_combined = None                   # 無高亮的 combined BGR（hover 還原用）
        self._panel_layout: dict | None = None       # img_w / top_h / divider_h
        self._depth_mm_current: np.ndarray | None = None  # 目前圖的 raw depth（uint16 mm）

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

    def _intrinsics(self) -> tuple[float, float, float, float]:
        """回傳 (fx, fy, cx, cy)。cx/cy 未指定時取畫面中心（對 D415 已是合理近似）。"""
        w = self.raw_image_bgr.shape[1] if self.raw_image_bgr is not None else 1280
        h = self.raw_image_bgr.shape[0] if self.raw_image_bgr is not None else 720
        cx = args.cx if args.cx is not None else w / 2.0
        cy = args.cy if args.cy is not None else h / 2.0
        return (args.fx, args.fy, cx, cy)

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

        # 3b. Hover-to-inspect：存 raw depth + 偵測 + 算每框 metric（raw depth + intrinsics）
        self._boxes_list = boxes_list
        self._metric_info = {}
        self._depth_mm_current = None
        if self.current_image_path:
            dimg_h, dimg_w = self.raw_image_bgr.shape[:2]
            depth_mm = load_depth_mm(self.current_image_path)
            if depth_mm is not None and depth_mm.shape[:2] == (dimg_h, dimg_w):
                self._depth_mm_current = depth_mm
        if boxes_list:
            for i, b in enumerate(boxes_list):
                self._metric_info[i] = box_metric_info(b, self._depth_mm_current, *self._intrinsics())

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

    # --- Hover-to-inspect (continued in App methods below) ---
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
            self._panel_layout = {"img_w": img_w, "top_h": img_h, "divider_h": divider_h}
        else:
            combined = col
            self._panel_layout = {"img_w": img_w, "top_h": img_h, "divider_h": 0}
        self.show_image(combined)

    # --- Hover-to-inspect ---

    @staticmethod
    def _class_hex(cls_id: int, brightness: float = 1.0) -> str:
        """BGR class color → hex (brightness∈[0,1] 縮放亮度；向白混用 mix_to_white)"""
        b, g2, r = CLASS_COLORS.get(cls_id, (0, 255, 0))
        return f"#{int(r*brightness):02x}{int(g2*brightness):02x}{int(b*brightness):02x}"

    @staticmethod
    def _class_hex_tinted(cls_id: int, alpha: float = 0.45) -> str:
        """BGR class color 向白混 alpha → 模擬半透明填色（Tk canvas 無真 alpha）"""
        b, g2, r = CLASS_COLORS.get(cls_id, (0, 255, 0))
        return f"#{int(r*(1-alpha)+255*alpha):02x}{int(g2*(1-alpha)+255*alpha):02x}{int(b*(1-alpha)+255*alpha):02x}"

    def _src_xy(self, canvas_x: float, canvas_y: float):
        """canvas px → combined-image source px（可為負數）"""
        g = self._canvas_geometry
        return ((canvas_x - g["offset_x"]) / g["scale"],
                (canvas_y - g["offset_y"]) / g["scale"])

    def _row_layout(self):
        """回傳 (half_w, top_h, divider_h)：combined 單格寬高"""
        L = self._panel_layout or {}
        img_w = L.get("img_w", 1280)
        img_h = L.get("top_h", 720)
        return (img_w, img_h, L.get("divider_h", 0))

    def _on_image_motion(self, event):
        self._last_pointer = (event.x, event.y)
        g = self._canvas_geometry
        if not g or not self._panel_layout:
            self._clear_hover()
            return
        sx, sy = self._src_xy(event.x, event.y)
        half_w, top_h, dv = self._row_layout()
        canvas_xy = (event.x, event.y)

        in_left = 0 <= sx < half_w
        in_right = half_w <= sx < half_w * 2
        bottom = sy >= top_h + dv
        local_x = int(max(0, min((sx if in_left else sx - half_w), half_w - 1)))
        local_y = int(max(0, min(sy if not bottom else sy - top_h - dv, top_h - 1)))

        # --- Box hover：右側 panel（OBB detection / Depth+OBB）內命中偵測框 ---
        idx = None
        if self._boxes_list and in_right:
            for i in reversed(range(len(self._boxes_list))):
                if cv2.pointPolygonTest(
                    self._boxes_list[i]['pts'].astype(np.float32).reshape((-1, 1, 2)),
                    (float(local_x), float(local_y)), False) >= 0:
                    idx = i
                    break

        # --- Depth probe：左下（全格）與右下（框外位置）顯示該點 depth ---
        if (idx is None and bottom and self._depth_mm_current is not None
                and (in_left or in_right)):
            mm = float(self._depth_mm_current[local_y, local_x])
            self._set_hover(None, canvas_xy)
            self._show_probe(f"{mm:.0f} mm" if mm > 0 else "no depth", canvas_xy)
            return

        self._set_hover(idx, canvas_xy)

    def _src_to_canvas(self, sx: float, sy: float, row: int, col: int) -> tuple:
        """單格 (sx, sy) → canvas px；row 0/1 上/下排，col 0/1 左/右側"""
        g = self._canvas_geometry
        half_w, top_h, dv = self._row_layout()
        ox_src = sx + col * half_w
        oy_src = sy + (0.0 if row == 0 else (top_h + dv))
        return (g["offset_x"] + ox_src * g["scale"],
                g["offset_y"] + oy_src * g["scale"])

    def _set_hover(self, idx, canvas_xy):
        if idx is not None and idx == self._hover_idx:
            self._position_label(canvas_xy)
            return
        self._clear_hover()
        if idx is None:
            self._update_status(None)
            return
        self._draw_hover_overlay(idx, canvas_xy)

    def _draw_hover_overlay(self, idx, canvas_xy):
        g = self._canvas_geometry
        b = self._boxes_list[idx]
        cls_id = b['cls']
        fill_col = self._class_hex_tinted(cls_id)
        bright_col = self._class_hex(cls_id)
        # 同一組偵測框同時存在於右上 (OBB) 與右下 (Depth+OBB)；兩排各畫一次高亮
        for row in (0, 1):
            poly = []
            for x, y in b['pts']:
                cx_, cy_ = self._src_to_canvas(float(x), float(y), row=row, col=1)
                poly.extend((cx_, cy_))
            self.image_canvas.create_polygon(
                poly, outline=bright_col, width=max(3, int(4 * g["scale"])),
                fill=fill_col, tags="hovpoly")
        cx_px, cy_px = map(int, b['xywhr'][:2])
        r = max(4, 6 * g["scale"])
        ex, ey = self._src_to_canvas(cx_px, cy_px, row=0, col=1)
        self.image_canvas.create_oval(ex - r, ey - r, ex + r, ey + r,
                                      outline=bright_col, fill=bright_col, width=1, tags="hovpoly")

        label = self._build_label(idx)
        font_size = max(14, int(g["scale"] * 24))
        font = ("DejaVu Sans Mono", font_size)
        rect = self.image_canvas.create_rectangle(0, 0, 0, 0,
                                                  fill="black", stipple="gray50",
                                                  outline=bright_col, width=1, tags="hovbg")
        text = self.image_canvas.create_text(0, 0, text=label, anchor="nw",
                                             font=font, fill="white", tags="hoverlbl")
        self._label_ids = (rect, text)
        self._position_label(canvas_xy)
        self._hover_idx = idx
        self._update_status(idx)

    def _build_label(self, idx: int) -> str:
        b = self._boxes_list[idx]
        m = self._metric_info.get(idx, {})
        units = m.get("units", "px")
        c = m.get("center", (0.0, 0.0))
        lines = [f"{b['name']}  {b['conf']:.2f}   angle={m.get('angle_deg', 0):.1f} deg"]
        if units == "mm":
            lines.append(f"pos cam frame: x={c[0]/10:+6.1f} y={c[1]/10:+6.1f} z={m['z']/10:5.1f} cm")
            gm = m.get("grip_mm")
            if gm is not None:
                lines.append(f"grip  {gm:5.1f} mm")
            else:
                short = m.get("size_px", (0.0, 0.0))[0]
                lines.append(f"grip  n/a  (short edge {short:.0f} px)")
        else:
            lines.append(f"pos   u={c[0]:6.1f}  v={c[1]:6.1f}  px")
            short = m.get("size_px", (0.0, 0.0))[0]
            lines.append(f"grip  {short:5.1f} px  (no depth)")
        return "\n".join(lines)

    def _position_label(self, canvas_xy):
        if not self._label_ids:
            return
        rect, text_id = self._label_ids
        g = self._canvas_geometry
        # 讓 canvas 先量出真實文字尺寸（rect 稍後依文字 bbox 重定位）
        self.image_canvas.coords(text_id, 0, 0)
        self.update_idletasks()
        bb_box = self.image_canvas.bbox(text_id) or (0, 0, 200, 80)
        pad_x, pad_y = 8, 6
        w = (bb_box[2] - bb_box[0]) + pad_x * 2
        h = (bb_box[3] - bb_box[1]) + pad_y * 2
        x, y = canvas_xy[0] + 18, canvas_xy[1] + 18
        if g:
            if x + w > g["canvas_w"] - 4:
                x = max(canvas_xy[0] - w - 6, 2)
            if y + h > g["canvas_h"] - 4:
                y = max(canvas_xy[1] - h - 6, 2)
        # Update rect to match text bounds
        self.image_canvas.coords(text_id, x + pad_x, y + pad_y)
        bb_after = self.image_canvas.bbox(text_id)
        if bb_after:
            rx0, ry0, rx1, ry1 = bb_after
            self.image_canvas.coords(
                rect,
                rx0 - pad_x, ry0 - pad_y,
                rx1 + pad_x, ry1 + pad_y)
        self.image_canvas.tag_raise(text_id)
        self._label_size = (w, h)

    def _show_probe(self, text: str, canvas_xy):
        g = self._canvas_geometry
        font = ("DejaVu Sans Mono", max(12, int(g["scale"] * 18)))
        pad_x, pad_y = 7, 4
        if not self._tip_ids:
            txt_id = self.image_canvas.create_text(0, 0, anchor="nw",
                                                   font=font, fill="#ffe45e", tags="probelbl")
            rect = self.image_canvas.create_rectangle(0, 0, 0, 0, fill="black",
                                                      stipple="gray50", outline="#ffe45e",
                                                      width=1, tags="probelbl")
            self._tip_ids = (rect, txt_id)
        rect, txt_id = self._tip_ids
        self.image_canvas.itemconfigure(txt_id, text=text, font=font)
        pad_x, pad_y = 7, 4
        self.image_canvas.coords(txt_id, canvas_xy[0] + 14, canvas_xy[1] - 6)
        bb = self.image_canvas.bbox(txt_id)
        if bb:
            self.image_canvas.coords(
                rect, bb[0] - pad_x, bb[1] - pad_y, bb[2] + pad_x, bb[3] + pad_y)
        self.image_canvas.tag_raise(txt_id)

    def _clear_hover(self):
        for tag in ("hoverlbl", "hovpoly", "hovbg", "probelbl"):
            for iid in self.image_canvas.find_withtag(tag):
                self.image_canvas.delete(iid)
        self._label_ids = ()
        self._label_size = (0, 0)
        self._tip_ids = ()
        self._hover_idx = None

    def _update_status(self, idx):
        if idx is None or not self._boxes_list:
            return

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
        img_id = self.image_canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            anchor="center",
            image=self.current_tk_image
        )

        self._canvas_geometry = {
            "img": self.current_tk_image,
            "scale": ratio,
            "offset_x": (canvas_w - new_w) / 2.0,
            "offset_y": (canvas_h - new_h) / 2.0,
            "combined_w": img_w,
            "combined_h": img_h,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
        }
        # resize 會把先前 overlay 打掉，重畫一次當前 hover（沿用最後 cursor 位置）
        hovered_idx, hover_xy = self._hover_idx, self._last_pointer
        self._clear_hover()
        if hovered_idx is not None and hover_xy is not None:
            self._draw_hover_overlay(hovered_idx, hover_xy)

if __name__ == "__main__":
    app = App()
    app.mainloop()
