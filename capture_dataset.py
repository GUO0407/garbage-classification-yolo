"""RealSense 拍照工具：一鍵存下 RGB + aligned depth 配對幀（離線資料累積）。

相機來自 dual_amm_arm_core.launch.py（align_depth.enable: True）：
  /arm/camera_{side}/realsense_camera_{side}/color/image_raw                    rgb8
  /arm/camera_{side}/realsense_camera_{side}/aligned_depth_to_color/image_raw   16UC1 (mm, 0=無效)

輸出（自動遞增編號、不覆蓋）：
  NNNN_color.png         原始 RGB
  NNNN_depth.npy         原始 uint16 mm（零損耗，後續演算用）
  NNNN_depth_jet.png     JET 熱圖預覽（人眼檢查用）

使用（相機已起的狀態下，於 yolo repo 目錄內）。
注意：要用**系統 python**（.venv 沒灌 rclpy），且必 source ROS：
  source /opt/ros/humble/setup.bash
  python3 capture_dataset.py --side left
  # 每次 Enter 拍一張；'q'+Enter 結束；Ctrl-C 也結束

存檔路徑預設 custom_dataset/demo/captures（test_gui dropdown 會自動出現該資料夾）。
"""

import argparse
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def color_msg_to_bgr(msg: Image) -> np.ndarray | None:
    """rgb8 -> BGR ndarray（不上 cv_bridge，§10 雷點）"""
    if msg.encoding != "rgb8" or not msg.data:
        return None
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def depth_msg_to_mm(msg: Image) -> np.ndarray | None:
    """16UC1 aligned depth -> uint16 mm ndarray（0 = 無效點）"""
    if msg.encoding != "16UC1" or not msg.data:
        return None
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)


def depth_to_jet(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 mm -> JET BGR uint8，0=黑色無效區；percentile + 2m cap 自適應刻度"""
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        return np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, (2, 98))
    hi = max(float(hi), 2000.0)
    clipped = np.clip(depth_mm, lo, hi).astype(np.float32)
    norm8 = ((clipped - lo) / (hi - lo + 1e-6)).astype(np.float32) * 255.0
    jet = cv2.applyColorMap(norm8.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(jet, cv2.COLOR_RGB2BGR)


def next_index(save_dir: str) -> int:
    max_idx = 0
    for f in os.listdir(save_dir):
        stem, ext = os.path.splitext(f)
        if not ext.lower() in (".png", ".jpg", ".jpeg"):
            continue
        prefix = stem.split("_color")[0]
        if "_color" in stem and prefix.isdigit():
            max_idx = max(max_idx, int(prefix))
    return max_idx + 1


def wait_for_pair(node: Node, timeout_sec: float) -> tuple[Image | None, Image | None]:
    """spin 直到 color 與 aligned depth 各收到 ≥1 幀，或逾時。

    Realsense2 兩 topic 同頻率同時間戳對齊（align_depth），先到者等後到者
    即為同一物理幀；以「各自最新一幀」即可。
    """
    color: Image | None = None
    depth: Image | None = None

    def on_color(msg: Image):
        nonlocal color
        color = msg

    def on_depth(msg: Image):
        nonlocal depth
        depth = msg

    client_color = node.create_subscription(Image, "color/image_raw", on_color, 1)
    client_depth = node.create_subscription(
        Image, "aligned_depth_to_color/image_raw", on_depth, 1
    )
    try:
        start = time.monotonic()
        while (color is None or depth is None) and time.monotonic() - start < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
        return color, depth
    finally:
        node.destroy_subscription(client_color)
        node.destroy_subscription(client_depth)


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense RGB + aligned depth capture")
    parser.add_argument("--side", type=str, choices=["left", "right"], required=True)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "custom_dataset", "demo", "captures"
        ),
    )
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="等待兩路同幀的逾時秒數（超時提示重試）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    rclpy.init(args=None)
    node = Node(
        f"capture_dataset_{args.side}",
        namespace=f"/arm/camera_{args.side}/realsense_camera_{args.side}",
    )

    idx = 0
    print(f"[capture_dataset] side={args.side} save={save_dir}")
    print("Enter=拍一張 (RGB+depth 配對), 'q'+Enter=結束, Ctrl-C=結束")
    try:
        while True:
            user_input = input("> ")
            if user_input.strip().lower() == "q":
                break
            color_msg, depth_msg = wait_for_pair(node, args.timeout)
            bgr = color_msg_to_bgr(color_msg) if color_msg is not None else None
            depth_mm = depth_msg_to_mm(depth_msg) if depth_msg is not None else None

            if bgr is None or depth_mm is None:
                missing = []
                if bgr is None:
                    missing.append("color")
                if depth_mm is None:
                    missing.append("depth")
                print(f"  [skip] {args.timeout:.1f}s 內未收齊: {'+'.join(missing)}（再按 Enter 重試）")
                continue

            if bgr.shape[:2] != depth_mm.shape:
                print(f"  [warn] 尺寸不一致 rgb={bgr.shape[:2]} depth={depth_mm.shape}，仍存但需檢查對齊")

            idx = next_index(save_dir)
            prefix = f"{idx:04d}"
            color_path = os.path.join(save_dir, f"{prefix}_color.png")
            depth_npy_path = os.path.join(save_dir, f"{prefix}_depth.npy")
            jet_path = os.path.join(save_dir, f"{prefix}_depth_jet.png")

            ok, buf = cv2.imencode(".png", bgr)
            if ok:
                buf.tofile(color_path)
            np.save(depth_npy_path, depth_mm)
            ok, buf = cv2.imencode(".png", depth_to_jet(depth_mm))
            if ok:
                buf.tofile(jet_path)
            print(f"  [saved] {color_path}\n          {depth_npy_path}\n          {jet_path}")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f"[capture_dataset] done, {idx} frames total in {save_dir}")


if __name__ == "__main__":
    main()
