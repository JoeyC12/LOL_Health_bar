import os
import cv2
import numpy as np
from pathlib import Path
import re
from paddleocr import PaddleOCR


# =========================================================
# 边缘检测：定位血量条位置
# =========================================================
def detect_hp_bar_by_edge(img_bgr, output_dir="./hp_edge_steps"):
    """通过边缘检测定位血量条位置"""
    print("[Health] Detecting HP bar by edge")
    H, W = img_bgr.shape[:2]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 提取底部中央区域
    roi_y1 = H * 4 // 5
    roi_y2 = H
    roi_x1 = W // 4
    roi_x2 = W * 3 // 4

    roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    # cv2.imwrite(str(out_dir / "step0_bottom_center_roi.png"), roi)

    # 灰度化与模糊
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imwrite(str(out_dir / "step1_gray.png"), gray)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    # cv2.imwrite(str(out_dir / "step2_edges.png"), edges)

    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imwrite(str(out_dir / "step3_edges_closed.png"), edges_closed)

    # 查找轮廓
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 绘制所有候选框
    roi_with_candidates = roi.copy()
    best = None
    best_width = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 4:
            continue
        cv2.rectangle(roi_with_candidates, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if w > best_width:
            best_width = w
            best = (x, y, w, h)
    
    cv2.imwrite(str(out_dir / "step4_all_candidates.png"), roi_with_candidates)

    if best is None:
        return None, None

    x, y, w, h = best
    # 绘制最终选中的bbox
    roi_final = roi.copy()
    cv2.rectangle(roi_final, (x, y), (x + w, y + h), (0, 0, 255), 3)
    step4_path = str(out_dir / "step4_final_bbox.png")
    cv2.imwrite(step4_path, roi_final)
    
    # 返回全图坐标的 bbox 和 step4 图片路径
    return (x + roi_x1, y + roi_y1, x + w + roi_x1, y + h + roi_y1), step4_path


# =========================================================
# OCR 识别：从图片中提取血量文本
# =========================================================
def extract_health_text_from_roi(roi, debug=False):
    """
    从 ROI 区域进行 OCR 识别，提取血量文本
    返回 (health_text, health_bbox_roi) 或 (None, None)
    """
    ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
    ocr_result = ocr.ocr(roi, cls=False)
    
    if not ocr_result:
        if debug:
            print("[Health] OCR found no text")
        return None, None

    health_text = None
    health_bbox_roi = None

    for line in ocr_result:
        if not line:
            continue
        for word in line:
            quad = word[0]
            text = word[1][0]
            conf = word[1][1]

            if '/' not in text:
                continue

            nums = re.findall(r'\d+', text)
            if len(nums) < 2:
                continue

            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            health_text = text
            health_bbox_roi = (x_min, y_min, x_max, y_max)

            if debug:
                print(f"[Health] Found text '{health_text}' conf={conf:.3f}")
            break

        if health_text:
            break

    return health_text, health_bbox_roi


# =========================================================
# 计算血量百分比
# =========================================================
def calculate_health_rate(health_text):
    """从血量文本计算百分比"""
    if health_text is None:
        return None
    
    nums = re.findall(r'\d+', health_text)
    if len(nums) < 2:
        return None
    
    cur = int(nums[0])
    max_hp = int(nums[1])
    
    if max_hp <= 0:
        return None
    
    return cur / max_hp


# =========================================================
# BBox 缓存管理
# =========================================================
def load_cached_bbox(bbox_save_path: str = "bbox.txt"):
    """从 bbox.txt 中读取保存的文本 bbox（全图坐标）"""
    if not os.path.exists(bbox_save_path):
        return None
    
    try:
        with open(bbox_save_path, "r") as f:
            content = f.read().strip()
        
        # 只读取 bbox= 这一行
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith("bbox="):
                bbox_str = line.split("=", 1)[1].strip("()")
                coords = [int(x.strip()) for x in bbox_str.split(",")]
                if len(coords) == 4:
                    return tuple(coords)
        return None
    except Exception:
        return None


def save_bbox(bbox_save_path: str, health_bbox_full):
    """只保存 bbox 到文件"""
    try:
        with open(bbox_save_path, "w") as f:
            f.write(f"bbox={health_bbox_full}\n")
    except Exception as e:
        print(f"[Health] Failed to save bbox: {e}")


def delete_bbox(bbox_save_path: str):
    """删除 bbox 文件"""
    try:
        if os.path.exists(bbox_save_path):
            os.remove(bbox_save_path)
            print("[Health] Deleted bbox.txt due to failure")
    except Exception as e:
        print(f"[Health] Failed to delete bbox: {e}")


# =========================================================
# 主函数：提取血量百分比
# =========================================================
def extract_health_rate_from_image(
    image_path: str,
    bbox_save_path: str = "bbox.txt",
    debug: bool = False
):
    """
    输入 image_path
    输出 health_rate (float or None)
    
    流程：
    1. 如果存在缓存的 bbox，直接使用
    2. 如果缓存失败，删除 bbox.txt，运行完整检测
    3. 完整检测成功，保存 bbox
    """
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to load image")

    # 尝试使用缓存的 bbox
    cached_bbox = load_cached_bbox(bbox_save_path)
    if cached_bbox is not None:
        print("[Health] Using cached bbox")
        health_rate = _ocr_with_bbox(img, cached_bbox, debug)
        if health_rate is not None:
            print(f"[Health] Cached bbox success, Rate = {health_rate:.2%}")
            return health_rate
        # 缓存失败，删除 bbox 文件
        print("[Health] Cached bbox failed, deleting bbox.txt")
        delete_bbox(bbox_save_path)

    # 运行完整的边缘检测
    print("[Health] Running full detection (edge detection)")
    hp_bar_bbox, step4_path = detect_hp_bar_by_edge(img)
    if hp_bar_bbox is None:
        if debug:
            print("[Health] No HP bar detected")
        return None

    # 读取 step4 图片进行 OCR
    if not os.path.exists(step4_path):
        if debug:
            print(f"[Health] Step4 image not found: {step4_path}")
        return None
    
    roi = cv2.imread(step4_path)
    if roi is None:
        if debug:
            print(f"[Health] Failed to load step4 image: {step4_path}")
        return None

    # OCR 识别
    health_text, health_bbox_roi = extract_health_text_from_roi(roi, debug)
    if health_text is None:
        if debug:
            print("[Health] No valid cur/max text found")
        return None

    # 计算血量百分比
    health_rate = calculate_health_rate(health_text)
    if health_rate is None:
        return None

    # 保存血量条的 bbox（final bbox）而不是文本 bbox
    save_bbox(bbox_save_path, hp_bar_bbox)
    
    # 转换文本 bbox 到全图坐标（用于可视化）
    H, W = img.shape[:2]
    roi_x1 = W // 4
    roi_y1 = H * 4 // 5
    
    health_bbox_full = (
        roi_x1 + health_bbox_roi[0],
        roi_y1 + health_bbox_roi[1],
        roi_x1 + health_bbox_roi[2],
        roi_y1 + health_bbox_roi[3],
    )

    # 保存可视化结果图
    result_img = img.copy()
    x1, y1, x2, y2 = hp_bar_bbox
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(result_img, 
                  (health_bbox_full[0], health_bbox_full[1]),
                  (health_bbox_full[2], health_bbox_full[3]),
                  (0, 0, 255), 2)
    cv2.putText(result_img, health_text, 
                (health_bbox_full[0], health_bbox_full[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite("hp_result_with_text.png", result_img)

    print(f"[Health] Full detection success, Rate = {health_rate:.2%}")

    return health_rate


# =========================================================
# 内部函数：使用 bbox 进行 OCR
# =========================================================
def _ocr_with_bbox(img, hp_bar_bbox, debug=False):
    """使用缓存的血量条 bbox 进行 OCR（类似 step4_final_bbox.png 的处理）"""
    x1, y1, x2, y2 = hp_bar_bbox
    
    # 检查 bbox 是否在图片范围内
    H, W = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        print(f"[Health] Bbox out of bounds: bbox=({x1},{y1},{x2},{y2}), image=({W},{H})")
        return None
    
    # 提取血量条区域（类似 step4_final_bbox.png）
    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        print(f"[Health] Bbox is invalid: bbox=({x1},{y1},{x2},{y2})")
        return None
    
    # OCR 识别（在整个血量条区域内查找文本）
    health_text, _ = extract_health_text_from_roi(roi, debug=False)
    if health_text is None:
        print("[Health] OCR failed: no valid health text found in cached bbox")
        # 保存失败的 ROI 用于调试
        cv2.imwrite("hp_ocr_failed_roi.png", roi)
        print(f"[Health] Saved failed ROI to hp_ocr_failed_roi.png (bbox={hp_bar_bbox})")
        return None
    
    # 计算血量百分比
    health_rate = calculate_health_rate(health_text)
    if health_rate is None:
        print(f"[Health] Failed to calculate health rate from text: {health_text}")
        return None
    
    return health_rate


# =========================================================
# 示例用法
# =========================================================
if __name__ == "__main__":
    rate = extract_health_rate_from_image(
        image_path="image2.png",
        bbox_save_path="bbox.txt",
        debug=True
    )

    print("\nFinal health rate:", rate)
