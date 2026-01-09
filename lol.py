import os
import cv2
import numpy as np
from pathlib import Path
import re
from paddleocr import PaddleOCR
import time

# =========================================================
# 全局OCR实例（只初始化一次，大幅提升性能）
# =========================================================
_ocr_instance = None

def get_ocr_instance():
    """获取全局OCR实例（懒加载，只初始化一次）"""
    global _ocr_instance
    if _ocr_instance is None:
        # 使用轻量级配置，提升速度
        _ocr_instance = PaddleOCR(
            lang="en",
            use_angle_cls=False,  # 禁用角度分类，提升速度
            show_log=False,
            use_gpu=False # 启用Intel MKL-DNN加速（如果可用）
        )
    return _ocr_instance


# =========================================================
# 边缘检测：定位血量条位置
# =========================================================
def detect_hp_bar_by_edge(img_bgr, output_dir="./hp_edge_steps", save_debug_images=False):
    """通过边缘检测定位血量条位置"""
    H, W = img_bgr.shape[:2]
    out_dir = Path(output_dir)
    if save_debug_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 提取底部中央区域
    roi_y1 = H * 4 // 5
    roi_y2 = H
    roi_x1 = W // 4
    roi_x2 = W * 3 // 4

    roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]  # 不需要copy，直接使用视图

    # 灰度化与模糊
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if save_debug_images:
        cv2.imwrite(str(out_dir / "step1_gray.png"), gray)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    if save_debug_images:
        cv2.imwrite(str(out_dir / "step2_edges.png"), edges)

    # 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    if save_debug_images:
        cv2.imwrite(str(out_dir / "step3_edges_closed.png"), edges_closed)

    # 查找轮廓
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 查找最佳候选框
    best = None
    best_width = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 4:
            continue
        if w > best_width:
            best_width = w
            best = (x, y, w, h)
    
    if save_debug_images:
        roi_with_candidates = roi.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= 40 and h >= 4:
                cv2.rectangle(roi_with_candidates, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(out_dir / "step4_all_candidates.png"), roi_with_candidates)

    if best is None:
        return None, None

    x, y, w, h = best
    # 只在需要时保存step4图片
    step4_path = None
    if save_debug_images:
        roi_final = roi.copy()
        cv2.rectangle(roi_final, (x, y), (x + w, y + h), (0, 0, 255), 3)
        step4_path = str(out_dir / "step4_final_bbox.png")
        cv2.imwrite(step4_path, roi_final)
    else:
        # 不保存文件，直接返回内存中的ROI
        step4_path = "memory"  # 标记为内存数据
    
    # 返回全图坐标的 bbox 和 step4 图片路径
    return (x + roi_x1, y + roi_y1, x + w + roi_x1, y + h + roi_y1), step4_path


# =========================================================
# OCR 识别：从图片中提取血量文本
# =========================================================
def extract_health_text_from_roi(roi, debug=False, show_all_text=False):
    """
    从 ROI 区域进行 OCR 识别，提取血量文本
    返回 (health_text, health_bbox_roi) 或 (None, None)
    
    show_all_text: 如果为True，即使失败也输出所有OCR检测到的文本
    """
    # 使用全局OCR实例，避免重复初始化
    ocr = get_ocr_instance()
    ocr_result = ocr.ocr(roi, cls=False)
    
    health_text = None
    health_bbox_roi = None
    all_texts = []  # 记录所有检测到的文本

    # 处理 OCR 结果
    if not ocr_result:
        if debug or show_all_text:
            print("[Health] OCR found no text (ocr_result is empty)")
        return None, None

    # 遍历所有 OCR 结果
    for line in ocr_result:
        if not line:
            continue
        for word in line:
            try:
                quad = word[0]
                text = word[1][0]
                conf = word[1][1]
                
                all_texts.append((text, conf))

                # if debug or show_all_text:
                #     print(f"[Health] OCR detected: '{text}' conf={conf:.3f}")

                if '/' not in text:
                    if debug or show_all_text:
                        print(f"[Health] Text '{text}' does not contain '/'")
                    continue

                nums = re.findall(r'\d+', text)
                if len(nums) < 2:
                    if debug or show_all_text:
                        print(f"[Health] Text '{text}' has less than 2 numbers (found: {nums})")
                    continue

                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                health_text = text
                health_bbox_roi = (x_min, y_min, x_max, y_max)

                if debug or show_all_text:
                    print(f"[Health] Found valid health text '{health_text}'")
                break
            except Exception as e:
                if debug or show_all_text:
                    print(f"[Health] Error processing OCR word: {e}, word={word}")

        if health_text:
            break

    # 如果失败且需要显示所有文本
    if health_text is None and show_all_text:
        if all_texts:
            print(f"[Health] All OCR texts detected: {all_texts}")
            print(f"[Health] Total texts: {len(all_texts)}, but none matched health format (need '/' and 2+ numbers)")
        else:
            print("[Health] No OCR texts detected (ocr_result structure may be empty or invalid)")
            if debug or show_all_text:
                print(f"[Health] OCR result structure: {ocr_result}")

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
    image_path: str = None,
    image_array: np.ndarray = None,
    bbox_save_path: str = "bbox.txt",
    debug: bool = False,
    cached_bbox_adjusted: tuple = None
):
    """
    输入 image_path 或 image_array (numpy array)
    输出 (health_rate, health_text) 或 None
    
    流程：
    1. 如果存在缓存的 bbox，直接使用
    2. 如果缓存失败，删除 bbox.txt，运行完整检测
    3. 完整检测成功，保存 bbox
    """
    # 支持直接传入numpy数组，避免文件IO
    if image_array is not None:
        img = image_array
    elif image_path is not None:
        if not os.path.exists(image_path):
            raise RuntimeError(f"Image not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Failed to load image")
    else:
        raise RuntimeError("Either image_path or image_array must be provided")

    # 尝试使用缓存的 bbox
    cached_bbox = load_cached_bbox(bbox_save_path)
    if cached_bbox is not None:
        # 如果提供了调整后的bbox（说明截屏时只截取了该区域），使用调整后的坐标
        bbox_to_use = cached_bbox_adjusted if cached_bbox_adjusted is not None else cached_bbox
        t0 = time.time()
        result = _ocr_with_bbox(img, bbox_to_use, debug)
        elapsed = (time.time() - t0) * 1000
        # print(f"[Timing] _ocr_with_bbox: {elapsed:.2f} ms")
        if result is not None:
            health_rate, health_text = result
            if debug:
                print(f"[Health] Cached bbox success, Rate = {health_rate:.2%}")
            return health_rate, health_text
        # 缓存失败，但不删除bbox.txt，也不重新检测，直接返回None
        # bbox.txt永远保持第一次成功检测的结果
        if debug:
            print("[Health] Cached bbox OCR failed, but keeping bbox.txt (no re-detection)")
        return None

    # 运行完整的边缘检测（只在第一次，没有缓存时运行）
    hp_bar_bbox, step4_data = detect_hp_bar_by_edge(img, save_debug_images=debug)
    if hp_bar_bbox is None:
        return None

    # 获取ROI（从文件或内存）
    if step4_data == "memory":
        # 从内存中提取ROI，避免文件读写
        H, W = img.shape[:2]
        roi_x1 = W // 4
        roi_x2 = W * 3 // 4
        roi_y1 = H * 4 // 5
        roi_y2 = H
        x, y, w, h = hp_bar_bbox
        # 转换回ROI坐标系
        x_roi = x - roi_x1
        y_roi = y - roi_y1
        roi = img[roi_y1:roi_y2, roi_x1:roi_x2][y_roi:y_roi+h, x_roi:x_roi+w].copy()  # 需要copy因为要传给OCR
    else:
        # 从文件读取（debug模式）
        if not os.path.exists(step4_data):
            if debug:
                print(f"[Health] Step4 image not found: {step4_data}")
            return None
        roi = cv2.imread(step4_data)
        if roi is None:
            if debug:
                print(f"[Health] Failed to load step4 image: {step4_data}")
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

    # 转换文本 bbox 到全图坐标
    H, W = img.shape[:2]
    roi_x1 = W // 4
    roi_y1 = H * 4 // 5
    
    health_bbox_full = (
        roi_x1 + health_bbox_roi[0],
        roi_y1 + health_bbox_roi[1],
        roi_x1 + health_bbox_roi[2],
        roi_y1 + health_bbox_roi[3],
    )

    # 左右各扩展50%（宽度扩展一倍）
    x1, y1, x2, y2 = health_bbox_full
    width = x2 - x1
    height = y2 - y1
    expanded_bbox = (
        max(0, int(x1 - width * 0.5)),  # 左边扩展50%
        int(y1 - height * 0.5),
        min(W, int(x2 + width * 0.5)),  # 右边扩展50%
        int(y2 + height * 0.2),
    )

    # 保存扩展后的文本 bbox（只在第一次成功时保存，如果已存在则不覆盖）
    if not os.path.exists(bbox_save_path):
        save_bbox(bbox_save_path, expanded_bbox)
        if debug:
            print(f"[Health] Saved bbox for first time: {expanded_bbox}")
    else:
        if debug:
            print(f"[Health] Bbox already exists, keeping original (not updating)")

    # 只在debug模式下保存可视化结果图
    if debug:
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
        # print(f"[Health] Saved bbox: {expanded_bbox} (text: '{health_text}', original: {health_bbox_full}, width: {width})")
        # print(f"[Health] Full detection success, Rate = {health_rate:.2%}")

    return health_rate, health_text


# =========================================================
# 内部函数：使用 bbox 进行 OCR
# =========================================================
def _ocr_with_bbox(img, text_bbox, debug=False):
    """使用缓存的文本 bbox（已扩展）进行 OCR"""
    x1, y1, x2, y2 = text_bbox
    
    # 检查 bbox 是否在图片范围内
    H, W = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        if debug:
            print(f"[Health] Bbox out of bounds: bbox=({x1},{y1},{x2},{y2}), image=({W},{H})")
        return None
    
    # 提取文本区域（已扩展的 bbox）
    roi = img[y1:y2, x1:x2].copy()  # 需要copy，避免视图问题
    if roi.size == 0:
        print(f"[Health] Bbox is invalid: bbox=({x1},{y1},{x2},{y2}), image_size={img.shape[:2]}")
        return None
    
    if debug:
        print(f"[Health] Extracted ROI: bbox=({x1},{y1},{x2},{y2}), roi_size={roi.shape}")
    
    # OCR 识别（在扩展后的区域内查找文本）
    health_text, _ = extract_health_text_from_roi(roi, debug=debug, show_all_text=debug)
    if health_text is None:
        if debug:
            print(f"[Health] OCR failed: no valid health text found in cached bbox (bbox={text_bbox}, image_size={img.shape[:2]})")
            # 保存失败的 ROI 用于调试
            cv2.imwrite("hp_ocr_failed_roi.png", roi)
            print(f"[Health] Saved failed ROI to hp_ocr_failed_roi.png (size: {roi.shape})")
        return None
    
    # 计算血量百分比
    health_rate = calculate_health_rate(health_text)
    if health_rate is None:
        if debug:
            print(f"[Health] Failed to calculate health rate from text: {health_text}")
        return None
    
    return health_rate, health_text


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
