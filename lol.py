import os
import cv2
import numpy as np
from pathlib import Path
import re
from paddleocr import PaddleOCR


def detect_hp_bar_by_edge(img_bgr, output_dir="./hp_edge_steps"):
    H, W = img_bgr.shape[:2]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_y1 = H * 4 // 5
    roi_y2 = H
    roi_x1 = W // 4
    roi_x2 = W * 3 // 4

    roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2].copy()
    cv2.imwrite(str(out_dir / "step0_bottom_center_roi.png"), roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(str(out_dir / "step1_gray.png"), gray)
    
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(out_dir / "step2_edges.png"), edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(str(out_dir / "step3_edges_closed.png"), edges_closed)

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
        # 绘制所有候选框
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
# ✅ 新函数：对外接口
# =========================================================
def extract_health_rate_from_image(
    image_path: str,
    bbox_save_path: str = "bbox.txt",
    debug: bool = True
):
    """
    输入 image_path
    输出 health_rate (float or None)

    同时把 health text 的 bbox（全图坐标）
    保存到 bbox_save_path
    """

    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to load image")

    bbox, step4_path = detect_hp_bar_by_edge(img)
    if bbox is None:
        if debug:
            print("[Health] No HP bar detected")
        return None

    x1, y1, x2, y2 = bbox
    
    # 直接读取 step4_final_bbox.png 进行 OCR
    if not os.path.exists(step4_path):
        if debug:
            print(f"[Health] Step4 image not found: {step4_path}")
        return None
    
    roi = cv2.imread(step4_path)
    if roi is None:
        if debug:
            print(f"[Health] Failed to load step4 image: {step4_path}")
        return None

    ocr = PaddleOCR(
        lang="en",
        use_angle_cls=False,
        show_log=False
    )

    ocr_result = ocr.ocr(roi, cls=False)
    if not ocr_result:
        if debug:
            print("[Health] OCR found no text")
        return None

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

    if health_text is None:
        if debug:
            print("[Health] No valid cur/max text found")
        return None

    nums = re.findall(r'\d+', health_text)
    cur = int(nums[0])
    max_hp = int(nums[1])

    if max_hp <= 0:
        return None

    health_rate = cur / max_hp

    # step4_final_bbox.png 是 ROI 坐标系的，需要转换到全图坐标
    # ROI 的起始位置：roi_x1 = W // 4, roi_y1 = H * 4 // 5
    H, W = img.shape[:2]
    roi_x1 = W // 4
    roi_y1 = H * 4 // 5
    
    # 转换 bbox 到全图坐标
    health_bbox_full = (
        roi_x1 + health_bbox_roi[0],
        roi_y1 + health_bbox_roi[1],
        roi_x1 + health_bbox_roi[2],
        roi_y1 + health_bbox_roi[3],
    )

    # 写入 bbox.txt
    with open(bbox_save_path, "w") as f:
        f.write(
            f"text={health_text}\n"
            f"cur={cur}\n"
            f"max={max_hp}\n"
            f"health_rate={health_rate:.6f}\n"
            f"bbox={health_bbox_full}\n"
        )

    # 保存带文本标注的结果图
    result_img = img.copy()
    # 绘制血量条 bbox（绿色）
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 绘制文本 bbox（红色）
    cv2.rectangle(result_img, 
                  (health_bbox_full[0], health_bbox_full[1]),
                  (health_bbox_full[2], health_bbox_full[3]),
                  (0, 0, 255), 2)
    # 添加文本标注
    cv2.putText(result_img, health_text, 
                (health_bbox_full[0], health_bbox_full[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite("hp_result_with_text.png", result_img)

    if debug:
        print(f"[Health] Rate = {health_rate:.2%}")
        print(f"[Health] BBox saved to {bbox_save_path}")

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
