import os
import cv2
import numpy as np
from pathlib import Path


def detect_hp_bar_by_edge(img_bgr, output_dir="./hp_edge_steps"):
    """
    使用边缘检测，在屏幕下半区域中
    直接选择“宽度最大的矩形”作为血条 UI
    返回: bbox (x1, y1, x2, y2) 或 None
    """
    H, W = img_bgr.shape[:2]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] Image size: {W} x {H}")

    # =========================================================
    # Step 0: 只取屏幕下 1/5 区域（更贴近 HUD）
    # =========================================================
    roi_y1 = H * 4 // 5
    roi = img_bgr[roi_y1:H, :].copy()
    cv2.imwrite(str(out_dir / "step0_bottom_roi.png"), roi)

    # =========================================================
    # Step 1: 灰度 + 模糊
    # =========================================================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(str(out_dir / "step1_gray.png"), gray)

    # =========================================================
    # Step 2: Canny 边缘
    # =========================================================
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(out_dir / "step2_edges.png"), edges)

    # =========================================================
    # Step 3: 形态学闭运算，连成横条
    # =========================================================
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(str(out_dir / "step3_edges_closed.png"), edges_closed)

    # =========================================================
    # Step 4: 找轮廓
    # =========================================================
    contours, _ = cv2.findContours(
        edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"[Detect] Total contours: {len(contours)}")

    # =========================================================
    # Step 5: 直接选择“宽度最大的矩形”
    # =========================================================
    best = None
    best_width = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # 忽略明显噪声
        if w < 40 or h < 4:
            continue

        if w > best_width:
            best_width = w
            best = (x, y, w, h)

    # =========================================================
    # Step 6: Debug — 没找到就画全部
    # =========================================================
    vis = img_bgr.copy()

    if best is None:
        print("[Result] No suitable rectangle found, drawing all contours")
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                vis,
                (x, y + roi_y1),
                (x + w, y + roi_y1 + h),
                (0, 255, 255),
                1
            )
        cv2.imwrite(str(out_dir / "step4_debug_all_contours.png"), vis)
        return None

    # =========================================================
    # Step 7: 生成最终 bbox
    # =========================================================
    x, y, w, h = best
    x1 = x
    y1 = y + roi_y1
    x2 = x + w
    y2 = y1 + h

    print("[Result] Selected widest rectangle:")
    print(f"  BBox: ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"  Size: {w} x {h}")

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / "step4_final_bbox.png"), vis)

    return (x1, y1, x2, y2)


if __name__ == "__main__":
    from paddleocr import PaddleOCR

    img_path = "image.png"

    if not os.path.exists(img_path):
        raise RuntimeError("image.png not found")

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError("failed to load image")

    # =========================================================
    # Step 1: 用边缘检测找血条 ROI
    # =========================================================
    bbox = detect_hp_bar_by_edge(img)

    if bbox is None:
        print("No HP bar detected")
        exit(0)

    print("HP bar bbox:", bbox)

    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2].copy()
    cv2.imwrite("hp_ocr_roi.png", roi)

    # =========================================================
    # Step 2: OCR
    # =========================================================
    ocr = PaddleOCR(
        lang="en",
        use_angle_cls=False,
        show_log=False
    )

    ocr_result = ocr.ocr(roi, cls=False)

    print("\n[OCR] Raw result:")
    print(ocr_result)

    print("\n[OCR] Texts with bounding boxes:\n")

    if not ocr_result:
        print("  (no text detected)")
        exit(0)

    for line_idx, line in enumerate(ocr_result):
        if not line:
            continue

        print(f"Line {line_idx}:")

        for word_idx, word in enumerate(line):
            # PaddleOCR 输出结构
            quad = word[0]          # 4 点坐标
            text = word[1][0]       # 文字
            conf = word[1][1]       # 置信度

            # 计算 axis-aligned bbox（方便后处理）
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            print(
                f"  [{word_idx}] text='{text}' conf={conf:.3f}\n"
                f"       quad={quad}\n"
                f"       bbox=(x1={x_min}, y1={y_min}, x2={x_max}, y2={y_max})"
            )

    print("\n[Debug] OCR ROI saved to: hp_ocr_roi.png")


