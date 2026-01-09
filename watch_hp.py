import time
import threading
import os
from datetime import datetime
import tkinter as tk
import numpy as np
import cv2

# 尝试使用 mss（更快的截屏库），如果不可用则回退到 PIL
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    from PIL import ImageGrab
    MSS_AVAILABLE = False
    print("[Health] Warning: mss not available, using PIL.ImageGrab (slower)")

import lol   # 你的主逻辑文件


SCREENSHOT_PATH = "screen.png"
BBOX_PATH = "bbox.txt"
INTERVAL = 0  # 秒


def capture_screen(save_path, save_to_file=True, bbox_region=None):
    """
    截屏函数（支持全屏或指定区域）
    
    save_to_file: 是否保存到文件（False时只返回numpy数组，避免文件IO）
    bbox_region: 如果提供，只截取该区域 (x1, y1, x2, y2)，否则截取整个副屏
    """
    MAIN_W = 1470
    SIDE_W = 1920
    SIDE_H = 1080
    
    if bbox_region is not None:
        # 只截取指定区域（bbox_region 已经包含了边距，直接使用）
        x1, y1, x2, y2 = bbox_region
        capture_bbox = {
            'left': max(MAIN_W, x1),
            'top': max(0, y1),
            'width': min(SIDE_W, x2 - x1),
            'height': min(SIDE_H, y2 - y1)
        }
    else:
        # 截取整个副屏
        capture_bbox = {
            'left': MAIN_W,
            'top': 0,
            'width': SIDE_W,
            'height': SIDE_H
        }
    
    if MSS_AVAILABLE:
        # 使用 mss（更快）
        with mss.mss() as sct:
            monitor = {
                'left': capture_bbox['left'],
                'top': capture_bbox['top'],
                'width': capture_bbox['width'],
                'height': capture_bbox['height']
            }
            img = sct.grab(monitor)
            # mss 返回 BGRA，转换为 numpy 数组
            img_array = np.array(img)
            # 转换为 RGB（去掉 alpha 通道）
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
            if save_to_file:
                from PIL import Image
                img_pil = Image.fromarray(img_array)
                img_pil.save(save_path)
            return img_array
    else:
        # 使用 PIL.ImageGrab（较慢）
        bbox = (
            capture_bbox['left'],
            capture_bbox['top'],
            capture_bbox['left'] + capture_bbox['width'],
            capture_bbox['top'] + capture_bbox['height']
        )
        img = ImageGrab.grab(bbox=bbox)
        if save_to_file:
            img.save(save_path)
        return np.array(img)



class HealthHUD:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LoL Health HUD")

        # ===============================
        # 🔴 关键：mac 悬浮窗口
        # ===============================
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.88)     # 半透明
        self.root.overrideredirect(True)          # 无边框 HUD

        # 窗口大小 + 位置（副屏左上角）
        # 主屏宽度 1470，副屏从 x=1470 开始
        MAIN_W = 1470
        WINDOW_X = MAIN_W + 20  # 副屏左上角，留 20px 边距
        WINDOW_Y = 40           # 顶部留 40px 边距
        self.root.geometry(f"220x100+{WINDOW_X}+{WINDOW_Y}")

        # UI - 具体血量显示（左上角）
        self.health_text_label = tk.Label(
            self.root,
            text="--/--",
            font=("Menlo", 16, "bold"),
            fg="white",
            bg="#141414"
        )
        self.health_text_label.pack(pady=(5, 0))

        # UI - 百分比显示
        self.label = tk.Label(
            self.root,
            text="HP: --",
            font=("Menlo", 22, "bold"),
            fg="lime",
            bg="#141414"
        )
        self.label.pack(expand=True, fill="both")

        # 退出绑定（Esc）
        self.root.bind("<Escape>", lambda e: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.running = True

    def update_health(self, health_data):
        """更新血量显示（线程安全）
        health_data: (health_rate, health_text) 或 None
        """
        try:
            if health_data is None:
                health_text = "--/--"
                text = "HP: N/A"
                color = "red"
            else:
                health_rate, health_text = health_data
                # 显示百分比，保留1位小数
                text = f"HP: {health_rate * 100:.1f}%"
                if health_rate < 0.3:
                    color = "red"
                elif health_rate < 0.6:
                    color = "yellow"
                else:
                    color = "lime"

            # 更新具体血量显示
            self.health_text_label.config(text=health_text)
            # 更新百分比显示
            self.label.config(text=text, fg=color)
            # 强制更新显示
            self.root.update_idletasks()
        except Exception as e:
            # 如果窗口已关闭，忽略错误
            pass

    def start(self):
        self.root.mainloop()

    def stop(self):
        """停止 HUD"""
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass


def health_loop(hud: HealthHUD):
    print("=" * 60)
    print(" LoL Health HUD started")
    print(" Press ESC to quit")
    print("=" * 60)

    while hud.running:
        try:
            start = time.time()

            # 1️⃣ 检查是否有缓存的bbox，如果有则只截取该区域
            cached_bbox = lol.load_cached_bbox(BBOX_PATH)
            t0 = time.time()
            
            if cached_bbox is not None:
                # 只截取bbox区域（更快！）
                # cached_bbox 是相对于副屏图片的坐标（0,0 开始）
                x1, y1, x2, y2 = cached_bbox
                margin = 50  # 边距，确保不遗漏
                MAIN_W = 1470
                
                # 计算截屏区域（屏幕坐标）
                # 需要将副屏图片坐标转换为屏幕坐标（副屏从 MAIN_W 开始）
                screen_x1 = MAIN_W + x1 - margin
                screen_y1 = y1 - margin
                screen_x2 = MAIN_W + x2 + margin
                screen_y2 = y2 + margin
                
                capture_region = (screen_x1, screen_y1, screen_x2, screen_y2)
                img_array = capture_screen(SCREENSHOT_PATH, save_to_file=False, bbox_region=capture_region)
                # 转换为BGR格式（OpenCV使用）
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # 调整bbox坐标（因为只截取了部分区域，坐标需要相对化）
                # 截取的图片从 (screen_x1, screen_y1) 开始
                # 原始 bbox 在副屏图片中的坐标是 (x1, y1, x2, y2)
                # 在截取的图片中，bbox 的坐标应该是相对于截取图片的
                # 因为截取时减去了 margin，所以 bbox 在截取图片中的位置是 (margin, margin, x2-x1+margin, y2-y1+margin)
                adjusted_bbox = (
                    margin,  # x1 调整为边距
                    margin,  # y1 调整为边距  
                    (x2 - x1) + margin,  # x2 调整为相对坐标（原始宽度 + margin）
                    (y2 - y1) + margin  # y2 调整为相对坐标（原始高度 + margin）
                )
                if False:  # 只在debug时显示
                    print(f"[Health] Using cached bbox: original={cached_bbox}, screen_region={capture_region}, adjusted={adjusted_bbox}, image_size={img_array.shape[:2]}")
            else:
                # 截取整个副屏
                img_array = capture_screen(SCREENSHOT_PATH, save_to_file=False)
                # 转换为BGR格式（OpenCV使用）
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                adjusted_bbox = None
            
            # elapsed = (time.time() - t0) * 1000
            # print(f"[Timing] capture_screen: {elapsed:.2f} ms")

            # 2️⃣ 血量检测（直接传数组，避免文件IO）
            result = lol.extract_health_rate_from_image(
                image_array=img_array,
                bbox_save_path=BBOX_PATH,
                debug=False,
                cached_bbox_adjusted=adjusted_bbox  # 传递调整后的bbox
            )

            # 3️⃣ 更新 HUD（线程安全）
            if hud.running:  # 确保窗口还在运行
                hud.root.after(0, hud.update_health, result)

            # 4️⃣ 日志（显示血量和时间）
            now = datetime.now().strftime("%H:%M:%S")
            if result is None:
                print(f"[{now}] HP: --/--")
            else:
                health_rate, health_text = result
                print(f"[{now}] HP: {health_text}")

            # 5️⃣ 控制频率
            elapsed = time.time() - start
            sleep_time = max(0.05, INTERVAL - elapsed)
            # time.sleep(sleep_time)
            
        except Exception as e:
            # 错误处理，避免程序崩溃
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Error: {e}")
            if hud.running:
                hud.root.after(0, hud.update_health, None)
            time.sleep(INTERVAL)

    print("HUD stopped.")


def main():
    # 第一次运行：删除旧的 bbox.txt，确保重新检测
    if os.path.exists(BBOX_PATH):
        try:
            os.remove(BBOX_PATH)
            print(f"[Health] Deleted existing {BBOX_PATH} for fresh detection")
        except Exception as e:
            print(f"[Health] Failed to delete {BBOX_PATH}: {e}")
    
    hud = HealthHUD()

    # 后台线程跑 OCR / 截屏
    t = threading.Thread(target=health_loop, args=(hud,), daemon=True)
    t.start()

    # 主线程跑 UI
    hud.start()


if __name__ == "__main__":
    main()
