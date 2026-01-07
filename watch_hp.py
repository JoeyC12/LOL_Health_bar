import time
import threading
from datetime import datetime
from PIL import ImageGrab
import tkinter as tk

import lol   # 你的主逻辑文件


SCREENSHOT_PATH = "screen.png"
BBOX_PATH = "bbox.txt"
INTERVAL = 1   # 秒


def capture_screen(save_path):
    """
    只截右侧副屏（逻辑坐标）
    主屏: 1470 x 956
    副屏: 1920 x 1080
    """
    MAIN_W = 1470
    SIDE_W = 1920
    SIDE_H = 1080

    bbox = (
        MAIN_W,
        0,
        MAIN_W + SIDE_W,
        SIDE_H
    )

    img = ImageGrab.grab(bbox=bbox)
    img.save(save_path)



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

        # 窗口大小 + 位置（左上角）
        self.root.geometry("220x70+20+40")

        # UI
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

    def update_health(self, health_rate):
        """更新血量显示（线程安全）"""
        try:
            if health_rate is None:
                text = "HP: N/A"
                color = "red"
            else:
                # 显示百分比，保留1位小数
                text = f"HP: {health_rate * 100:.1f}%"
                if health_rate < 0.3:
                    color = "red"
                elif health_rate < 0.6:
                    color = "yellow"
                else:
                    color = "lime"

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

            # 1️⃣ 截屏
            capture_screen(SCREENSHOT_PATH)

            # 2️⃣ 血量检测
            health_rate = lol.extract_health_rate_from_image(
                image_path=SCREENSHOT_PATH,
                bbox_save_path=BBOX_PATH,
                debug=False
            )

            # 3️⃣ 更新 HUD（线程安全）
            if hud.running:  # 确保窗口还在运行
                hud.root.after(0, hud.update_health, health_rate)

            # 4️⃣ 日志
            now = datetime.now().strftime("%H:%M:%S")
            if health_rate is None:
                print(f"[{now}] HP: N/A")
            else:
                print(f"[{now}] HP: {health_rate:.1%}")

            # 5️⃣ 控制频率
            elapsed = time.time() - start
            sleep_time = max(0.05, INTERVAL - elapsed)
            time.sleep(sleep_time)
            
        except Exception as e:
            # 错误处理，避免程序崩溃
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] Error: {e}")
            if hud.running:
                hud.root.after(0, hud.update_health, None)
            time.sleep(INTERVAL)

    print("HUD stopped.")


def main():
    hud = HealthHUD()

    # 后台线程跑 OCR / 截屏
    t = threading.Thread(target=health_loop, args=(hud,), daemon=True)
    t.start()

    # 主线程跑 UI
    hud.start()


if __name__ == "__main__":
    main()
