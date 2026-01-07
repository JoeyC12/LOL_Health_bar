from PIL import ImageGrab

def capture_side_screen(save_path):
    """
    只截右侧副屏（逻辑坐标）
    主屏: 1470 x 956
    副屏: 1920 x 1080
    """
    MAIN_W = 1470
    SIDE_W = 1920
    SIDE_H = 1080

    bbox = (
        MAIN_W,          # left
        0,               # top
        MAIN_W + SIDE_W, # right
        SIDE_H           # bottom
    )

    img = ImageGrab.grab(bbox=bbox)
    img.save(save_path)


if __name__ == "__main__":
    capture_side_screen("side_screen_test.png")