from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=False,
    use_doc_preprocessor=False,   
  
)

img = cv2.imread("/Users/joeychen/Desktop/Hisense/lol_hp_ocr_steps/step2_ocr_roi_crop.png")  # 换成你自己的截图
res = ocr.ocr(img)


print(res)
