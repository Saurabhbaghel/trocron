import os
import numpy as np
import cv2
import shutil
from trocr import indic_trocr
import utils
from PIL import Image

image_name = "ABBv3_4_ori.jpg"
pages = []
# preprocessing

# prediction from textron
from textron.main  import textron

pages.append(image_name)
detection_model = textron()
bboxes = detection_model(image_name)

# crop images
channels_last = len(pages) == 0 or isinstance(pages[0], np.ndarray) or isinstance(pages[0], str) # pages[0] will be string for textron

crops, location_preds = utils.prepare_crops_(
    pages,
    bboxes,
    channels_last,
    assume_straight_pages=True
    
)
recognition_predictor = indic_trocr(model_path="trocr_files")
# print(crops)
for num, cropped_img in enumerate(crops[0]):
    if not os.path.isdir("cropped_imgs"):
        os.mkdir("cropped_imgs")
    cv2.imwrite(os.path.join("cropped_imgs", f"img_{num}.jpg"),cropped_img)
    img_ = Image.open(os.path.join("cropped_imgs", f"img_{num}.jpg")).convert("RGB")
    
    word_preds = recognition_predictor(img_)
    print(word_preds["preds"])

    
# trocr prediction


# for img in os.listdir("cropped_imgs"):
#     img_ = Image.open(os.path.join("cropped_imgs", img)).convert("RGB")
    
#     word_preds = recognition_predictor(img_)
#     print(word_preds["preds"])

shutil.rmtree("cropped_imgs")

# # word_preds = recognition_predictor([crop for page_crops in crops for crop in page_crops])

# print(word_preds)