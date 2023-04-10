import os, sys
import numpy as np
import cv2
import shutil
from trocr import indic_trocr
import utils
from PIL import Image

from textron.main  import textron

def ocr(image_name:str):
    pages = []
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
    word_preds = []
    for num, cropped_img in enumerate(crops[0]):
        if not os.path.isdir("cropped_imgs"):
            os.mkdir("cropped_imgs")
        cv2.imwrite(os.path.join("cropped_imgs", f"img_{num}.jpg"),cropped_img)
        img_ = Image.open(os.path.join("cropped_imgs", f"img_{num}.jpg")).convert("RGB")
        
        word_preds.append(recognition_predictor(img_)["preds"])
        
    words = {"preds":word_preds} 
    shutil.rmtree("cropped_imgs")   
    return words




if __name__ == "__main__":
    image_name = sys.argv[1]
    print(ocr(image_name))