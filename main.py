import numpy as np
from trocr import indic_trocr
import utils

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

print(crops)
# trocr prediction
recognition_predictor = indic_trocr(model_path="trocr_files")
word_preds = recognition_predictor([crop for page_crops in crops for crop in page_crops])

print(word_preds)