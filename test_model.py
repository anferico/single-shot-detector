from tensorflow.keras.applications import inception_v3
from PIL import Image
import numpy as np

from ssd_inception_v3 import (
    build_ssd_inception_v3_model, 
    get_ssd_inception_v3_feature_map_sizes
)
from ssd_common import decode_predictions, non_max_suppression
from default_boxes_generator import DefaultBoxesGenerator
from voc_utils import get_index_to_class_map
from utils import draw_predicted_boxes

index_to_class_map = get_index_to_class_map()

n_classes = len(index_to_class_map)
ssd_inception_v3 = build_ssd_inception_v3_model(
    n_classes=n_classes,
    n_default_boxes=5,
    weights='/home/anfri/weights.h5'
)

image_obj = Image.open('/home/anfri/MyProjects/ssd/images/000021.jpg')
image = np.asarray(image_obj)
image_preprocessed = inception_v3.preprocess_input(image)

def_boxes_gen = DefaultBoxesGenerator(
    get_ssd_inception_v3_feature_map_sizes,
    default_boxes_aspect_ratios=[1, 2, 3, 1/2, 1/3],
    min_scale=0.2,
    max_scale=0.9
)

y_pred_conf, y_pred_loc = ssd_inception_v3(
    image_preprocessed[np.newaxis, :, :, :],
    training=False
)

pred_boxes_gen = decode_predictions(
    y_pred_conf.numpy()[0],
    y_pred_loc.numpy()[0],
    default_boxes_generator=def_boxes_gen,
    image_height=image_obj.height,
    image_width=image_obj.width,
    n_classes=n_classes,
    clip_boxes_to_image_bounds=True    
)

predictions = non_max_suppression(
    pred_boxes_gen,
    background_class_id=0,
    confidence_threshold=0.05, 
    iou_threshold=0.45 
)

annotated_image = draw_predicted_boxes(image, predictions, index_to_class_map)
annotated_image.save('prediction.jpg')
