from PIL import Image, ImageDraw, ImageFont

from boxes import BoundingBox

def class_to_one_hot_vector(class_id, n_classes, smoothing_factor=0.0):
    one_hot_vector = [0.0] * n_classes
    one_hot_vector[class_id] = 1.0
    one_hot_vector = [(1 - smoothing_factor)*l + smoothing_factor/n_classes 
                      for l in one_hot_vector]
    return one_hot_vector


def get_output_size(height, width, filter_size, padding, stride):
    """Gets the spatial dimensions of the output feature map of a 
    convolution or pooling operator.

    Args:
        height (int): Height of the image input to the operator.
        width (int): Width of the image input to the operator.
        filter_size (int): Size of the filter used by the operator. The 
          filter is assumed to be square, so only one integer needs to 
          be provided.
        padding (string): Padding type to apply to the input image. Can 
          be either 'same' or 'valid'.
        stride (int): Stride value used when sliding the filter across 
          the input image. It is assumed that the horizontal stride and 
          the vertical stride are equal, so only one integer needs to be
          provided.

    Raises:
        ValueError: `padding` is neither 'same' nor 'valid'.

    Returns:
        tuple(int, int): Pair containing the height and the width of the 
          output feature map.
    """
    
    if isinstance(padding, str):
        if padding == 'same':
            int_padding = int(filter_size / 2)
        elif padding == 'valid':
            int_padding = 0
        else:
            raise ValueError(
                'The value for argument `padding` must be either "same" '
                f'or "valid" (got {padding}).')
    else:
        int_padding = padding
    return (
        int((height + 2*int_padding - filter_size) / stride) + 1,
        int((width + 2*int_padding - filter_size) / stride) + 1
    )


def adjust_bounding_box_annotations(annotation, new_image_size):
    """Adjusts the coordinates of every bounding box appearing in an 
    image after it has been resized.

    Args:
        annotation (dict): Dictionary having the following structure:
          {
              'width': The image's width before resizing,
              'height': The image's height before resizing,
              'objects': [
                  {
                      'class': The class label for this object,
                      'bounding_box': The bounding box for this 
                      object. It must be an instance of the 
                      `BoundingBox` class
                  },
                  ...
              ]
          }
        new_image_size (tuple(int, int)): Pair of integers representing 
          the image's width and height after resizing.

    Returns:
        dict: A dictionary having the same structure as that of 
          `annotation` where all the information has been updated to 
          reflect the change in the image's size.
    """

    old_width = annotation['width']
    old_height = annotation['height']
    new_width, new_height = new_image_size
    rescale_factor_x = new_width / old_width
    rescale_factor_y = new_height / old_height
    for obj in annotation['objects']:
        old_bounding_box = obj['bounding_box']
        new_bounding_box = BoundingBox(
            int(old_bounding_box.x_min * rescale_factor_x),
            int(old_bounding_box.y_min * rescale_factor_y),
            int(old_bounding_box.x_max * rescale_factor_x),
            int(old_bounding_box.y_max * rescale_factor_y)
        )
        obj['bounding_box'] = new_bounding_box
    
    annotation['width'] = new_width
    annotation['height'] = new_height
    return annotation


def clip_predicted_box(
    predicted_box, 
    image_width, 
    image_height, 
    visible_area_threshold=0.5
):
    """Clips a predicted bounding box if it extends outside the image's 
    boundaries.

    Args:
        predicted_box (BoundingBox): `BoundingBox` instance represting 
          the predicted box.
        image_width (int): The image's width, in pixels.
        image_height (int): The image's height, in pixels.
        visible_area_threshold (float, optional): Threshold for 
          filtering out boxes that mostly extends outside the image's 
          boundaries. If the fraction of the visible area (that is, the 
          portion of the predicted box that is within the image's 
          boundaries) to the total area of the box is less than this 
          threshold, the box is discarded and `None` is returned. 
          Defaults to 0.5.

    Returns:
        BoundingBox: The predicted box clipped to the image's boundaries 
          or `None` if the box gets filtered out by the treshold.
    """

    visible_x_min = max(predicted_box.x_min, 1)
    visible_y_min = max(predicted_box.y_min, 1)
    visible_x_max = min(predicted_box.x_max, image_width - 1)
    visible_y_max = min(predicted_box.y_max, image_height - 1)

    visible_area = ((visible_x_max - visible_x_min) 
                    * (visible_y_max - visible_y_min))
    total_area = predicted_box.get_area()
    fraction_of_visible_area = visible_area / total_area
    if fraction_of_visible_area >= visible_area_threshold:
        clipped_box = BoundingBox(
            visible_x_min,
            visible_y_min,
            visible_x_max,
            visible_y_max
        )
    else:
        clipped_box = None
    return clipped_box


def draw_predicted_boxes(image, predictions, index_to_class_map):
    """Draws bounding boxes predicted by an object detection model on 
    top of an existing image.

    Args:
        image (numpy.array): 3D array representing an image.
        predictions (dict): Dictionary mapping class identifiers 
          (integers) to a list of (box, conf) pairs, where `box` is an 
          instance of the `BoundingBox` class representing a bounding 
          box that encloses an object of that class (with a confidence 
          score of `conf`), as predicted by the model.
        index_to_class_map (dict): Dictionary for mapping integer 
          identifiers to class names.

    Returns:
        PIL.Image: Image object representing `image` with the predicted 
          boxes drawn on top of it.
    """

    image_obj = Image.fromarray(image)
    draw = ImageDraw.Draw(image_obj)
    font = ImageFont.load_default()
    for class_id, box_confidence_pairs in predictions.items():
        for (box, confidence) in box_confidence_pairs:
            # Draw the bounding box
            draw.rectangle([box.x_min, box.y_min, box.x_max, box.y_max])
            # Draw the class label and the confidence
            class_name = index_to_class_map[class_id]
            text = f'{class_name} ({confidence:.2f})'
            w, h = font.getsize(text)
            draw.rectangle(
                [box.x_min, box.y_min, box.x_min + w, box.y_min + h],
                fill='black'
            )
            draw.text([box.x_min, box.y_min], text, fill='white', font=font)
    return image_obj
