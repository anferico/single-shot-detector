from tensorflow.keras.layers import (
    Reshape,
    Softmax,
    Concatenate
)
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

from ssd_common import (
    _conv_bn_relu, 
    _class_prediction_layer, 
    _offsets_prediction_layer
)
from utils import get_output_size

def build_ssd_inception_v3_model(
    n_classes,
    n_default_boxes, 
    input_shape=None, 
    l2_regularization=0.0, 
    weights=None
):
    """Builds an SSD model with Inception-v3 as the base network.

    Args:
        n_classes (int): Number of different classes that objects can 
          belong to, including the background class.
        n_default_boxes (int): Number of default boxes for each feature 
          map location.
        input_shape (tuple, optional): Triplet of the form (height, 
          width, channels) representing the shape of the input. 
          `channels` must be equal to 3, otherwise pre-trained weights 
          for Inception-v3 cannot be used. If None, it will be set to 
          (None, None, 3). Defaults to None.
        l2_regularization (float, optional): L2 regularization factor. 
          Defaults to 0.0.
        weights (string, optional): Path to an .h5 file containing 
          pretrained weights for SSD Inception-v3. Defaults to None.

    Returns:
        keras.Model: The SSD Inception-v3 object detection model.
    """

    inception_v3_head = InceptionV3(
        include_top=False,
        weights='imagenet' if weights is None else None,
        input_shape=input_shape
    )
    inception_v3_head.trainable = False
    
    mixed_0 = inception_v3_head.get_layer('mixed0').output
    mixed_1 = inception_v3_head.get_layer('mixed1').output
    mixed_2 = inception_v3_head.get_layer('mixed2').output
    mixed_0_2 = Concatenate()([mixed_0, mixed_1, mixed_2])

    mixed_3 = inception_v3_head.get_layer('mixed3').output
    mixed_4 = inception_v3_head.get_layer('mixed4').output
    mixed_5 = inception_v3_head.get_layer('mixed5').output
    mixed_6 = inception_v3_head.get_layer('mixed6').output
    mixed_7 = inception_v3_head.get_layer('mixed7').output
    mixed_3_7 = Concatenate()([mixed_3, mixed_4, mixed_5, mixed_6, mixed_7])

    mixed_8 = inception_v3_head.get_layer('mixed8').output
    mixed_9 = inception_v3_head.get_layer('mixed9').output
    mixed_10 = inception_v3_head.get_layer('mixed10').output
    mixed_8_10 = Concatenate()([mixed_8, mixed_9, mixed_10])

    # Extra feature layers
    conv_0 = _conv_bn_relu(
        filters=512,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_regularizer=l2(l2_regularization)
    )(mixed_10)
    feature_layer_0 = _conv_bn_relu(
        filters=1024,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid',
        kernel_regularizer=l2(l2_regularization)
    )(conv_0)

    conv_1 = _conv_bn_relu(
        filters=256,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_regularizer=l2(l2_regularization)
    )(feature_layer_0)
    feature_layer_1 = _conv_bn_relu(
        filters=512,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        kernel_regularizer=l2(l2_regularization)
    )(conv_1)

    # Class prediction layers
    class_probs_from_mixed_0_2 = _class_prediction_layer(
        n_classes=n_classes,
        n_default_boxes=n_default_boxes, 
        kernel_regularizer=l2(l2_regularization)
    )(mixed_0_2)
    class_probs_from_mixed_3_7 = _class_prediction_layer(
        n_classes=n_classes,
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(mixed_3_7)
    class_probs_from_mixed_8_10 = _class_prediction_layer(
        n_classes=n_classes,
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(mixed_8_10)
    class_probs_from_feature_layer_0 = _class_prediction_layer(
        n_classes=n_classes,
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(feature_layer_0)
    class_probs_from_feature_layer_1 = _class_prediction_layer(
        n_classes=n_classes,
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(feature_layer_1)

    # Offsets prediction layers
    offsets_from_mixed_0_2 = _offsets_prediction_layer(
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(mixed_0_2)
    offsets_from_mixed_3_7 = _offsets_prediction_layer(
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(mixed_3_7)
    offsets_from_mixed_8_10 = _offsets_prediction_layer(
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(mixed_8_10)
    offsets_from_feature_layer_0 = _offsets_prediction_layer(
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(feature_layer_0)
    offsets_from_feature_layer_1 = _offsets_prediction_layer(
        n_default_boxes=n_default_boxes,
        kernel_regularizer=l2(l2_regularization)
    )(feature_layer_1)

    # Concatenate class confidences predictions
    class_probs = Concatenate()([
        class_probs_from_mixed_0_2,
        class_probs_from_mixed_3_7,
        class_probs_from_mixed_8_10,
        class_probs_from_feature_layer_0,
        class_probs_from_feature_layer_1
    ])

    # Softmax activation
    reshape_0 = Reshape((-1, n_classes))(class_probs)
    softmax = Softmax()(reshape_0)
    class_probs = Reshape((-1,), name='confidence')(softmax)

    # Concatenate offsets predictions
    offsets = Concatenate(name='localization')([
        offsets_from_mixed_0_2,
        offsets_from_mixed_3_7,
        offsets_from_mixed_8_10,
        offsets_from_feature_layer_0,
        offsets_from_feature_layer_1
    ])

    final_model = Model(
        inputs=inception_v3_head.input, 
        outputs=[class_probs, offsets], 
        name='ssd_inception_v3'
    )
    if weights is not None:
        final_model.load_weights(weights)
        
    return final_model


def get_ssd_inception_v3_feature_map_sizes(height, width):
    """Gets the spatial dimensions of the feature maps used for 
    predictions in an SSD model, given the input size.

    Args:
        height (int): The input image's height (in pixels). Must be at 
          least 267 for this function to return a meaningful output.
        width (int): The input image's width (in pixels). Must be at 
          least 267 for this function to return a meaningful output.

    Raises:
        ValueError: Either `height` or `width` is less than 267.

    Returns:
        list(tuple): List of (height, width) pairs representing the 
          spatial dimensions of the feature maps used for predictions.
    """

    if height < 267 or width < 267:
        raise ValueError('Height and width must both be >= 267.')

    feature_map_sizes = []
    m, n = height, width
    # Path to `mixed_2` (35x35 with 299x299 input images)
    m, n = get_output_size(m, n, 3, 'valid', 2) # conv2d_94
    m, n = get_output_size(m, n, 3, 'valid', 1) # conv2d_95
    m, n = get_output_size(m, n, 3, 'valid', 2) # max_pooling2d_4
    m, n = get_output_size(m, n, 3, 'valid', 1) # conv2d_98
    m, n = get_output_size(m, n, 3, 'valid', 2) # max_pooling2d_5
    feature_map_sizes.append((m, n))

    # Path to `mixed_7` (17x17 with 299x299 input images)
    m, n = get_output_size(m, n, 3, 'valid', 2) # conv2d_120 
                                                # or conv2d_123 
                                                # or max_pooling2d_6
    feature_map_sizes.append((m, n))

    # Path to `mixed_10` (8x8 with 299x299 input images)
    m, n = get_output_size(m, n, 3, 'valid', 2) # conv2d_165 
                                                # or conv2d_169 
                                                # or max_pooling2d_7    
    feature_map_sizes.append((m, n))

    # Extra feature layers
    # Path to `feature_layer_0` (3x3 with 299x299 input images)
    m, n = get_output_size(m, n, 3, 'valid', 2) # feature_layer_0
    feature_map_sizes.append((m, n))
    # Path to `feature_layer_1` (1x1 with 299x299 input images)
    m, n = get_output_size(m, n, 3, 'valid', 1) # feature_layer_1
    feature_map_sizes.append((m, n))

    return feature_map_sizes
