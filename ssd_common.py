from collections import defaultdict

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Flatten
)
import tensorflow as tf
import numpy as np

from boxes import BoundingBox

def _conv_bn_relu(
    filters,
    kernel_size,
    strides,
    padding,
    kernel_regularizer
):
    def block(x):
        conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_regularizer=kernel_regularizer
        )(x)
        bn = BatchNormalization()(conv)
        relu = Activation('relu')(bn)
        return relu
    return block


def _class_prediction_layer(n_classes, n_default_boxes, kernel_regularizer):
    def prediction_layer(x):
        conv = Conv2D(
            filters=(n_classes * n_default_boxes),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_regularizer=kernel_regularizer
        )(x)
        predictions = Flatten()(conv)
        return predictions
    return prediction_layer


def _offsets_prediction_layer(n_default_boxes, kernel_regularizer):
    def prediction_layer(x):
        conv = Conv2D(
            filters=(4 * n_default_boxes),
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            kernel_regularizer=kernel_regularizer
        )(x)
        offsets_predictions = Flatten()(conv)
        return offsets_predictions
    return prediction_layer    


def _smooth_l1(x):
    abs_x = tf.abs(x)
    return tf.where(abs_x < 1, 0.5 * x**2, abs_x - 0.5)


def _get_hard_negatives_mask(
    y_pred_background, 
    negatives_mask, 
    negatives_to_positives_ratio=3.0
):
    """Computes a binary mask that retains hard negatives, filtering out 
    all other negatives.

    Args:
        y_pred_background (Tensor): 2D tensor of shape (batch_size, 
          total_default_boxes) containing the estimated probability for 
          the background class, for each default box.
        negatives_mask (Tensor): 2D tensor of shape (batch_size, 
          total_default_boxes) containing binary flags indicating 
          whether a given default box is negative (i.e. it didn't match 
          any of the ground truth boxes).
        negatives_to_positives_ratio (float, optional): Ratio between 
          the number of negative and positive boxes after the hard 
          negative mining step. Defaults to 3.0, which corresponds to 
          retaining at most 3 negative boxes for each positive box (as 
          in the original SSD paper).
    Returns:
        Tensor: Binary mask of shape (batch_size, total_default_boxes) 
          that is equal to 1 if the default box is a hard negative, and 
          0 otherwise.
    """

    # Compute the number of hard negatives for each sample in the batch. 
    # This number will be at most `negatives_to_positives_ratio` times 
    # `n_positives`. If `n_positives` is 0, then all negatives are 
    # considered hard negatives
    n_positives = tf.reduce_sum(1 - negatives_mask, axis=1) # (batch_size,)
    n_negatives = tf.reduce_sum(negatives_mask, axis=1) # (batch_size,)
    n_hard_negatives = tf.where(
        n_positives == 0,
        n_negatives,
        tf.minimum(negatives_to_positives_ratio * n_positives, n_negatives)
    )
    n_hard_negatives = tf.cast(n_hard_negatives, tf.int32) # (batch_size,)
    # For each sample in the batch, repeat the number of hard negatives 
    # along the column axis for a number of times equal to the total 
    # amount of default boxes. This will allow us to build a preliminary 
    # version of the hard negatives mask (refer to the next step below)
    n_hard_negatives = tf.broadcast_to( # (batch_size, total_default_boxes)
        n_hard_negatives[:, tf.newaxis], # (batch_size, 1)
        tf.shape(negatives_mask)
    )
    # Build a mask that retains the desired number of hard negatives. 
    # Note that this mask is not quite what we're looking for, because 
    # it only works under the assumption that the default boxes are 
    # sorted in increasing order of "hardness", with all the negative 
    # boxes coming before any of the positive boxes
    ranges = tf.broadcast_to( # (batch_size, total_default_boxes)
        tf.range(tf.shape(n_hard_negatives)[1]), 
        tf.shape(n_hard_negatives)
    )
    hard_negatives_mask_sorted = tf.where( # (batch_size, total_default_boxes)
        ranges < n_hard_negatives, 
        1., 
        0.
    )
    # What we would like to have is a binary mask that retains the 
    # desired amount of hard negatives when applied to the original 
    # `y_pred_background`, where the default boxes appear in their 
    # original order. The trick we'll use involves "unsorting" the mask 
    # obtained in the previous step. First of all, we're going to sort 
    # the scores relative to the negative boxes in descending order of 
    # confidence loss. Since the confidence loss is the standard softmax 
    # cross-entropy, this is equivalent to sorting the probability 
    # scores in ascending order. Such probability scores are a measure 
    # of the "hardness" of negative boxes

    # Let's use a trick and divide `y_pred_background` by 
    # `negatives_mask` element-wise. Since `negatives_mask` is a binary 
    # array, there are two possible scenarios: the value of the mask is 
    # 1 (that is, we're dealing with a negative box), in which case we 
    # keep the original probability score; the value is 0 (indicating a 
    # positive box), in which case the original score blows up to 
    # infinity due to the division by 0. This way, when we try to sort 
    # the probability scores, those relative to negative boxes will be 
    # brought to the top, whereas those relative to positive boxes will 
    # be moved to the bottom, since infinity is bigger than any other 
    # value. This trick allows us to sort the probability scores for 
    # negative boxes while also separating them from positive boxes
    negatives_scores = y_pred_background / negatives_mask
    negatives_scores_argsort = tf.argsort(
        negatives_scores,
        axis=1
    )    
    # The `negatives_scores_argsort` we computed earlier can be used to 
    # sort the probability scores. What we're trying to compute here is 
    # the "inverse" of that, namely an argsort that if applied to the 
    # sorted probabilities, gives us back the probability scores in 
    # their original order. As it turns out, if we apply the very same 
    # "inverse" argsort to `hard_negatives_mask_sorted`, we will obtain 
    # a mask that retains only hard negatives when applied to 
    # `y_pred_background`
    combined_indices = tf.stack(
        [ranges, negatives_scores_argsort], 
        axis=-1
    )
    negatives_confidences_argsort_inv = tf.argsort(
        combined_indices[:, :, 1:], 
        axis=1
    )[:, :, 0]
    # Unsort `hard_negatives_mask_sorted` to get the final version of 
    # the hard negatives mask
    batch_indices = tf.reshape(
        tf.range(tf.shape(y_pred_background)[0]), 
        (-1, 1)
    )
    batch_indices = tf.broadcast_to(
        batch_indices, 
        tf.shape(negatives_scores)
    )
    hard_negatives_mask = tf.gather_nd(
        hard_negatives_mask_sorted, 
        tf.stack(
            [batch_indices, negatives_confidences_argsort_inv], 
            axis=-1
        )
    )
    return hard_negatives_mask


def confidence_loss(
    y_true, 
    y_pred, 
    n_classes,
    background_class_id=0,
    hard_negative_mining=True,
    negatives_to_positives_ratio=3.0
):
    """Computes the confidence loss of an SSD model.

    Args:
        y_true (Tensor): 2D tensor of shape (batch_size, n_output_conf) 
          representing the target vector.
        y_pred (Tensor): 2D tensor of shape (batch_size, n_output_conf) 
          representing the predicted vector.
        n_classes (int): Number of different classes that objects can 
          belong to, including the background class.
        background_class_id (int, optional): Integer identifier of the 
          background class. Must be between 0 and (n_classes - 1). 
          Defaults to 0.
        hard_negative_mining (bool, optional): Whether or not to perform 
          hard negative mining so to reduce the imbalance between 
          negative and positive examples. Usually leads to faster 
          optimization and a more stable training. Defaults to True.
        negatives_to_positives_ratio (float, optional): Ratio between 
          the number of negative and positive boxes after the hard 
          negative mining step. Ignored if `hard_negative_mining` is 
          False. Defaults to 3.0, which corresponds to retaining at most 
          3 negative boxes for each positive box (as in the original SSD 
          paper).
    Returns:
        Tensor: 1D tensor of shape (batch_size,) representing the 
          confidence loss for each sample in the batch.
    """

    batch_size = tf.shape(y_true)[0]
    n_output_conf = tf.shape(y_true)[1]
    total_default_boxes = int(n_output_conf / n_classes)
    # Derive the negatives mask from `y_true`. Negative boxes have the 
    # background class set as their target class
    y_true_reshaped = tf.reshape(
        y_true, 
        (batch_size, total_default_boxes, n_classes)
    )
    negatives_mask = tf.where(
        tf.argmax(y_true_reshaped, axis=2) == background_class_id, 
        1.0, 
        0.0
    )

    if hard_negative_mining:
        # Reshape `y_pred` to have one set of probability scores 
        # for each row
        y_pred_reshaped = tf.reshape(
            y_pred,
            (batch_size, total_default_boxes, n_classes)
        )
        # Find the hard negatives
        y_pred_background = y_pred_reshaped[:, :, background_class_id]
        hard_negatives_mask = _get_hard_negatives_mask(
            y_pred_background, 
            negatives_mask,
            negatives_to_positives_ratio
        )
        # Reshape `hard_negatives_mask` so that it can be applied to 
        # `y_pred`
        hard_negatives_mask_flat = tf.reshape(
            tf.broadcast_to(
                hard_negatives_mask[:, :, tf.newaxis],
                (batch_size, total_default_boxes, n_classes)
            ),
            tf.shape(y_pred)
        )
    else:
        hard_negatives_mask_flat = tf.ones_like(y_pred)

    # Reorganize `negatives_mask` in such a way that it can be 
    # multiplied element-wise with `y_pred` to retain confidence scores 
    # relative to negative boxes only
    negatives_mask_flat = tf.reshape(
        tf.broadcast_to(
            negatives_mask[:, :, tf.newaxis],
            (batch_size, total_default_boxes, n_classes)
        ),
        tf.shape(y_true)
    )
    positives_mask_flat = 1 - negatives_mask_flat
    # Build a mask that when multiplied element-wise with `y_pred` 
    # retains confidence scores relative to positive or hard negative 
    # boxes only. Notice that `tf.maximum` plays the role of a logical 
    # OR here
    final_mask_flat = tf.maximum(positives_mask_flat, hard_negatives_mask_flat)
    
    # Compute the loss (I'm adding a small constant to log(...) to keep 
    # it from blowing up to infinity when the model predicts a value 
    # that is very close to 0)
    total_loss = -tf.reduce_sum(
        final_mask_flat * y_true * tf.math.log(y_pred + 1e-16), 
        axis=1
    )
    
    n_matched_boxes = tf.reduce_sum(positives_mask_flat, axis=1) / n_classes
    # We have to be careful when using `tf.where`. The point is that if 
    # one branch can produce NaN or Inf values (such as when performing 
    # a division by 0), the gradient computation will be messed up (yup, 
    # good job TensorFlow team). Notice that it doesn't matter if the 
    # boolean condition is meant to avoid exactly that (such as checking 
    # if the divisor is 0): you'd incur in that bug regardless.
    # The trick is to use another `tf.where` to replace "problematic" 
    # values (e.g. zeros in the case of division) in the tensor with 
    # "safe" values.
    # To have a sense of what causes this problem, check the link below:
    # stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444
    n_matched_boxes_safe = tf.where(
        n_matched_boxes != 0, 
        n_matched_boxes, 
        tf.ones_like(n_matched_boxes) # "Safe" value for division
    )
    average_loss_per_box = tf.where(
        n_matched_boxes != 0, 
        total_loss / n_matched_boxes_safe, 
        0
    )
    # As prescribed by Keras's documentation (check the link below), we 
    # should return one loss value for each sample in the batch, NOT the 
    # average loss over the samples in the batch. The weird thing is 
    # that it works even if I do return the average loss over the 
    # samples in the batch.
    # https://keras.io/api/losses/#creating-custom-losses
    return average_loss_per_box
    # average_loss_per_batch_sample = tf.reduce_mean(average_loss_per_box)
    # return average_loss_per_batch_sample


def localization_loss(y_true, y_pred):
    """Computes the localization loss of an SSD model.

    Args:
        y_true (Tensor): 2D tensor of shape (batch_size, n_output_loc) 
          representing the target vector.
        y_pred (Tensor): 2D tensor of shape (batch_size, n_output_loc) 
          representing the predicted vector.

    Returns:
        Tensor: 1D tensor of shape (batch_size,) representing the 
          localization loss for each sample in the batch.
    """

    # Derive the positives mask from `y_true`. A value of `np.inf` for a 
    # target offset inside `y_true` indicates that the box that the 
    # offset refers to is negative. On the other hand, if a value is not 
    # `np.inf`, it means that it refers to a positive box
    positives_mask_flat = tf.where(y_true == np.inf, 0.0, 1.0)
    # Clean up `y_true` before computing the loss. In particular, we 
    # have to replace `np.inf` values whose purpose was to mark negative 
    # boxes with a real number, for example 0, so that we don't incur 
    # into errors when using it in further computations 
    y_true_cleaned = tf.where(y_true == np.inf, 0.0, y_true)
    # Compute the loss
    total_loss = tf.reduce_sum(
        positives_mask_flat * _smooth_l1(y_pred - y_true_cleaned), 
        axis=1
    )
    
    n_matched_boxes = tf.reduce_sum(positives_mask_flat, axis=1) / 4
    n_matched_boxes_safe = tf.where(
        n_matched_boxes != 0, 
        n_matched_boxes, 
        tf.ones_like(n_matched_boxes) # "Safe" value for division
    )
    average_loss_per_box = tf.where(
        n_matched_boxes != 0, 
        total_loss / n_matched_boxes_safe, 
        0
    )
    # As prescribed by Keras's documentation (check the link below), we 
    # should return one loss value for each sample in the batch, NOT the 
    # average loss over the samples in the batch. The weird thing is 
    # that it works even if I do return the average loss over the 
    # samples in the batch.
    # https://keras.io/api/losses/#creating-custom-losses    
    return average_loss_per_box
    # average_loss_per_batch_sample = tf.reduce_mean(average_loss_per_box)
    # return average_loss_per_batch_sample


def decode_predictions(
    y_pred_conf,
    y_pred_loc,
    default_boxes_generator,
    image_height,
    image_width,
    n_classes,
    clip_boxes_to_image_bounds=True
):
    """Decodes raw predictions made by an SSD object detection model.

    Args:
        y_pred_conf (numpy.array): 1D array containing class confidences 
          predictions for a single input image.
        y_pred_loc (numpy.array): 1D array containing default box 
          offsets predictions for a single input image.
        default_boxes_generator (DefaultBoxesGenerator): Instance of the 
          `DefaultBoxesGenerator` class providing default boxes.
        image_height (int): Height of the image that produced the 
          predictions (needed by `default_boxes_generator`).
        image_width (int): Width of the image that produced the 
          predictions (needed by `default_boxes_generator`).
        n_classes (int): Number of different classes that objects can 
          belong to, including the background class.
        clip_boxes_to_image_bounds (bool, optional): Whether or not to 
          clip the predicted boxes to the image's bounds, in case they 
          include regions outside the image. Note that this will affect 
          any post-processing of the predicted boxes, such as non-max 
          suppression. Defaults to True.

    Yields:
        tuple(BoundingBox, int, float): Triplets containing the 
          predicted box, the predicted class and the associated 
          probability score for that class.
    """

    y_pred_conf = y_pred_conf.reshape((-1, n_classes))
    predicted_classes = np.argmax(y_pred_conf, axis=1).tolist()
    predicted_confidences = np.max(y_pred_conf, axis=1).tolist()
    predicted_deviations = y_pred_loc.reshape((-1, 4)).tolist()
    
    default_boxes = default_boxes_generator.generate_default_boxes(
        image_height, 
        image_width
    )
    for default_box, predicted_class, predicted_confidence, devs in zip(
        default_boxes, 
        predicted_classes, 
        predicted_confidences, 
        predicted_deviations
    ):
        # Apply the predicted deviations to the default box to get a 
        # prediction for the ground truth box
        dev_x, dev_y, dev_w, dev_h = devs
        predicted_box = default_box.apply_deviations(
            dev_x, 
            dev_y, 
            dev_w, 
            dev_h
        )
        if clip_boxes_to_image_bounds:
            # Clip the predicted box if it extends beyond the image's 
            # boundaries
            predicted_box = BoundingBox(
                max(predicted_box.x_min, 1),
                max(predicted_box.y_min, 1),
                min(predicted_box.x_max, image_width - 1),
                min(predicted_box.y_max, image_height - 1)
            )
        
        yield predicted_box, predicted_class, predicted_confidence


def non_max_suppression(
    predicted_boxes_generator,
    background_class_id=0,
    confidence_threshold=0.01, 
    iou_threshold=0.45
):
    """Performs non-max suppression (NMS) on boxes predicted by an 
    object detection model (such as SSD).

    Args:
        predicted_boxes_generator (generator object): Generator object 
          yielding triplets of the form (predicted_box, predicted_class, 
          predicted_confidence) where `predicted_box` is an instance of 
          the `BoundingBox` class representing a single box predicted by 
          the model, `predicted_class` is the predicted class for the 
          object enclosed in the predicted box and `predicted_confidence` 
          is the probability estimate that the object enclosed in the 
          predicted box belongs to the predicted class.
        background_class_id (int, optional): Integer identifier of the 
          "background" class (corresponding to "no object"). Defaults 
          to 0.
        confidence_threshold (float, optional): Minimum confidence score 
          for a prediction to be considered valid. Predictions having a 
          confidence score lower than `confidence_threshold` are 
          filtered out. Defaults to 0.01.
        iou_threshold (float, optional): Minimum value of the 
          intersection over union between two boxes for them to be 
          considered as overlapping with each other. Defaults to 0.45.

    Returns:
        dict: Dictionary mapping class identifiers (integers) to a list 
          of (box, conf) pairs, where `box` is an instance of the 
          `BoundingBox` class representing a bounding box that encloses 
          an object of that class (with a confidence score of `conf`), 
          as identified by the model inside an image. If no objects of a 
          given class were identified, the list associated with that 
          class will be empty.
    """

    final_predictions = defaultdict(list)
    for pred_box, pred_class, pred_conf in predicted_boxes_generator:
        # Filter out boxes of the background class
        if pred_class == background_class_id:
            continue
        # Filter out boxes with very low confidence
        if pred_conf < confidence_threshold:
            continue
        # Get boxes of class `predicted_class` as well as the 
        # corresponding confidence scores from the final predictions
        box_confidence_pairs = final_predictions[pred_class]
        # Look for overlappings between boxes of the same class
        pairs_to_drop = []
        worse_than_existing_box = False
        for pair in box_confidence_pairs:
            box, confidence = pair
            iou = pred_box.intersection_over_union(box)
            if iou > iou_threshold:                    
                if pred_conf > confidence:
                    pairs_to_drop.append(pair)
                else:
                    worse_than_existing_box = True
                    break
        # Add this predicted box to the final predictions if it wasn't 
        # found to be worse than another predicted box
        if not worse_than_existing_box:
            final_predictions[pred_class].append((pred_box, pred_conf))
        # Drop boxes that were found to be worse than this predicted box
        for p in pairs_to_drop:
            final_predictions[pred_class].remove(p)
    
    return final_predictions
