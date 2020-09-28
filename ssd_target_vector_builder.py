import numpy as np

from utils import class_to_one_hot_vector

class SSDTargetVectorBuilder(object):
    """Class for producing a target vector (as required by the SSD 
    object detection model) for images given their bounding box 
    annotations.

    Args:
        default_boxes_generator (DefaultBoxesGenerator): Instance of the
          `DefaultBoxesGenerator` class which provides default boxes in 
          the form of instances of the `BoundingBox` class.
        class_to_index_map (dict): Dictionary for mapping class names 
          (including the background class) to integer identifiers.
        background_class_name (string): The name of the "background" 
          class, that is the class that corresponds to no object.
        iou_threshold (float, optional): Threshold on the IoU between a 
          default box and a ground truth box for determining whether 
          there's a match between the two or not. An IoU greater than or 
          equal to the threshold indicates a match. Defaults to 0.5.
    """

    def __init__(
        self, 
        default_boxes_generator,
        class_to_index_map,
        background_class_name,
        iou_threshold=0.5
    ):
        self.default_boxes_generator = default_boxes_generator
        self.class_to_index_map = class_to_index_map
        self.background_class_name = background_class_name
        self.iou_threshold = iou_threshold
    

    def build_target_vector(self, annotation):
        """Builds a target vector for an image given its bounding box 
        annotations.

        Args:
            annotation (dict): Dictionary containing annotations for a 
              single image. The dictionary is expected to have a 
              specific structure, that is:
              {
                  'width': The image's width,
                  'height': The image's height,
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

        Returns:
            tuple(numpy.array, numpy.array): A pair containing the 
              target vector for classification and the target vector for 
              localization. Both vectors are 1D numpy arrays.
        """

        target_vector_for_classification = []
        target_vector_for_localization = []
        for default_box in self.default_boxes_generator.generate_default_boxes(
            annotation['height'],
            annotation['width']
        ):
            # Match the default box with at most 1 ground truth box
            best_iou = 0
            best_ground_truth_box = None
            best_ground_truth_box_class = None
            for obj in annotation['objects']:
                # Compute the intersection over union between the 
                # default box and the ground truth box
                ground_truth_box = obj['bounding_box']
                iou = default_box.intersection_over_union(
                    ground_truth_box
                )
                if (iou >= self.iou_threshold 
                    and iou > best_iou):
                    # Update the best match for this default box
                    best_iou = iou
                    best_ground_truth_box = ground_truth_box
                    best_ground_truth_box_class = obj['class']
            if best_ground_truth_box is not None:
                # Set the target class for the default box as that of 
                # the matched ground truth box
                class_id = self.class_to_index_map[best_ground_truth_box_class]
                target_vector_for_classification.extend(
                    class_to_one_hot_vector(
                        class_id=class_id,
                        n_classes=len(self.class_to_index_map)
                    )
                )
                # Set the offsets for the default box as the offsets 
                # from the matched ground truth box
                target_vector_for_localization.extend(
                    best_ground_truth_box.deviation_relative_to(
                        default_box
                    )
                )
            else:
                # Set the target class for the default box as the
                # background class
                class_id = self.class_to_index_map[self.background_class_name]
                target_vector_for_classification.extend(
                    class_to_one_hot_vector(
                        class_id=class_id, 
                        n_classes=len(self.class_to_index_map)
                    )
                )
                # This default box didn't match any ground truth box, 
                # meaning that it is a "negative" box. Negative boxes do 
                # not contribute to the loss, so technically it doesn't 
                # matter how we set their target offsets. At the same 
                # time, however, one must be able to identify the 
                # negative boxes in order to factor them out while 
                # computing the loss. The trick here is to use a special 
                # value (`np.inf` in this case) for all 4 target offsets 
                # so that during the computation of the loss we can 
                # identify negative boxes by checking which values of 
                # `target_vector_for_localization` equal that special 
                # value
                target_vector_for_localization.extend((np.inf,) * 4)

        return (
            np.array(target_vector_for_classification), 
            np.array(target_vector_for_localization)
        )
