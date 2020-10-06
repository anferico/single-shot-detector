import math

from PIL import Image, ImageEnhance
import numpy as np

from boxes import BoundingBox

class FixedSizeImageProvider(object):
    """Class for turning arbitrarily-sized images into fixed-size images 
    while updating objects annotations at the same time. 

    Args:
        target_width (int): The desired image width.
        target_height (int): The desired image height.
    """

    def __init__(self, target_width, target_height):
        self.target_width = target_width
        self.target_height = target_height

    def extract_fixed_size_image(
        self, 
        image,
        annotation
    ):
        """Crop a rectangular portion of fixed size from the given image.

        Args:
            image (PIL.Image): The image from which to crop a 
              rectangular portion.
            annotation (dict): Dictionary containing annotations for the 
              given image. The dictionary is expected to have a specific 
              structure, that is:
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
            tuple(PIL.Image, dict): A pair containing the cropped 
              rectangular portion of the given image and the updated 
              information about the objects appearing in the image.

        Raises:
            NotImplementedError: This class doesn't provide a default 
              implementation for this method.
        """

        raise NotImplementedError(
            '`extract_fixed_size_image` has not been implemented in '
            f'{type(self).__name__}'
        )


class SSDImageAugmentator(FixedSizeImageProvider):
    """Class for cropping fixed-size rectangular portions from 
    arbitrarily-sized images while also performing some data 
    augmentation (random contrast/brightness, channel shift and 
    horizontal flip), updating objects annotations accordingly. 

    Args:
        target_width (int): The desired image width after resizing. 
          Notice the "after resizing" clarification: the width of the 
          crop is chosen randomly, then scaled up or down to match 
          `target_width`.
        target_height (int): The desired image height after resizing. 
          Notice the "after resizing" clarification: the height of the 
          crop is chosen randomly, then scaled up or down to match 
          `target_height`.
        min_rel_crop_size (float, optional): Minimum size of the random 
          rectangular crop relative to the original image size. Must be 
          between 0 and 1. Defaults to 0.1, that is 10% of the original 
          image size.
        contrast_range (float, optional): Range for the random contrast 
          to apply to an image. If `contrast_range` is equal to R and 
          the original contrast is 1, the new contrast will be between 
          (1 - R) and (1 + R). As a result, `contrast_range` must be 
          between 0 and 1. Defaults to 0.25.
        brightness_range (float, optional): Range for the random 
          brightness to apply to an image. If `brightness_range` is 
          equal to B and the original brightness is 1, the new 
          brightness will be between (1 - B) and (1 + B). As a result, 
          `brightness_range` must be between 0 and 1. Defaults to 0.25.
        channel_shift (bool, optional): Whether or not to perform a 
          random shift of the RGB color channels. Defaults to True.
        flip_prob (float, optional): Probability that an image gets 
          flipped horizontally. Defaults to 0.5.
        seed (int, optional): The random seed used when cropping random 
          rectangular portions and applying random photometric 
          distortions to images. Defaults to None.
    """

    def __init__(
        self, 
        target_width, 
        target_height,
        min_rel_crop_size=0.1,
        contrast_range=0.25, 
        brightness_range=0.25, 
        channel_shift=True,
        flip_prob=0.5, 
        seed=None
    ):
        super(SSDImageAugmentator, self).__init__(
            target_width, 
            target_height
        )
        self.min_rel_crop_size = min_rel_crop_size
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.channel_shift = channel_shift
        self.flip_prob = flip_prob
        np.random.seed(seed)


    def _apply_photometric_distortions(
        self,
        image, 
        contrast_factor, 
        brightness_factor,
        flip_horizontally, 
        shift_channels=True
    ):
        def contrast_enhancer(image): 
            return ImageEnhance.Contrast(image).enhance(contrast_factor)
        
        def brightness_enhancer(image):
            return ImageEnhance.Brightness(image).enhance(brightness_factor)
        
        def horizontal_flipper(image):
            if not flip_horizontally:
                return image
            image_arr = np.array(image, dtype=np.uint8)
            image_arr = image_arr[:, ::-1, :]
            return Image.fromarray(image_arr)

        def channel_shifter(image):
            if not shift_channels:
                return image
            channels_permutation = np.random.permutation(3)
            image_arr = np.array(image, dtype=np.uint8)
            image_arr = image_arr[:, :, channels_permutation]
            return Image.fromarray(image_arr)

        image_augmentation_pipeline = [
            contrast_enhancer, 
            brightness_enhancer, 
            horizontal_flipper, 
            channel_shifter
        ]
        # Shuffle the order of application of the distortions. Despite 
        # the order of horizontal flip and channel shift is irrelevant, 
        # in general the order of application matters
        image_augmentation_pipeline = np.random.permutation(
            image_augmentation_pipeline
        )

        image_distorted = image
        for distortion in image_augmentation_pipeline:
            image_distorted = distortion(image_distorted)
        return image_distorted


    def extract_fixed_size_image(
        self, 
        image,
        annotation
    ):
        """Crop a rectangular portion from the given image while also 
        applying some photometric distortions to it.

        Args:
            image (PIL.Image): The image from which to crop a 
              rectangular portion.
            annotation (dict): Dictionary containing annotations for the 
              given image. The dictionary is expected to have a specific 
              structure, that is:
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
            tuple(PIL.Image, dict): A pair containing the cropped 
              rectangular portion of the given image and the updated 
              information about the objects appearing in the image.
        """

        while True:
            rel_crop_size = np.random.uniform(
                low=self.min_rel_crop_size, 
                high=1.0
            )
            crop_width = int(math.sqrt(rel_crop_size) * image.width)
            crop_height = int(math.sqrt(rel_crop_size) * image.height)
            offset_width = np.random.randint(0, image.width - crop_width + 1)
            offset_height = np.random.randint(0, image.height - crop_height + 1)
            flip_horizontally = np.random.rand() <= self.flip_prob
            valid_objects = []
            for obj in annotation['objects']:
                ground_truth_box = obj['bounding_box']
                # Check if the center of the ground truth box falls 
                # within the chosen rectangular crop. If it doesn't, the 
                # ground truth box is dropped
                cx, cy = ground_truth_box.center_x, ground_truth_box.center_y
                if (offset_width <= cx <= offset_width + crop_width
                    and offset_height <= cy <= offset_height + crop_height):
                    # Compute the new coordinates of the ground truth 
                    # box relative to the rectangular crop
                    new_x_min = max(0, ground_truth_box.x_min - offset_width)
                    new_y_min = max(0, ground_truth_box.y_min - offset_height)
                    new_x_max = min(
                        crop_width, 
                        ground_truth_box.x_max - offset_width
                    )
                    new_y_max = min(
                        crop_height, 
                        ground_truth_box.y_max - offset_height
                    )
                    # Account for the possible horizontal flipping of 
                    # the image. Notice that `new_y_min` and `new_y_max` 
                    # remain unchanged
                    if flip_horizontally:
                        new_x_min, new_x_max = (
                            crop_width - new_x_max, 
                            crop_width - new_x_min
                        )
                    # Update the bounding box
                    updated_box = BoundingBox(
                        (new_x_min / crop_width) * self.target_width,
                        (new_y_min / crop_height) * self.target_height,
                        (new_x_max / crop_width) * self.target_width,
                        (new_y_max / crop_height) * self.target_height
                    )
                    valid_objects.append({
                        'class': obj['class'],
                        'bounding_box': updated_box
                    })
            # Exit the loop if there's at least one valid bounding box 
            # within the random rectangular crop, otherwise start over 
            # and try another crop
            if len(valid_objects) > 0:
                break

        # Crop the original image
        crop_coordinates = (
            offset_width,               # Left
            offset_height,              # Top
            offset_width + crop_width,  # Right
            offset_height + crop_height # Bottom
        )
        image_cropped = image.crop(box=crop_coordinates)
        # Resize the image to the desired size
        image_resized = image_cropped.resize((
            self.target_width, 
            self.target_height
        ))
        # Apply photometric distortions to the image
        brightness_factor = np.random.uniform(
            low=1-self.brightness_range, 
            high=1+self.brightness_range
        )
        contrast_factor = np.random.uniform(
            low=1-self.contrast_range, 
            high=1+self.contrast_range
        )
        image_distorted = self._apply_photometric_distortions(
            image_resized,
            contrast_factor,
            brightness_factor,
            flip_horizontally,
            self.channel_shift
        )
        # Create a new annotation
        new_annotation = {
            'width': self.target_width,
            'height': self.target_height,
            'objects': valid_objects
        }
        return image_distorted, new_annotation
