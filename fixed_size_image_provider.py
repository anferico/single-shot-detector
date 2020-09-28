import numpy as np

from boxes import BoundingBox

class FixedSizeImageProvider(object):
    """Class for turning arbitrarily-sized images into fixed-size images 
    while updating objects annotations at the same time. 

    Args:
        target_width (int): The desired width for the images.
        target_height (int): The desired height for the images.
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


class SSDSquareCropper(FixedSizeImageProvider):
    """Class for cropping random square portions from arbitrarily-sized 
    images while updating objects annotations at the same time. 

    Args:
        target_side_length (int): The desired side length of the square 
          crop, after resizing. Notice the "after resizing" 
          clarification: the size of the crop is chosen randomly, then 
          the crop is resized to `target_side_length`.
        seed (int): The random seed used when cropping random square 
          portions from images. Defaults to None.
    """    

    def __init__(self, target_side_length, seed=None):
        super(SSDSquareCropper, self).__init__(
            target_side_length, 
            target_side_length
        )
        np.random.seed(seed)

    def extract_fixed_size_image(
        self, 
        image,
        annotation
    ):
        """Crop a square portion from the given image.

        Args:
            image (PIL.Image): The image from which to crop a square
              portion.
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
            tuple(PIL.Image, dict): A pair containing the cropped square
              portion of the given image and the updated information 
              about the objects appearing in the image.
        """

        short_side = min(image.width, image.height)
        long_side = max(image.width, image.height)
        random_offset = np.random.randint(0, long_side - short_side + 1)
        valid_objects = []
        if short_side == image.width: # Image is tall-thin
            for obj in annotation['objects']:
                ground_truth_box = obj['bounding_box']
                # Check if the center of the ground truth box falls 
                # within the chosen square crop. If it doesn't, the 
                # ground truth box is dropped
                if (ground_truth_box.center_y >= random_offset 
                    and ground_truth_box.center_y <= random_offset + short_side):
                    # Compute the new coordinates of the ground truth 
                    # box relative to the square crop
                    new_x_min = ground_truth_box.x_min
                    new_y_min = max(ground_truth_box.y_min - random_offset, 0)
                    new_x_max = ground_truth_box.x_max
                    new_y_max = min(
                        ground_truth_box.y_max - random_offset, 
                        short_side
                    )
                    # Update the bounding box
                    resized_box = BoundingBox(
                        (new_x_min / short_side) * self.target_width,
                        (new_y_min / short_side) * self.target_width,
                        (new_x_max / short_side) * self.target_width,
                        (new_y_max / short_side) * self.target_width
                    )                                      
                    valid_objects.append({
                        'class': obj['class'],
                        'bounding_box': resized_box
                    })
            
            # Crop the original image
            image_cropped = image.crop((
                0, 
                random_offset, 
                image.width, 
                random_offset + image.width
            ))
        else: # short_side == image.height (Image is short-fat)
            for obj in annotation['objects']:
                ground_truth_box = obj['bounding_box']      
                # Check if the center of the ground truth box falls 
                # within the chosen square crop. If it doesn't, the 
                # ground truth box is dropped                          
                if (ground_truth_box.center_x >= random_offset 
                    and ground_truth_box.center_x <= random_offset + short_side):
                    # Compute the new coordinates of the ground truth 
                    # box relative to the square crop                    
                    new_x_min = max(ground_truth_box.x_min - random_offset, 0)
                    new_y_min = ground_truth_box.y_min
                    new_x_max = min(
                        ground_truth_box.x_max - random_offset, 
                        short_side
                    )
                    new_y_max = ground_truth_box.y_max
                    # Update the bounding box
                    resized_box = BoundingBox(
                        (new_x_min / short_side) * self.target_width,
                        (new_y_min / short_side) * self.target_width,
                        (new_x_max / short_side) * self.target_width,
                        (new_y_max / short_side) * self.target_width
                    )      
                    valid_objects.append({
                        'class': obj['class'],
                        'bounding_box': resized_box
                    })

            # Crop the original image
            image_cropped = image.crop((
                random_offset, 
                0, 
                random_offset + image.height,
                image.height, 
            ))

        # Resize the image to the desired size
        image_resized = image_cropped.resize((
            self.target_width, 
            self.target_height # It's actually equal to the width
        ))

        # Create a new annotation
        new_annotation = {
            'width': self.target_width,
            'height': self.target_height,
            'objects': valid_objects
        }

        return image_resized, new_annotation
