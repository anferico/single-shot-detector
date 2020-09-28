from pathlib import Path

from tensorflow.keras.preprocessing.image import (
    Iterator,
    load_img,
    img_to_array
)
import tensorflow as tf
import numpy as np

from utils import adjust_bounding_box_annotations

class SSDDirectoryIterator(Iterator):
    """Class for iterating over a directory containing images. Provides 
    batches of images and their correspoding target vectors for use with 
    an SSD object detection model.

    Args:
        directory (string or pathlib.Path): Path to a directory 
          containing images.
        annotations (dict): Dictionary containing object annotations for
          every image in `directory`. The keys of this dictionary must 
          be the filenames (NOT the complete path) of the images, and 
          the values must be dictionaries having this structure:
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
        ssd_target_vector_builder (SSDTargetVectorBuilder): Instance of 
          the `SSDTargetVectorBuilder` class used to build target 
          vectors from object annotations.
        mode (str, optional): Mode in which this iterator will be used.
          Passing 'train' will instruct the iterator to operate in 
          training mode: in this case, the images are resized to fixed 
          size by means of the `fixed_size_image_provider` parameter (in
          which case it cannot be None) before they are returned. Notice 
          that if the images are meant to be fed to a fully-convolutional 
          network (such as SSD), this is not strictly necessary as such 
          type of networks can handle inputs of different sizes. However,
          images of varying sizes cannot be organized in batches, which 
          means that the training process would become terribly 
          inefficient. On the other hand, passing 'eval' will tell the 
          iterator to operate in evaluation mode: in this case, images 
          are returned as they are, which forces `batch_size` to be 
          automatically set to 1 for the reasons discussed above (the 
          value passed to `batch_size` is ignored). Defaults to 'train'.
        fixed_size_image_provider (FixedSizeImageProvider, optional): 
          Instance of a subclass of `FixedSizeImageProvider` used to 
          turn arbitrarily-sized images into fixed-size images. Must be 
          specified when mode='train'. Ignored when mode='eval'. 
          Defaults to None.
        min_image_size (int, optional): The minimum size of an image. If 
          the short side of the image is less than `min_image_size`, the
          image is resized so that its short side is `min_image_size` 
          (the original aspect ratio is preserved). Ignored when 
          mode='train'. Defaults to None.
        input_preprocessing_function (function, optional): Function used 
          to preprocess the images. Defaults to None.
        batch_size (int, optional): Number of images in each batch. 
          Automatically set to 1 when mode='eval'. Defaults to 32.
        shuffle (bool, optional): Whether or not to return the images in 
          a random order. Defaults to True.
        seed (int, optional): Random seed used when determining the 
          random order in which the images will be returned (only if 
          shuffle=True). Defaults to None.

    Raises:
        ValueError: `mode` is neither 'train' nor 'eval'.
        ValueError: `mode` is 'train' but `fixed_size_image_provider` is 
          None.
    """

    def __init__(
        self, 
        directory, 
        annotations,
        ssd_target_vector_builder,
        mode='train',
        fixed_size_image_provider=None,
        min_image_size=None,
        input_preprocessing_function=None,
        batch_size=32,
        shuffle=True,        
        seed=None
    ):
        if mode not in ['train', 'eval']:
            raise ValueError(
                f'`mode` must be either "train" or "eval" (got "{mode}").'
            )

        if mode == 'train' and fixed_size_image_provider is None:
            raise ValueError(
                '`fixed_size_image_provider` must be specified when'
                'mode="train".'
            )

        if mode == 'eval':
            batch_size = 1
            if fixed_size_image_provider is not None:
                tf.get_logger().warning(
                    'Since you passed mode="eval", the value for '
                    'parameter `fixed_size_image_provider` will be '
                    'ignored. Also, `batch_size` will be automatically '
                    'set to 1.'
                )

        self.annotations = annotations
        self.ssd_target_vector_builder = ssd_target_vector_builder
        self.mode = mode
        self.fixed_size_image_provider = fixed_size_image_provider
        self.min_image_size = min_image_size
        self.input_preprocessing_function = input_preprocessing_function
        self.filepaths = list(Path(directory).iterdir())

        super(SSDDirectoryIterator, self).__init__(
            len(self.annotations),
            batch_size,
            shuffle,
            seed
        )

    # Overridden method
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        Args:
            index_array (list(int)): List of sample indices to include 
              in batch.

        Returns:
            tuple(numpy.array, tuple): Pair containing a batch of images 
              and a batch of target vectors, where the latter is itself 
              a pair consisting of a batch of target vectors for 
              classification and a batch of target vectors for 
              localization.
        """     

        batch_x = []
        batch_y_conf = []
        batch_y_loc = []
        # Build batches of (x, y) pairs, where y = (y_conf, y_loc)
        for i in index_array:
            file_path = Path(self.filepaths[i])
            # Load an image and its object annotations
            img = load_img(str(file_path))
            annotation = self.annotations[file_path.name]
            if self.mode == 'train':
                # Turn `img` (whose size is arbitrary) into a fixed-size
                # image. This allows to organize images in batches
                res = self.fixed_size_image_provider.extract_fixed_size_image(
                    img, 
                    annotation
                )
                img, annotation = res
            elif self.min_image_size is not None: # (and self.mode == 'eval')
                short_side = min(img.width, img.height)
                if short_side < self.min_image_size:                
                    if short_side == img.width:
                        new_size = (
                            self.min_image_size, 
                            int((img.height / img.width) * self.min_image_size)
                        )
                    else: # short_side == img.height
                        new_size = (
                            int((img.width / img.height) * self.min_image_size),
                            self.min_image_size
                        )
                    # Resize the image and adjust the bounding box 
                    # annotations
                    img = img.resize(new_size)
                    annotation = adjust_bounding_box_annotations(
                        annotation, 
                        new_size
                    )
            
            x = img_to_array(img)
            if self.input_preprocessing_function is not None:
                # Preprocess the image using the provided function
                x = self.input_preprocessing_function(x)
            # Build the target vector for this particular image
            y_conf, y_loc = self.ssd_target_vector_builder.build_target_vector(
                annotation
            )
            # Append the image and the target vectors to their 
            # respective batches
            batch_x.append(x)
            batch_y_conf.append(y_conf)
            batch_y_loc.append(y_loc)
            # Pillow images should be closed after `load_img`,but not 
            # PIL images
            if hasattr(img, 'close'):
                img.close()

        batch_x_arr = np.array(batch_x, dtype=np.float32)
        batch_y_conf_arr = np.array(batch_y_conf, dtype=np.float32)
        batch_y_loc_arr = np.array(batch_y_loc, dtype=np.float32)
        return batch_x_arr, (batch_y_conf_arr, batch_y_loc_arr)
