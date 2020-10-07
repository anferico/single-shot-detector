import math

from boxes import BoundingBox

class DefaultBoxesGenerator(object):
    """Class for generating default boxes for use with an SSD object 
    detection model.

    Args:
        feature_map_sizes_extractor (function): Function that takes two 
          integer arguments, namely the height and the width of an 
          image, and returns a list of pairs representing the sizes of 
          the feature maps used by the SSD model to make predictions.
        default_boxes_aspect_ratios (list(float)): List containing the 
          aspect ratios to consider for each default box.
        min_scale (float, optional): Minimum scale of the default boxes 
          (relative to the size of the image). This will be the scale of
          the default boxes associated with the first feature map used 
          to make predictions. Defaults to 0.2.
        max_scale (float, optional): Maximum scale of the default boxes 
          (relative to the size of the image). This will be the scale of
          the default boxes associated with the last feature map used to 
          make predictions. Defaults to 0.9.
    """

    def __init__(
        self, 
        feature_map_sizes_extractor,
        default_boxes_aspect_ratios,
        min_scale=0.2,
        max_scale=0.9
    ):
        self.feature_map_sizes_extractor = feature_map_sizes_extractor
        self.default_boxes_aspect_ratios = default_boxes_aspect_ratios
        self.min_scale = min_scale 
        self.max_scale = max_scale 


    def generate_default_boxes(self, image_height, image_width):
        """Generates default boxes for an image of the given size.

        Args:
            image_height (int): The image's height.
            image_width (int): The image's width.

        Yields:
            BoundingBox: The default boxes for an image of the given 
              size, returned one at a time. The order in which they are 
              yielded is from the first feature map to the last one. 
              Within a single feature map, the boxes are yielded 
              left-to-right, top-to-bottom.
        """

        feature_map_sizes = self.feature_map_sizes_extractor(
            image_height, 
            image_width
        )
        n_feature_maps = len(feature_map_sizes)
        # Compute the relative scales of the default boxes. Boxes 
        # associated with higher-resolution feature maps will have a 
        # smaller scale, whereas boxes associated with lower-resolution 
        # feature maps are assigned a larger scale
        def_boxes_rel_scales = [self.min_scale] + [
            (self.min_scale 
             + (self.max_scale - self.min_scale) * k / (n_feature_maps - 1))
            for k in range(1, n_feature_maps)
        ]
        # TODO: Find a way to include the additional scale for the 
        # special case of when the aspect ratio is 1:1
        for k, (height, width) in enumerate(feature_map_sizes):
            for i in range(height):
                for j in range(width):
                    # Compute the center of the bounding boxes 
                    # associated with the grid cell of coordinates (i,j)
                    # of the k-th feature map. Note that all such boxes 
                    # have the same center, irrespective of their aspect 
                    # ratios
                    boxes_center_x = ((j + 0.5) / width) * image_width                    
                    boxes_center_y = ((i + 0.5) / height) * image_height
                    for r in self.default_boxes_aspect_ratios:
                        # Compute the width and height of a default box 
                        # with aspect ratio `r`.
                        # NOTE: In the SSD paper, the authors don't 
                        # mention how to deal with non-square input 
                        # images (e.g. at test time). The 
                        # `rel_scale` appearing below in the computation 
                        # of `abs_scale` represents the scale of the 
                        # default box relative to the size of the input 
                        # image. If the input image is square, then the 
                        # absolute scale of the default box can be 
                        # obtained by multiplying the side length of the 
                        # image by the relative scale. Once the absolute 
                        # scale is computed, we can use it to set the 
                        # width and height of the default box so that 
                        # its aspect ratio is exactly `r` (check how 
                        # `box_width` and `box_height` are computed 
                        # below). 
                        # But what if the input image is non-square? In 
                        # such case, we would have two absolute scales 
                        # for each default box: an horizontal scale and 
                        # a vertical scale. It is easy to see that if we 
                        # compute the width and the height of the 
                        # default box as we did when considering square 
                        # input images, its aspect ratio would not be 
                        # `r`, but rather `r` times the aspect ratio of 
                        # the input image. 
                        # With this in mind, my suggestion is to 
                        # multiply the relative scale of the default box 
                        # by the short side of the image so that the 
                        # aspect ratio we get is exactly `r`. Notice 
                        # that we could use the long side instead and 
                        # achieve the same result. 
                        # The key point here is the following: if you 
                        # have a 1000x300 image, would you want to have 
                        # default boxes that would normally be used for 
                        # a 1000x1000 image, or would you rather have 
                        # default boxes that would normally be used for 
                        # 300x300 images? If you think the first option 
                        # is the best, then multiply the relative scale 
                        # by the long side of the image, otherwise 
                        # multiply it by the short side. 
                        # Another option could be to generate 2 default 
                        # boxes for each aspect ratio, at which point 
                        # you could multiply the scale of one of them by 
                        # the short side and the other by the long side.
                        # Yet another possibility would be to multiply 
                        # the scale by the average of the two sides                  
                        short_side = min(
                            image_width, 
                            image_height
                        )
                        rel_scale = def_boxes_rel_scales[k]
                        abs_scale = rel_scale * short_side
                        box_width = abs_scale * math.sqrt(r)
                        box_height = abs_scale / math.sqrt(r)
                        # Create an instance of the `BoundingBox` class 
                        # representing the default box
                        default_box = BoundingBox(
                            boxes_center_x - box_width/2,
                            boxes_center_y - box_height/2,
                            boxes_center_x + box_width/2,
                            boxes_center_y + box_height/2
                        )
                        yield default_box
