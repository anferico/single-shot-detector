import math

class BoundingBox(object):

    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max        

        self.width = x_max - x_min
        self.height = y_max - y_min
        self.center_x = x_min + self.width/2.0
        self.center_y = y_min + self.height/2.0

    
    @classmethod
    def from_cx_cy_w_h(cls, center_x, center_y, width, height):
        x_min = center_x - width/2.0
        y_min = center_y - height/2.0
        x_max = x_min + width
        y_max = y_min + height
        return cls(x_min, y_min, x_max, y_max)


    def __repr__(self):
        return ('BoundingBox('
                  f'x_min={self.x_min}, '
                  f'y_min={self.y_min}, '
                  f'x_max={self.x_max}, '
                  f'y_max={self.y_max})')


    def __eq__(self, other):
        return (abs(self.x_min - other.x_min) < 1e-8
                and abs(self.y_min - other.y_min) < 1e-8
                and abs(self.x_max - other.x_max) < 1e-8
                and abs(self.y_max - other.y_max) < 1e-8)


    def get_area(self):
        """Computes the area (in pixels²) of this bounding box.

        Returns:
            float: The area of this bounding box.
        """

        return self.width * self.height


    def intersection_over_union(self, other_box):
        """Computes the intersection over union (IoU) between two boxes.

        Args:
            other_box (BoundingBox): The box used to compute the IoU 
              with this box.

        Returns:
            float: Intersection over union between this box and 
              `other_box`. The value will be between 0 and 1.
        """
        
        offset_x = (min(self.x_max, other_box.x_max) 
                    - max(self.x_min, other_box.x_min))
        offset_y = (min(self.y_max, other_box.y_max) 
                    - max(self.y_min, other_box.y_min))
        intersection_width = max(0, offset_x)
        intersection_height = max(0, offset_y)
        intersection_area = intersection_width * intersection_height
        union_area = self.get_area() + other_box.get_area() - intersection_area
        return intersection_area / union_area


    def deviation_relative_to(self, other_box):
        """Computes values that indicate how much this box deviates from
        the given box. 

        Args:
            other_box (BoundingBox): Bounding box you wish to compute 
              deviations from.

        Returns:
            Tuple: Quadruplet of the form (d_cx, d_cy, d_w, d_h) 
              containing deviations from the center (first two elements) 
              and from the width/height (last two elements) of 
              `other_box`.
        """        

        return (
            (self.center_x - other_box.center_x) / other_box.width, 
            (self.center_y - other_box.center_y) / other_box.height, 
            math.log(self.width / other_box.width), 
            math.log(self.height / other_box.height)
        )


    def apply_deviations(self, dev_x, dev_y, dev_w, dev_h):
        """Applies the given deviations (returned by a call to 
        `deviation_relative_to`) to this box.

        Args:
            dev_x (float): Deviation from the x-coordinate of this box's 
              center, as computed by `deviations_relative_to`.
            dev_y (float): Deviation from the y-coordinate of this box's 
              center, as computed by `deviations_relative_to`.
            dev_w (float): Deviation from this box's width, as computed 
              by `deviations_relative_to`.
            dev_h (float): Deviation from this box's height, as computed 
              by `deviations_relative_to`.

        Returns:
            BoundingBox: The bounding box resulting from applying the 
              given deviations to a copy of this box (the original box 
              is not modified).
        """

        return BoundingBox.from_cx_cy_w_h(
            (dev_x * self.width) + self.center_x,
            (dev_y * self.height) + self.center_y,
            math.exp(dev_w) * self.width,
            math.exp(dev_h) * self.height
        )
