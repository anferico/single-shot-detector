from xml.etree import ElementTree 
from pathlib import Path

from boxes import BoundingBox

background_class_name = 'background'
classes = [
    # Default class
    background_class_name,
    # People
    'person',
    # Animals
    'bird', 
    'cat', 
    'cow', 
    'dog', 
    'horse', 
    'sheep',
    # Vehicles
    'aeroplane', 
    'bicycle', 
    'boat', 
    'bus', 
    'car', 
    'motorbike', 
    'train',
    # Objects
    'bottle', 
    'chair', 
    'diningtable', 
    'pottedplant', 
    'sofa', 
    'tvmonitor'
]
n_classes = len(classes)


def get_background_class_name():
    return background_class_name


def get_number_of_classes():
    return n_classes


def get_index_to_class_map():
    return dict(enumerate(classes))


def get_class_to_index_map():
    return dict(zip(classes, range(len(classes))))


def parse_single_VOC_annotation(annotation_file):
    """Parses a single XML file representing an annotation in the Pascal 
    VOC format.

    Args:
        annotation_file (string or file object): Either a path to an XML 
          file containing an annotation in the Pascal VOC format or a 
          file object.

    Returns:
        tuple(string, dict): Pair containing the name of the file and a 
          dictionary having the following structure:
          {
              'width': Width of the image the annotation refers to,
              'height': Height of the image the annotation refers to,
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
    """

    annotation = ElementTree.parse(annotation_file).getroot()
    filename = annotation.find('filename').text
    image_width = int(annotation.find('size/width').text)
    image_height = int(annotation.find('size/height').text)
    objects = []
    for obj in annotation.findall('object'):
        class_ = obj.find('name').text
        x_min, y_min, x_max, y_max = obj.find('bndbox')
        bounding_box = BoundingBox(
            int(x_min.text),
            int(y_min.text),
            int(x_max.text),
            int(y_max.text)
        )
        objects.append({
            'class': class_,
            'bounding_box': bounding_box
        })
        
    return filename, {
        'width': image_width,
        'height': image_height,
        'objects': objects
    }


def parse_VOC_annotations(annotations_dir):
    """Parses all the annotation files contained in `annotations_dir`. 
    The annotations must be XML files in the Pascal VOC format.

    Args:
        annotations_dir (string or pathlib.Path): Path to a directory 
          containing annotation files in the Pascal VOC format.

    Raises:
        ValueError: `annotations_dir` does not exist.
        ValueError: `annotations_dir` is not a directory.

    Returns:
        dict: Dictionary mapping filenames of the files contained in 
          `annotations_dir` to dictionaries having the following 
          structure:
          {
              'width': Width of the image the annotation refers to,
              'height': Height of the image the annotation refers to,
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
    """

    annotations_dir = Path(annotations_dir)
    if not annotations_dir.exists():
        raise ValueError(
            f'Directory "{annotations_dir}" does not exist.'
        )
    if not annotations_dir.is_dir():
        raise ValueError(
            '`annotations_dir` must be a directory containing XML '
            'annotations files in VOC format.'
        )

    annotations = {}
    for xml_file in annotations_dir.iterdir():
        filename, annotation = parse_single_VOC_annotation(xml_file)
        annotations[filename] = annotation

    return annotations
