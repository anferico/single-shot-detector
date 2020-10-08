# Single Shot MultiBox Detector (TensforFlow 2 & Keras)
Keras implementation of [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325). Unlike the original SSD model, which uses **VGG16** as the "base network" for feature extraction, I'm using **Inception-V3** here (specifically, I've used the version with pre-trained weights available under [`tf.keras.applications.InceptionV3`](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)). 

Apart from using an alternative base network, there are a couple other differences between this version and the original SSD model:
 - The data augmentation strategy used in this project is very similar, although not exactly the same, to the one mentioned in the paper
 - I've used label smoothing for classification labels, whereas the authors of the paper did not

## Train the model
In order to train the model on the Pascal VOC 2007 dataset, simply run [`train_model.py`](train_model.py) providing the following arguments:
```
-ti, --training-images-directory DIRECTORY         # Directory containing the training images
-ta, --training-annotations-directory DIRECTORY    # Directory containing the annotations for training images, in XML format
-vi, --validation-images-directory DIRECTORY       # Directory containing the validation images
-va, --validation-annotations-directory DIRECTORY  # Directory containing the annotations for validation images, in XML format
-vs, --validation-steps INTEGER                    # Number of validation samples to run through the network at the end of each epoch to estimate the validation error
-c, --config-file FILE                             # Configuration file containing training parameters, in JSON format
-w, --pretrained-weights FILE                      # Binary file containing pre-trained weights, in H5 format. Useful to resume training. If not specified, the weights will be initialized randomly
-o, --output-dir DIRECTORY                         # Directory where the final weights and the training history will be saved
```
See [`config.json`](config.json) for an example of a configuration file.

## Pre-trained weights
Pre-trained weights can be found here: <insert_link>

## Customize the model
As mentioned before, the base network used for this project is **Inception-V3**, whereas the dataset used for training the model is **Pascal VOC 2007**. You can, however, use whichever base network and dataset you wish.

### Custom base network
In order to use a custom base network, you'd have to implement two simple functions: one for creating the model and one for extracting the size of the feature maps used to make predictions, given an image size. You can see how those two functions were implemented for the SSD Inception-V3 model by opening [`ssd_inception_v3.py`](ssd_inception_v3.py).

### Custom training dataset
On the other hand, if you wish to use a custom dataset, you'd have to provide one or more functions to parse ground-truth annotations. In this case, since ground-truth annotations in the Pascal VOC 2007 dataset are in XML format, I've implemented a simple function that parses the content of an XML file (see [`voc_utils.py`](voc_utils.py)). If, for example, you were to train the model on the COCO dataset, you'd implement a function that parses the content of a JSON file. 
Finally, ground truth annotations need to be organized as a Python dictionary with the following structure:
```Python
{
    'first_image.jpg': {
        'width': 500 # Width of the image the annotation refers to
        'height': 375 # Height of the image the annotation refers to
        'objects': [
            {
                'class': 'cat' # The class label for this object
                'bounding_box': BoundingBox(
                    50,  # x_min
                    100, # y_min
                    320, # x_max
                    180  # y_max
                ) # The bounding box for this object. It must be a `BoundingBox` instance
            },
            ... # Other objects appearing in the image
        ]
    },
    'second_image.jpg': {
          ... # As above
    },
    ... # The remaining images
}
```
