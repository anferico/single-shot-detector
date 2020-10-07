from pathlib import Path
import json

from tensorflow.keras.applications import inception_v3
import tensorflow as tf
import pandas as pd
import click

from ssd_inception_v3 import (
    get_ssd_inception_v3_feature_map_sizes,
    build_ssd_inception_v3_model
)
from voc_utils import (
    parse_VOC_annotations, 
    get_background_class_name, 
    get_class_to_index_map
)
from ssd_target_vector_builder import SSDTargetVectorBuilder
from ssd_common import confidence_loss, localization_loss
from default_boxes_generator import DefaultBoxesGenerator
from fixed_size_image_provider import SSDImageAugmentator
from ssd_directory_iterator import SSDDirectoryIterator 

@click.command()
@click.option(
    '-ti', '--training-images-directory',
    type=click.Path(exists=True, file_okay=False),
    default='/home/anfri/Downloads/Torrent/voc2007/trainval/VOC2007/JPEGImages'
)
@click.option(
    '-ta', '--training-annotations-directory',
    type=click.Path(exists=True, file_okay=False),
    default='/home/anfri/Downloads/Torrent/voc2007/trainval/VOC2007/Annotations'
)
@click.option(
    '-vi', '--validation-images-directory',
    type=click.Path(exists=True, file_okay=False),
    default='/home/anfri/Downloads/Torrent/voc2007/test/VOC2007/JPEGImages'
)
@click.option(
    '-va', '--validation-annotations-directory',
    type=click.Path(exists=True, file_okay=False),
    default='/home/anfri/Downloads/Torrent/voc2007/test/VOC2007/Annotations'
)
@click.option('-vs', '--validation-steps', type=int, default=32)
@click.option(
    '-c', '--config-file',
    type=click.Path(exists=True, dir_okay=False),
    default='/home/anfri/MyProjects/single-shot-detector/config.json'
)
@click.option(
    '-w', '--pretrained-weights',
    type=click.Path(exists=True, dir_okay=False),
    default=None
)
@click.option(
    '-o', '--output-dir',
    type=click.Path(file_okay=False),
    default='/home/anfri'
)
def main(
    training_images_directory, 
    training_annotations_directory, 
    validation_images_directory, 
    validation_annotations_directory,
    validation_steps,
    config_file,
    pretrained_weights, 
    output_dir
):
    train_annotations = parse_VOC_annotations(training_annotations_directory)
    val_annotations = parse_VOC_annotations(validation_annotations_directory)
    with open(config_file, 'r') as fp:
        config = json.load(fp)

    def_boxes_gen = DefaultBoxesGenerator(
        feature_map_sizes_extractor=get_ssd_inception_v3_feature_map_sizes,
        default_boxes_aspect_ratios=config['aspect_ratios'],
        min_scale=config['min_scale'],
        max_scale=config['max_scale']
    )

    class_to_index_map = get_class_to_index_map()
    target_vector_builder = SSDTargetVectorBuilder(
        default_boxes_generator=def_boxes_gen,
        iou_threshold=config['iou_threshold'],
        class_to_index_map=class_to_index_map,
        background_class_name=get_background_class_name(),
        label_smoothing_factor=0.1
    )

    image_augmentator = SSDImageAugmentator(
        target_width=config['input_size'],
        target_height=config['input_size'],
        min_rel_crop_size=0.1,
        contrast_range=0.25,
        brightness_range=0.25,
        channel_shift=True,
        flip_prob=0.5,
        seed=None
    )

    train_iterator = SSDDirectoryIterator(
        directory=training_images_directory,
        annotations=train_annotations,
        ssd_target_vector_builder=target_vector_builder,
        mode='train',
        fixed_size_image_provider=image_augmentator,
        input_preprocessing_function=inception_v3.preprocess_input,
        batch_size=config['batch_size'],
        shuffle=True,
        seed=None
    )
    # Unlike training data, validation data cannot be passed to fit(...) 
    # in the form of an iterator. Instead, we must wrap it in a 
    # `tf.data.Dataset` instance
    val_iterator = lambda: SSDDirectoryIterator(
        directory=validation_images_directory,
        annotations=val_annotations,
        ssd_target_vector_builder=target_vector_builder,
        mode='eval',
        min_image_size=267,
        input_preprocessing_function=inception_v3.preprocess_input,
        shuffle=True
    )
    val_dataset = tf.data.Dataset.from_generator(
        val_iterator,
        output_types=(tf.float32, (tf.float32, tf.float32))
    )

    # Create the model
    n_classes = len(class_to_index_map)
    model = build_ssd_inception_v3_model(
        n_classes=n_classes,
        n_default_boxes=len(config['aspect_ratios']), 
        l2_regularization=config['l2_regularization'],
        weights=pretrained_weights
    )

    # Compile and train the model
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=config['learning_rate'], 
        momentum=config['momentum']
    )
    confidence_loss_closure = lambda y_true, y_pred: confidence_loss(
        y_true,
        y_pred,
        n_classes=n_classes,
        background_class_id=0,
        hard_negative_mining=config['hard_negative_mining'],
        negatives_to_positives_ratio=3.0
    )
    model.compile(
        optimizer=optimizer,
        loss=[confidence_loss_closure, localization_loss],
        loss_weights=[1.0, config['localization_loss_weight']]
    )
    train_history = model.fit(
        train_iterator, 
        epochs=1, #TODO: config['epochs'], 
        validation_data=val_dataset,
        validation_steps=validation_steps
    )
    
    # Write outputs to disk
    output_dir = Path(output_dir)
    train_history_df = pd.DataFrame(train_history.history)
    train_history_df.to_csv(output_dir/'train_history.csv', index=False)
    model.save_weights(output_dir/'weights.h5')

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
