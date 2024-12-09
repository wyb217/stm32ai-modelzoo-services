# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import tensorflow as tf
from random_utils import check_dataaug_argument
from segm_random_utils import segm_apply_change_rate


def check_segm_random_crop_arguments(crop_center_x, crop_center_y, crop_width, crop_height, interpolation):

    def check_value_range(arg_value, arg_name):
        if isinstance(arg_value, (tuple, list)):
            if arg_value[0] <= 0 or arg_value[0] >= 1 or arg_value[1] <= 0 or arg_value[1] >= 1:
                raise ValueError(f"\nArgument `{arg_name}` of function `segm_random_crop`: expecting "
                                 f"float values greater than 0 and less than 1. Received {arg_value}")
        else:
            if arg_value <= 0 or arg_value >= 1:
                raise ValueError(f"\nArgument `{arg_name}` of function `segm_random_crop`: expecting "
                                 f"float values greater than 0 and less than 1. Received {arg_value}")
    
    check_dataaug_argument(crop_center_x, "crop_center_x", function_name="segm_random_crop", data_type=float, tuples=2)
    check_value_range(crop_center_x, "crop_center_x")
    
    check_dataaug_argument(crop_center_y, "crop_center_y", function_name="segm_random_crop", data_type=float, tuples=2)
    check_value_range(crop_center_y, "crop_center_y")

    check_dataaug_argument(crop_width, "crop_width", function_name="segm_random_crop", data_type=float, tuples=1)
    check_value_range(crop_width, "crop_width")
        
    check_dataaug_argument(crop_height, "crop_height", function_name="segm_random_crop", data_type=float, tuples=1)
    check_value_range(crop_height, "crop_height")
    
    if interpolation not in ("bilinear", "nearest"):
        raise ValueError("\nArgument `interpolation` of function `segm_random_crop`: expecting "
                         f"either 'bilinear' or 'nearest'. Received {interpolation}")

def segm_random_crop(
            images: tf.Tensor,
            labels: tf.Tensor,
            crop_center_x: tuple = (0.25, 0.75),
            crop_center_y: tuple = (0.25, 0.75),
            crop_width: float = (0.6, 0.9),
            crop_height: float = (0.6, 0.9),
            interpolation: str = "bilinear",
            change_rate: float = 0.9) -> tuple:
            
    """
    This function randomly crops input images and their associated  bounding boxes.
    The output images have the same size as the input images.
    We designate the portions of the images that are left after cropping
    as 'crop regions'.
    
    Arguments:
        images:
            Input images to crop.
            Shape: [batch_size, width, height, channels]
        labels:
            Labels associated to the images. The class is first, the bounding boxes
            coordinates are (x1, y1, x2, y2) absolute coordinates.
            Shape: [batch_size, num_labels, 5]
        crop_center_x:
            Sampling range for the x coordinates of the centers of the crop regions.
            A tuple of 2 floats between 0 and 1.
        crop_center_y:
            Sampling range for the y coordinates of the centers of the crop regions.
            A tuple of 2 floats between 0 and 1.
        crop_width:
            Sampling range for the widths of the crop regions. A tuple of 2 floats
            between 0 and 1.
            A single float between 0 and 1 can also be used. In this case, the width 
            of all the crop regions will be equal to this value for all images.
        crop_height:
            Sampling range for the heights of the crop regions. A tuple of 2 floats
            between 0 and 1.
            A single float between 0 and 1 can also be used. In this case, the height 
            of all the crop regions will be equal to this value for all images.
        interpolation:
            Interpolation method to resize the cropped image.
            Either 'bilinear' or 'nearest'.
        change_rate:
            A float in the interval [0, 1], the number of changed images
            versus the total number of input images average ratio.
            For example, if `change_rate` is set to 0.25, 25% of the input
            images will get changed on average (75% won't get changed).
            If it is set to 0.0, no images are changed. If it is set
            to 1.0, all the images are changed.

    Returns:
        cropped_images:
            The cropped images.
            Shape: [batch_size, width, height, channels]
        cropped_labels:
            Labels with cropped bounding boxes.
            Shape: [batch_size, num_labels, 5]
    """
    
    # Check the function arguments
    check_segm_random_crop_arguments(crop_center_x, crop_center_y, crop_width, crop_height, interpolation)

    if not isinstance(crop_width, (tuple, list)):
        crop_width = (crop_width, crop_width)
    if not isinstance(crop_height, (tuple, list)):
        crop_height = (crop_height, crop_height)

    # Sample the coordinates of the center, width and height of the crop regions
    batch_size = tf.shape(images)[0]
    crop_center_x = tf.random.uniform([batch_size], crop_center_x[0], maxval=crop_center_x[1], dtype=tf.float32)
    crop_center_y = tf.random.uniform([batch_size], crop_center_y[0], maxval=crop_center_y[1], dtype=tf.float32)
    crop_width = tf.random.uniform([batch_size], crop_width[0], maxval=crop_width[1], dtype=tf.float32)
    crop_height = tf.random.uniform([batch_size], crop_height[0], maxval=crop_height[1], dtype=tf.float32)

    # Calculate and clip the (x1, y1, x2, y2) normalized
    # coordinates of the crop regions relative to the
    # upper-left corners of the images
    x1 = tf.clip_by_value(crop_center_x - crop_width/2, 0, 1)
    y1 = tf.clip_by_value(crop_center_y - crop_height/2, 0, 1)
    x2 = tf.clip_by_value(crop_center_x + crop_width/2, 0, 1)
    y2 = tf.clip_by_value(crop_center_y + crop_height/2, 0, 1)

    # Crop the input images and resize them to their initial size
    image_size = tf.shape(images)[1:3]
    crop_regions = tf.stack([y1, x1, y2, x2], axis=-1) 
    crop_region_indices = tf.range(batch_size)
    cropped_images = tf.image.crop_and_resize(images, crop_regions, crop_region_indices,
                                              crop_size=image_size, method=interpolation)
    
    # Crop the input labels and resize them to their initial size
    cropped_labels = tf.image.crop_and_resize(labels, crop_regions, crop_region_indices,
                                              crop_size=image_size, method=interpolation)

    # Apply the change rate to images and labels
    images_aug, labels_aug = segm_apply_change_rate(
            images, labels, cropped_images, cropped_labels, change_rate=change_rate)

    return images_aug, labels_aug


def segm_random_periodic_resizing(
                images,
                labels,
                period=None,
                image_sizes=None,
                interpolation=None,
                batch=None,
                last_image_size=None):
    """
    This function periodically resizes the input images. The size of
    the images is held constant for a specified number of batches,
    referred to as the "resizing period". Every time a period ends,
    a new size is sampled from a specified set of sizes. Then, the
    size is held constant for the next period, etc.
    
    This function is intended to be used with the 'data_augmentation.py'
    package as it needs the current batch number and the size of the
    images of the previous batch.
    
    Arguments:
        images:
            Input RGB or grayscale images, a tensor with shape
            [batch_size, width, height, channels]. 
        period:
            An integer, the resizing period.
        image_sizes:
            A tuple or list of integers, the set of sizes the image
            sizes are sampled from.
        interpolation:
            A string, the interpolation method used to resize the images.
            Supported values are "bilinear", "nearest", "area", "gaussian",
            "lanczos3", "lanczos5", "bicubic" and "mitchellcubic"
            (resizing is done using the Tensorflow tf.image.resize() function).
        batch:
            An integer, the current batch number starting from the beginning
            of the training.
        last_image_size:
            An tuple of 2 integers, the size of the images of the previous
            batch of images.

    Returns:
        The periodally resized images.
    """

    # Check arguments
    check_dataaug_argument(period, "period", function_name="random_periodic_resizing",
                           data_type=int, tuples=0)

    input_image_size = tf.shape(images)[1:3]
    
    # Sample an image size
    num_sizes = tf.shape(image_sizes)[0]
    i = tf.random.uniform([1], minval=0, maxval=num_sizes, dtype=tf.int32)[0]
    random_size = tf.gather(image_sizes, i)

    # Use the random size if the batch number is a multiple 
    # of the resizing period. Otherwise, keep the size 
    # of the previous batch.
    cond = tf.logical_and(batch > 0, batch % period == 0)
    target_size = tf.where(cond, random_size, last_image_size)

    def resize_images_and_labels():
        resized_images = tf.image.resize(images, [target_size[1], target_size[0]], method=interpolation)
        resized_labels = tf.image.resize(labels, [target_size[1], target_size[0]], method=interpolation)
        return resized_images, resized_labels
        
    def keep_original_size():
        return images, labels

    # Resize the images and GT boxes if the target size
    # is not the same as the size of the input images
    images, labels = tf.cond(tf.reduce_any(tf.not_equal(target_size, input_image_size)),
                             resize_images_and_labels, keep_original_size)

    return images, labels
