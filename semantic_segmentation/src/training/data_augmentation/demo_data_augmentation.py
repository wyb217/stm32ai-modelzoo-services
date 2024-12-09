# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
import argparse
from munch import DefaultMunch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../common/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../common/data_augmentation'))
sys.path.append(os.path.abspath('../../utils'))
sys.path.append(os.path.abspath('../../preprocessing'))
sys.path.append(os.path.abspath('../../data_augmentation'))
sys.path.append(os.path.abspath('../../models'))

from data_loader import get_ds
import random_color, random_misc, segm_random_affine, segm_random_misc
from random_utils import remap_pixel_values_range
from data_augmentation import data_augmentation


def plot_images_and_labels(image, label, image_aug, label_aug, grayscale=None, legend=None):
    """
    Displays side by side the original and augmented image
    with their groundtruth bounding boxes.

    Arguments:
        images:
            Original image.
            A numpy array with shape [width, height, channels]
        labels:
            Groundtruth labels of the original image.
            A numpy array with shape [num_labels, 5]
        images_aug:
            Augmented images.
            A numpy rray with shape [width, height, channels]
        images_aug:
            Groundtruth labels of the augmented images.
            A numpy array with shape [num_labels, 5]
        class_names:
            A list of strings, the class names.
            
    Returns:
        None
    """

    # Calculate the dimensions of the displayed images
    image_width, image_height = np.shape(image)[:2]
    display_size = 9
    if image_width >= image_height:
        x_size = display_size
        y_size = round((image_width / image_height) * display_size)
    else:
        y_size = display_size
        x_size = round((image_height / image_width) * display_size)

    fig, axes = plt.subplots(2, 2, figsize=(x_size, y_size))
    (ax1, ax2), (ax3, ax4) = axes

    if grayscale:
        ax1.imshow(image, cmap='gray')
        ax1.title.set_text("Original")
        ax2.imshow(image_aug, cmap='gray')
    else:
        ax1.imshow(image)
        ax1.title.set_text("Original")
        ax2.imshow(image_aug)

    ax2.title.set_text(legend)
    ax3.imshow(label)
    ax4.imshow(label_aug)
    ax3.imshow(label)
    ax4.imshow(label_aug)
        
    plt.show()
    plt.close()
    
    
def augment_images(images, labels, fn_name=None):

    if fn_name == "random_contrast":
        images_aug = random_color.random_contrast(images, factor=0.7)
        return images_aug, labels
        
    elif fn_name == "random_brightness":
        images_aug = random_color.random_brightness(images, factor=0.4)
        return images_aug, labels

    elif fn_name == "random_gamma":
        images_aug = random_color.random_gamma(images, gamma=(0.2, 2.0))
        return images_aug, labels

    elif fn_name == "random_hue":
        images_aug = random_color.random_hue(images, delta=0.1)
        return images_aug, labels

    elif fn_name == "random_saturation":
        images_aug = random_color.random_saturation(images, delta=0.2)
        return images_aug, labels

    elif fn_name == "random_value":           
        images_aug = random_color.random_value(images, delta=0.2)        
        return images_aug, labels 
        
    elif fn_name == "random_hsv":
        images_aug = random_color.random_hsv(
                        images, hue_delta=0.1, saturation_delta=0.2, value_delta=0.2)
        return images_aug, labels

    elif fn_name == "random_rgb_to_hsv":
        images_aug = random_color.random_rgb_to_hsv(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_rgb_to_grayscale":        
        images_aug = random_color.random_rgb_to_grayscale(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_sharpness":
        images_aug = random_color.random_sharpness(images, factor=(1.0, 4.0))
        return images_aug, labels

    elif fn_name == "random_posterize":
        images_aug = random_color.random_posterize(images, bits=(1, 8))
        return images_aug, labels

    elif fn_name == "random_invert":
        images_aug = random_color.random_invert(images, change_rate=1.0)
        return images_aug, labels 

    elif fn_name == "random_solarize":
        images_aug = random_color.random_solarize(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_autocontrast":
        images_aug = random_color.random_autocontrast(images, change_rate=1.0)
        return images_aug, labels

    elif fn_name == "random_blur":
        images_aug = random_misc.random_blur(images, filter_size=(2, 4))
        return images_aug, labels 

    elif fn_name == "random_gaussian_noise":
        images_aug = random_misc.random_gaussian_noise(images, stddev=(0.02, 0.1))
        return images_aug, labels 

    elif fn_name == "random_crop":
        return segm_random_misc.segm_random_crop(images, labels, change_rate=1.0)

    elif fn_name == "random_flip":
        return segm_random_affine.segm_random_flip(images, labels, mode="horizontal_and_vertical", change_rate=1.0)

    elif fn_name == "random_translation":
        return segm_random_affine.segm_random_translation(images, labels, width_factor=0.2, height_factor=0.2)

    elif fn_name == "random_rotation":
        return segm_random_affine.segm_random_rotation(images, labels, factor=0.075)

    elif fn_name == "random_shear":
        return segm_random_affine.segm_random_shear(images, labels, factor=0.075, axis='xy')

    elif fn_name == "random_shear_x":
        return segm_random_affine.segm_random_shear(images, labels, factor=0.075, axis='x')

    elif fn_name == "random_shear_y":
        return segm_random_affine.segm_random_shear(images, labels, factor=0.075, axis='y')

    elif fn_name == "random_zoom":
        return segm_random_affine.segm_random_zoom(images, labels, width_factor=0.3)
        

def demo_data_augmentation(images_path, masks_path, sets_path, grayscale=None, num_images=None):
    """
    Samples a batch of images with their groundtruth labels from 
    the training set, applies to them the data augmentation functions
    specified in the YAML configuration file, and displays side by side 
    the original images and augmented images with their groundtruth
    bounding boxes.
    """
    
    function_names = [
        "random_contrast", "random_brightness", "random_gamma", "random_hue",
        "random_saturation", "random_value", "random_hsv", "random_rgb_to_hsv",
        "random_rgb_to_grayscale", "random_sharpness", "random_posterize", 
        "random_invert", "random_solarize", "random_autocontrast", "random_blur",
        "random_gaussian_noise", "random_crop", "random_flip", "random_translation",
        "random_rotation", "random_shear", "random_shear_x", "random_shear_y",
        "random_zoom"
    ]

    color_only_functions = [
        "random_hue", "random_saturation", "random_value", "random_hsv",
        "random_rgb_to_hsv", "random_rgb_to_grayscale", "random_autocontrast"
    ]
    
    # If grayscale was requested, remove the functions
    # that are only applicable to color images from
    # the list of function names.
    if grayscale:
        for fn in color_only_functions:
            function_names.remove(fn)

    # Create a configuration dictionary with the
    # information needed to create the data loader
    scale = 1./255
    offset = 0
    cfg = DefaultMunch.fromDict({
                "preprocessing": {
                    "rescaling": { "scale": scale, "offset": offset  },
                    "resizing": { "interpolation": "bilinear", "aspect_ratio": "fit" },
                    "color_mode": "grayscale" if grayscale else "rgb",
                }
          })

    # Create the data loader
    data_loader= get_ds(images_path, masks_path, sets_path,
                        cfg=cfg,
                        image_size=(224, 224),
                        batch_size=num_images,
                        seed=None,
                        shuffle=True,
                        to_cache=False)
    
    print("Demonstrating data augmentation functions:")
    for fn in function_names:
        print("  " + fn)
        
    for i, data in enumerate(data_loader):

        images, labels = data
        batch_size = tf.shape(images)[0]

        images_aug, labels_aug = augment_images(images, labels, fn_name=function_names[i])
        
        # Plot the images and labels
        for k in range(batch_size):
            plot_images_and_labels(images[k], labels[k], images_aug[k], labels_aug[k], 
                                   grayscale=grayscale, legend=function_names[i])

        # Stop when all the data augmentation functions have been demo'ed
        if i == len(function_names) - 1:
            exit()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default='', required=True,
                        help='images directory path')
    parser.add_argument('--masks', type=str, default='', required=True,
                        help='masks directory path')
    parser.add_argument('--sets', type=str, default='', required=True,
                        help='sets file path')
    parser.add_argument('--grayscale', action='store_true',
                        help='demo data augmentation functions on grayscale images')
    parser.add_argument('--num_images', type=int, default=4,
                        help='number of images to display for each data augmentation function (default: 4)')
    
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        raise ValueError(f"\nCould not find images directory: {args.images}")
        
    if not os.path.isdir(args.masks):
        raise ValueError(f"\nCould not find masks directory: {args.masks}")
        
    if not os.path.isfile(args.sets):
        raise ValueError(f"\nCould not find sets file: {args.sets}")
    
    demo_data_augmentation(args.images, args.masks, args.sets,
                           grayscale=args.grayscale, num_images=args.num_images)

if __name__ == '__main__':
    main()

