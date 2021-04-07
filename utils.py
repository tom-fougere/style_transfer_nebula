import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def load_image(image_path):
    """
    Load image (float32 conversion and normalization)

    :param image_path: path of the image to load, string
    :return:
        - image: Image, ndarray
    """
    image = plt.imread(image_path)
    image = image.astype(np.float32)
    image = image / 255.

    return image


def load_images(image_folder_path):
    """
    Load all images in the defined folder (float32 conversion and normalization)

    :param image_folder_path: path of the folder containing images, string
    :return:
        - images: Images of the folder, list of ndarray
    """

    image_list = []
    for img in os.listdir(image_folder_path):
        image = plt.imread(image_folder_path + img)
        image = image.astype(np.float32)
        image = image / 255.
        image_list.append(image)

    return image_list


def preprocess_image(images, image_size=(256, 256)):
    images_list = []
    for image in images:
        image = tf.image.resize(image, image_size)
        images_list.append(image)
    images = np.asarray(images_list)
    tensor = tf.convert_to_tensor(images)
    return tensor


def show_images(images, titles=('',)):
    nb_images = len(images)
    for i in range(nb_images):
        plt.subplot(1, nb_images, i+1)
        plt.imshow(images[i], aspect='equal')
        plt.axis('off')
        plt.title(titles[i] if len(titles) > i else '')
    plt.show()


def postprocess_tensor(tensor, image_size=(256, 256)):
    image = tensor[0]
    image = tf.image.resize(image, image_size)
    return image
