import tensorflow as tf
import numpy as np
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


def preprocess_image(image, image_size=(256, 256)):
    image = tf.image.resize(image, image_size)
    tensor = tf.expand_dims(image, axis=0)
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
