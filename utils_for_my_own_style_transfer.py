import tensorflow as tf


def preprocess_image(image, model=None):
    """
    Preprocesses a given image to use with Inception model

    :param image:
    :return:
    """
    image = tf.cast(image, dtype=tf.float32)

    if model == 'inception':
        image = (image / 127.5) - 1.0
    elif model == 'vgg19':
        image = tf.keras.applications.vgg19.preprocess_input(image)
    else:
        image = image / 255.

    return image
