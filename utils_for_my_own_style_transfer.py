import tensorflow as tf
import matplotlib.pyplot as plt


def preprocess_image(image, model=None):
    """
    Preprocesses a given image to use with Inception model

    :param image: image to preprocess, numpy array
    :param model: name of the wanted preprocessing, string
    :return:
        - image: the preprocessed image
    """
    image = tf.cast(image, dtype=tf.float32)

    if model == 'inception':
        image = (image / 127.5) - 1.0
    elif model == 'vgg19':
        image = tf.keras.applications.vgg19.preprocess_input(image)
    else:
        image = image / 255.

    return image


def display(image_tensor):
    """
    Display an image
    :param image_tensor: tensor containing an image
    """
    display_image = tensor_to_image(image_tensor)
    plt.figure()
    plt.imshow(display_image)
    plt.show()


def tensor_to_image(tensor):
    """
    Converts a tensor to an image
    :param tensor: tensor with a shape (batch, height, width, layer)
    :return: image: first image from the tensor
    """

    image = tensor[0].numpy()
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype('uint8')
    return image
