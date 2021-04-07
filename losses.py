import tensorflow as tf


def get_style_loss(features, targets):
    """Expects two images of dimension h, w, c

    Args:
      features: tensor with shape: (height, width, channels)
      targets: tensor with shape: (height, width, channels)

    Returns:
      style loss (scalar)
    """
    # get the average of the squared errors
    style_loss = tf.reduce_mean(tf.square(features - targets))

    return style_loss


def get_content_loss(features, targets):
    """Expects two images of dimension h, w, c

    Args:
      features: tensor with shape: (height, width, channels)
      targets: tensor with shape: (height, width, channels)

    Returns:
      content loss (scalar)
    """
    # get the sum of the squared error multiplied by a scaling factor
    content_loss = 0.5 * tf.reduce_sum(tf.square(features - targets))

    return content_loss


def gram_matrix(input_tensor):
    """ Calculates the gram matrix and divides by the number of locations
    Args:
      input_tensor: tensor of shape (batch, height, width, channels)

    Returns:
      scaled_gram: gram matrix divided by the number of locations
    """

    # calculate the gram matrix of the input tensor
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

    # get the height and width of the input tensor
    input_shape = tf.shape(input_tensor)
    height = input_shape[1]
    width = input_shape[2]

    # get the number of locations (height times width), and cast it as a tf.float32
    num_locations = tf.cast(height * width, tf.float32)

    # scale the gram matrix by dividing by the number of locations
    scaled_gram = gram / num_locations

    return scaled_gram