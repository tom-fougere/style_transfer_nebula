import tensorflow as tf


def get_style_loss(reality, expectation):
    """Expects two images of dimension h, w, c

    :param reality: tensor with shape: (height, width, channels)
    :param expectation: tensor with shape: (height, width, channels)

    :return style_loss: style loss, scalar
    """
    # get the average of the squared errors
    style_loss = tf.reduce_mean(tf.square(reality - expectation))

    return style_loss


def get_content_loss(reality, expectation):
    """Expects two images of dimension h, w, c

    :param reality: tensor with shape: (height, width, channels)
    :param expectation: tensor with shape: (height, width, channels)

    :return content_loss: content loss, scalar
    """
    # get the sum of the squared error multiplied by a scaling factor
    content_loss = 0.5 * tf.reduce_sum(tf.square(reality - expectation))

    return content_loss


def get_style_content_loss(style_targets, style_outputs, content_targets,
                           content_outputs, style_weight, content_weight):
    """ Combine the style and content loss

    :param style_targets: style features of the style image
    :param style_outputs: style features of the generated image
    :param content_targets: content features of the content image
    :param content_outputs: content features of the generated image
    :param style_weight: weight given to the style loss
    :param content_weight: weight given to the content loss

    :return total_loss: the combined style and content loss

    """

    # Sum of the style losses
    style_loss = tf.add_n([get_style_loss(style_output, style_target)
                           for style_output, style_target in zip(style_outputs, style_targets)])

    # Sum up the content losses
    content_loss = tf.add_n([get_content_loss(content_output, content_target)
                             for content_output, content_target in zip(content_outputs, content_targets)])

    # scale the style loss by multiplying by the style weight and dividing by the number of style layers
    style_loss = style_loss * style_weight / len(style_outputs)

    # scale the content loss by multiplying by the content weight and dividing by the number of content layers
    content_loss = content_loss * content_weight / len(content_outputs)

    # sum up the style and content losses
    total_loss = style_loss + content_loss

    # return the total loss
    return total_loss
