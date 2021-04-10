import tensorflow as tf
import matplotlib.pyplot as plt

from losses import *
from utils_for_my_own_style_transfer import preprocess_image, display

# Global variables
selected_model_name = ''
selected_model = None
nb_style_layers = 0


def create_model(model_name, layer_names, num_style_layers=1):
    global selected_model_name, selected_model, nb_style_layers

    selected_model_name = model_name
    if model_name == 'inception':
        selected_model = inception_model(layer_names)
    elif model_name == 'vgg19':
        selected_model = vgg_model(layer_names)
    else:
        print('You need to define a model !')

    nb_style_layers = num_style_layers


def vgg_model(layer_names):
    """ Creates a vgg model that outputs the style and content layer activations.

    :param layer_names: a list of strings, representing the names of the desired content and style layers
    :return
        - model: A model that takes the regular vgg19 input and outputs just the content and style layers
    """

    # load the the pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

    # freeze the weights of the model's layers (make them not trainable)
    vgg.trainable = False

    # create a list of layer objects that are specified by layer_names
    outputs = [vgg.get_layer(name).output for name in layer_names]

    # create the model that outputs content and style layers only
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    return model


def inception_model(layer_names):
    """ Creates a inception model that returns a list of intermediate output values.

    :param layer_names: a list of strings, representing the names of the desired content and style layers

    :return model: A model that takes the regular inception v3 input and outputs just the content and style layers

    """

    # Load InceptionV3 with the imagenet weights and **without** the fully-connected layer at the top of the network
    inception = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)

    # Freeze the weights of the model's layers (make them not trainable)
    inception.trainable = False

    # Create a list of layer objects that are specified by layer_names
    output_layers = [inception.get_layer(name).output for name in layer_names]

    # Create the model that outputs the content and style layers
    model = tf.keras.Model(inputs=inception.inputs, outputs=output_layers)

    # return the model
    return model


def get_style_image_features(image):
    """Get the style image features

    :param image: an input image
    :return gram_style_features: the style features as gram matrices
    """
    global selected_model, nb_style_layers

    # preprocess the image using the given preprocessing function
    preprocessed_style_image = preprocess_image(image, selected_model_name)

    # get the outputs from the inception model that you created using inception_model()
    outputs = selected_model(preprocessed_style_image)

    # Get just the style feature layers (exclude the content layer)
    style_outputs = outputs[:nb_style_layers]

    # for each style layer, calculate the gram matrix for that layer and store these results in a list
    gram_style_features = [gram_matrix(style_output) for style_output in style_outputs]

    return gram_style_features


def get_content_image_features(image):
    """ Get the content image features

    :param image: an input image
    :return content_outputs: the content features of the image
    """
    global selected_model

    # preprocess the image
    preprocessed_content_image = preprocess_image(image, selected_model_name)

    # get the outputs from the inception model
    outputs = selected_model(preprocessed_content_image)

    # get the content layer of the outputs
    content_outputs = outputs[nb_style_layers:]

    return content_outputs


def calculate_gradients(image, style_targets, content_targets,
                        style_weight, content_weight):
    """ Calculate the gradients of the loss with respect to the generated image

    :param image: generated image
    :param style_targets: style features of the style image
    :param content_targets: content features of the content image
    :param style_weight: weight given to the style loss
    :param content_weight: weight given to the content loss

    :return gradients: gradients of the loss with respect to the input image
    """

    with tf.GradientTape() as tape:
        # get the style image features
        style_features = get_style_image_features(image)

        # get the content image features
        content_features = get_content_image_features(image)

        # get the style and content loss
        loss = get_style_content_loss(style_targets, style_features,
                                      content_targets, content_features,
                                      style_weight, content_weight)

    # calculate gradients of loss with respect to the image
    gradients = tape.gradient(loss, image)

    return gradients


def update_image_with_style(image, style_targets, content_targets, style_weight,
                            content_weight, optimizer):
    """
    Update an image with a specified style

    :param image: generated image
    :param style_targets: style features of the style image
    :param content_targets: content features of the content image
    :param style_weight: weight given to the style loss
    :param content_weight: weight given to the content loss
    :param optimizer: optimizer for updating the input image
    """

    # Calculate gradients using the function that you just defined.
    gradients = calculate_gradients(image,
                                    style_targets, content_targets,
                                    style_weight, content_weight)

    # apply the gradients to the given image
    optimizer.apply_gradients([(gradients, image)])

    # Clip the image using the given clip_image_values() function
    # image.assign(clip_image_values(image, min_value=0.0, max_value=255.0))


def fit_style_transfer(style_image, content_image, style_weight=1e-2, content_weight=1e-4,
                       optimizer='adam', epochs=1, steps_per_epoch=1):
    """
    Performs neural style transfer.
    :param style_image: image to get style features from
    :param content_image: image to stylize
    :param style_weight: weight given to the style loss
    :param content_weight: weight given to the content loss
    :param optimizer: optimizer for updating the input image
    :param epochs: number of epochs
    :param steps_per_epoch = steps per epoch

    :return generated_image: generated image at final epoch
    :return images: collection of generated images per epoch
    """

    updated_images = []
    step = 0

    # get the style image features
    style_targets = get_style_image_features(style_image)

    # get the content image features
    content_targets = get_content_image_features(content_image)

    # initialize the generated image for updates
    generated_image = tf.cast(content_image, dtype=tf.float32)
    generated_image = tf.Variable(generated_image)

    # collect the image updates starting from the content image
    updated_images.append(content_image)

    for i_epoch in range(epochs):
        for i_step_per_epoch in range(steps_per_epoch):
            step += 1

            # Update the image with the style using the function that you defined
            update_image_with_style(generated_image,
                                    style_targets, content_targets,
                                    style_weight, content_weight,
                                    optimizer)

            print(".", end='')
            if (i_step_per_epoch + 1) % 10 == 0:
                updated_images.append(generated_image)
                display(generated_image)

        # display the current stylized image
        display(generated_image)

        # append to the image collection for visualization later
        updated_images.append(generated_image)
        print("Train step: {}".format(step))

    # convert to uint8 (expected dtype for images with pixels in the range [0,255])
    generated_image = tf.cast(generated_image, dtype=tf.uint8)

    return generated_image, updated_images
