import tensorflow as tf
from models import *


# ######################################################################################################################
# Load raw images
content_image = plt.imread('terre2.jpg')
style_image = plt.imread('images_style/lyre.jpg')

# Pre-process images (tensor and reshape)
content_image = tf.image.resize(content_image, (512, 512))
content_image = tf.expand_dims(content_image, 0)
content_image = tf.cast(content_image, tf.uint8)
style_image = tf.image.resize(style_image, (512, 512))
style_image = tf.expand_dims(style_image, 0)
style_image = tf.cast(style_image, tf.uint8)

# ######################################################################################################################
# clear session to make layer naming consistent when re-running this cell
tf.keras.backend.clear_session()

# download the vgg19/Inception model and inspect the layers
# tmp_model = tf.keras.applications.vgg19.VGG19()
# tmp_model = tf.keras.applications.InceptionV3()
# tmp_model.summary()

# del tmp_model

# ######################################################################################################################
# choose the content layer and put in a list
content_layers = ['block5_conv2']

# choose the five style layers of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# combine the content and style layers into one list
content_and_style_layers = style_layers + content_layers

# ######################################################################################################################
# clear session to make layer naming consistent if re-running the cell
tf.keras.backend.clear_session()

create_model('vgg19', content_and_style_layers, num_style_layers=5)
# model = vgg_model(output_layers)
# my_model = inception_model(content_and_style_layers)

# ######################################################################################################################
# define style and content weight
style_weight = 1e-12
content_weight = 1
var_weight = 1

# define optimizer. learning rate decreases per epoch.
adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=30.0, decay_steps=100, decay_rate=0.80
    )
)

# start the neural style transfer
stylized_image, display_images = fit_style_transfer(style_image=style_image, content_image=content_image,
                                                    style_weight=style_weight, content_weight=content_weight,
                                                    var_weight=var_weight,
                                                    optimizer=adam, epochs=10, steps_per_epoch=50)
