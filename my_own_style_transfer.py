import tensorflow as tf
from models import *
from utils import load_image, show_images, preprocess_image


# ######################################################################################################################
# Load raw images
content_image = plt.imread('terre.jpg')
style_image = plt.imread('images_style/lyre.jpg')

# Show images
show_images([content_image, style_image], titles=('Content', 'Style'))

# Pre-process images (tensor and reshape)
content_tensor = preprocess_image([content_image], image_size=(512, 512))
style_tensor = preprocess_image([style_image], image_size=(512, 512))
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
# content_layers = ['conv2d_88']
content_layers = ['block5_conv2']

# choose the five style layers of interest
# style_layers = ['conv2d',
#                 'conv2d_1',
#                 'conv2d_2',
#                 'conv2d_3',
#                 'conv2d_4']
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

create_model('vgg19', content_and_style_layers)
# model = vgg_model(output_layers)
# my_model = inception_model(content_and_style_layers)

# ######################################################################################################################
# define style and content weight
style_weight = 1
content_weight = 1e-32

# define optimizer. learning rate decreases per epoch.
adam = tf.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=80.0, decay_steps=100, decay_rate=0.80
    )
)

# start the neural style transfer
stylized_image, display_images = fit_style_transfer(style_image=style_tensor, content_image=content_tensor,
                                                    style_weight=style_weight, content_weight=content_weight,
                                                    optimizer=adam, epochs=10, steps_per_epoch=100)
