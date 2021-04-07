from utils import *
import tensorflow_hub as hub

STYLE_IMAGE_DIR = 'images_style'

# Load raw images
content_image = load_image('terre.jpg')
style_image = load_image('images_style/lyre.jpg')

# Show images
# show_images([content_image, style_image], titles=('Content', 'Style'))

# Load TF-Hub module.
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Pre-process images (tensor and reshape)
content_tensor = preprocess_image(content_image, image_size=(512, 512))
style_tensor = preprocess_image(style_image, image_size=(512, 512))

# Stylization
outputs = hub_module(content_tensor, style_tensor)
tensor_output = outputs[0]

# Pre-process images (tensor and reshape)
stylized_image = postprocess_tensor(tensor_output, image_size=(content_image.shape[0], content_image.shape[1]))

# Show images
show_images([content_image, style_image, stylized_image], titles=('Content', 'Style', 'Stylized image'))
