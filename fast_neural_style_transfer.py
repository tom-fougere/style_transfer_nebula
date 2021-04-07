from utils import *
import tensorflow_hub as hub

STYLE_IMAGE_DIR = 'images_style'

# Load raw images
content_image = load_image('terre.jpg')
# style_image = load_image('images_style/aigle.jpg')
style_images = load_images('images_style/')

# Show images
show_images(style_images, titles='Content')

# Load TF-Hub module.
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Pre-process images (tensor and reshape)
content_tensor = preprocess_image([content_image], image_size=(512, 512))
style_tensor = preprocess_image(style_images, image_size=(512, 512))

# Stylization
outputs = hub_module(content_tensor, style_tensor)
tensor_output = outputs[0]

# Pre-process images (tensor and reshape)
stylized_image = postprocess_tensor(tensor_output, image_size=(content_image.shape[0], content_image.shape[1]))

# Show images
showed_images = [content_image]
# showed_images.extend(style_images)
showed_images.append(stylized_image)
show_images(showed_images, titles=('Content', 'Style', 'Stylized image'))
