import sys
from PIL import Image
import numpy as np
import tensorflow as tf
from nst_utils import *

raw_content_image=sys.argv[1]
raw_style_image=sys.argv[2]
n_epochs=int(sys.argv[3])


# convert and reshape
content_image=image_to_array(raw_content_image)
style_image=image_to_array(raw_style_image)

# create a generated image from content image
generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
generated_image = tf.Variable(generated_image)


# load the VGG
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
# encoding
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style enconder



# Assign the content image to be the input of the VGG model.  
# Set a_C to be the hidden layer activation from the layer we have selected
preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))

a_C = vgg_model_outputs(preprocessed_content)
a_S = vgg_model_outputs(preprocessed_style)

print("Starting the Optimization")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def train_step(generated_image):
    with tf.GradientTape() as tape:
        # In this function you must use the precomputed encoded images a_S and a_C
        # Compute a_G as the vgg_model_outputs for the current generated image
        
        a_G = vgg_model_outputs(generated_image)
        
        # Compute the style cost
        J_style = compute_style_cost(a_S, a_G)

        
        # Compute the content cost
        J_content = compute_content_cost(a_C, a_G)
        # Compute the total cost
        J = total_cost(J_content, J_style, alpha=10, beta=40)
        
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))

    return J

def main():
    create_art_image(n_epochs, generated_image, train_step)
    

if __name__=="__main__":
    main()    