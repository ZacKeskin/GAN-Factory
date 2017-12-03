import tensorflow as tf
import os
import numpy as np


def Tensorfy_images(dir,HEIGHT,WIDTH,CHANNEL, BATCH_SIZE):   
   
    #get the current directory passed to the function. 
    target_dir = os.path.join(os.getcwd(),dir)
    
    images = [] #create an empty array to store the images

    #for each item in the data folder...put that image into the images array    
    for each in os.listdir(target_dir):
        images.append(os.path.join(target_dir,each))

    # Convert all the images in the images array to tensors (strings)
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    # Create a list of tensors one for each tensor (image)
    images_queue = tf.train.slice_input_producer([all_images])
    # Reads all the images in the queue and outputs them into 'content'                                   
    content = tf.read_file(images_queue[0])
    # Takes a jpeg encoded image which is of the type tf.string and decodes it to a proper image. Furthermore add the colour to the image (coloured images have 3 channels Red, Green, Blue)
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # Image altering features   TODO: can you expand on these, why they are necessary? etc.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

    # Resize the image
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    # Set the shape of the image to a 64*64*3 image
    image.set_shape([HEIGHT,WIDTH,CHANNEL])

    # Returns a tensor with the desired shape to image
    image = tf.cast(image, tf.float32)

    image = image / 255.0
    # List of tensors in desired batch size
    images_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    # Number of images in the dataset
    num_images = len(images)
    # Return the image batch and the number of images

    return images_batch, num_images