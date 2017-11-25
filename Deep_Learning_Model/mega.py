
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
import pdb
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils import *
import imghdr

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 64, 64, 3
BATCH_SIZE = 64
EPOCH = 2000
os.environ['CUDA_VISIBLE_DEVICES'] = '15' #TODO: can we comment to explain what this is?
version = 'NewFaces'
output_path = './' + version

print('Warming up...')
# Function to calculate leaky ReLu 
def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
 
def target_data():  
    
    #get the current directory we are in. 
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, 'target')
    
    images = [] #create an empty array to store the images

    #for each item in the data folder...put that image into the images array    
    for each in os.listdir(target_dir):
        if imghdr.what(os.path.join(target_dir,each)) != None:
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
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    # Number of images in the dataset
    num_images = len(images)
    # Return the image batch and the number of images
    return iamges_batch, num_images

# Do the same for the images to be converted
def input_data():   
    current_dir = os.getcwd()
   
    input_dir = os.path.join(current_dir, 'input')
    images = []

    for each in os.listdir(input_dir):
        if imghdr.what(os.path.join(input_dir,each)) != None:
            images.append(os.path.join(input_dir,each))
    
    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer([all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_image(content, channels = CHANNEL)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
  
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])

    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    iamges_batch1 = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images1 = len(images)

    return iamges_batch1, num_images1

#takes in a 64*64*3 image and attempts to output a different 64*64*3 image.
def generator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 32, 64, 128, 256  # channel num: 64, 128, 256, 512
    output_dim = CHANNEL
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
  
        # Convolution, activation, bias, repeat! 
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        # Regularisation layer in every convolution.
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')

        act1 = lrelu(bn1, n='act1')
        act1 = tf.nn.dropout(act1, keep_prob=0.5)
        #TODO: Can we explain why act1 is being reassigned immediately? Same for all activation layers below...

         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        act2 = tf.nn.dropout(act2, keep_prob=0.5)
         #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
        act3 = tf.nn.dropout(act3, keep_prob=0.5)
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
        act4 = tf.nn.dropout(act4, keep_prob=0.5)
        #deconvolution, activation, bias, repeat! 
        conv5 = tf.layers.conv2d_transpose(act4, c8, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        #deconvolution, activation, bias, repeat! 
        conv6 = tf.layers.conv2d_transpose(act5, c4, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv6')
        bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn6')
        act6 = tf.nn.relu(bn6, name='act6')
        #deconvolution, activation, bias, repeat! 
        conv7 = tf.layers.conv2d_transpose(act6, c2, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv7')
        bn7 = tf.contrib.layers.batch_norm(conv7, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn7')
        act7 = tf.nn.relu(bn7, name='act7')
        #deconvolution, activation, bias, repeat! 
        conv8 = tf.layers.conv2d_transpose(act7, output_dim, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv8')
        bn8 = tf.contrib.layers.batch_norm(conv8, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn8')
        act8 = tf.nn.relu(bn8, name='act8')

        return act8 # Return generated image (eventually we want this generator to take in one image and output another that's what we're training it to do)

def generator2(input, is_train, reuse=False):
    c2, c4, c8, c16 = 32, 64, 128, 256  # channel num: 64, 128, 256, 512
    output_dim = CHANNEL
    with tf.variable_scope('gene') as scope:
        if reuse:
            scope.reuse_variables()
        
        #Convolution, activation, bias, repeat! 
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        #regularisation layer in every convolution.
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')

        act1 = lrelu(bn1, n='act1')
        act1 = tf.nn.dropout(act1, keep_prob=0.5)

         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        act2 = tf.nn.dropout(act2, keep_prob=0.5)
        #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
        act3 = tf.nn.dropout(act3, keep_prob=0.5)
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
        act4 = tf.nn.dropout(act4, keep_prob=0.5)
        #deconvolution, activation, bias, repeat! 
        conv5 = tf.layers.conv2d_transpose(act4, c8, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        #deconvolution, activation, bias, repeat! 
        conv6 = tf.layers.conv2d_transpose(act5, c4, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv6')
        bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn6')
        act6 = tf.nn.relu(bn6, name='act6')
        #deconvolution, activation, bias, repeat! 
        conv7 = tf.layers.conv2d_transpose(act6, c2, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv7')
        bn7 = tf.contrib.layers.batch_norm(conv7, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn7')
        act7 = tf.nn.relu(bn7, name='act7')
        #deconvolution, activation, bias, repeat! 
        conv8 = tf.layers.conv2d_transpose(act7, output_dim, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name ='conv8')
        bn8 = tf.contrib.layers.batch_norm(conv8, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn8')
        act8 = tf.nn.relu(bn8, name='act8')

        return act8

def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 32, 64, 128, 256  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()
         
        #Convolution, activation, bias, repeat! 
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(bn1, n='act1')

         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
       
        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        #convert the 4*4*256 tensor into a flat and fully connectecd layer
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
      
        #intialise random weight variables
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        #intialize random bias variables
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # multiply the weights by the fully connected layer add the biases (multiplying 4096 row vector by 4096 column vector giving 1 number )
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # perform sigmoid on the one number putting the value of the number between 0 and 1.
        acted_out = tf.nn.sigmoid(logits)
        return logits #return a value between 0 and 1


def train():
 
    print('CUDA Visible Devices: ' + str(os.environ['CUDA_VISIBLE_DEVICES'])) 
    
    with tf.variable_scope('input'):
        #real image placeholder
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        #fake image place holder
        people_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='people_image')
        is_train = tf.placeholder(tf.bool, name='is_train')
    
    #feed the fake image into the generator.
    fake_image = generator(people_image, is_train)
    reconstructed_image = generator2(fake_image, is_train)
    #feed the real image into the discriminator 
    real_result = discriminator(real_image, is_train)
    #feed the output of the generator into the disccriminator
    fake_result = discriminator(fake_image, is_train, reuse=True)

    #reconstructed loss
    #pdb.set_trace()
    reconstructed_loss=tf.metrics.mean_squared_error(people_image, reconstructed_image)
    #calculate the loss between the generated image and real image
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result) + tf.reduce_mean(reconstructed_loss) # This optimizes the generator. # This optimizes the generator.
       
    #returns a list of trainable variables (likes weights and biases)
    t_vars = tf.trainable_variables()
    #trainable discriminator variables are stored in d_vars
    d_vars = [var for var in t_vars if 'dis' in var.name]
    #trainable generator variables are stored in g_vars
    g_vars = [var for var in t_vars if 'gen' in var.name]
   
    #minmise the d_loss using the rms optimizer by altering the discriminator weights and biases      
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-3).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights between the values -0.01 and 0.01
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    
    batch_size = BATCH_SIZE
    image_batch, samples_num = target_data()
    image_batch2, samples_num2 = input_data()
    

    batch_num = int(samples_num / batch_size)
    total_batch = 0
    #start the tf session
    #pdb.set_trace()
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    #saver = tf.train.import_meta_graph('4500.meta')
    #saver.restore(sess, tf.train.latest_checkpoint('./'))
    print(tf.train.latest_checkpoint('./'))
    #initialise the variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('total training sample num:%d' % samples_num)
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')

   
    for i in range(EPOCH):
        print('\n Epoch: ' + str(i+1) + ' of ' + EPOCH) #i+1 due to python's zero-indexing
        for j in range(batch_num):
            print('\n Batch: ' + str(j) + ' of ' + str(batch_num)) #i+1 due to python's zero-indexing
            d_iters = 2
            g_iters = 1

            
            for k in range(d_iters):
                print(k)
                print(image_batch.shape)
                train_image = sess.run(image_batch)
                train_image2 = sess.run(image_batch2)
                
                sess.run(d_clip)
            # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={people_image: train_image2, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                
                _, gLoss = sess.run([trainer_g, g_loss],
                                   feed_dict={people_image: train_image2, is_train: True})   

        # save check point every 100 epochs
        if i%100 == 0:
            if not os.path.exists('./checkpoints/' + '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())):
                os.makedirs('./checkpoints/' + '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
            saver.save(sess, './checkpoints/' +'{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))  
        if i%5 == 0:
            # save images
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            imgtest = sess.run(fake_image, feed_dict={people_image: train_image2, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
           
            save_images(imgtest, [8,8] ,output_path + '/epoch' + str(i) + '.jpg')
            
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
    coord.request_stop()
    coord.join(threads)

#random
#def test():
    # random_dim = 100
    # with tf.variable_scope('input'):
        # real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        # random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        # is_train = tf.placeholder(tf.bool, name='is_train')
    
    # # wgan
    # fake_image = generator(random_input, random_dim, is_train)
    # real_result = discriminator(real_image, is_train)
    # fake_result = discriminator(fake_image, is_train, reuse=True)
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    # print(variables_to_restore)
    # saver = tf.train.Saver(variables_to_restore)
    # ckpt = tf.train.latest_checkpoint('./model/' + version)
    # saver.restore(sess, ckpt)


if __name__ == "__main__":
    train()
   # test()


