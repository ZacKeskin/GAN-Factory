import os
import data
import Discogan, mega
from Discogan import generator, generator2, discriminator, discriminator2
import tensorflow as tf

class GAN():

    def __init__(self, GANtype, kernel_size=[5,5], activation='relu'):
        
        valid = ['WGAN','','Discogan']
        if GANtype.upper() not in (v.upper() for v in valid):
            raise ValueError("GANtype must be one of %r." % valid)
        else:
            self.GANtype = GANtype

        if not (isinstance(kernel_size,(list,tuple)) and len(kernel_size)==2):
            raise ValueError("Kernel_Size must be a list containing two elements") #should probably check for positive ints too
        else:
            self.kernel_size = kernel_size

    def train(self, 
                input_dir=os.path.join(os.getcwd(),'input'), 
                target_dir=os.path.join(os.getcwd(),'target'), 
                batch_size=128, 
                n_epochs=10,
                img_height=64,
                img_width=64,
                img_channels=3
                ):  #optimser,loss, checkpoint_after_epoch
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.input_data = data.Tensorfy_images(input_dir,img_height,img_width,img_channels,batch_size)
        self.target_data = data.Tensorfy_images(input_dir,img_height,img_width,img_channels,batch_size)
            
        
        if self.GANtype.upper() == "WGAN": 
            #train
            pass
        elif self.GANtype.upper() == "":
            #train
            pass
        elif self.GANtype.upper() == "DISCOGAN":
            print('Training DiscoGAN')
            Discogan.train(self.batch_size, self.n_epochs,self.input_data,self.target_data)

    def test(self):
        pass

class Generator():
    def __init__(self,input, is_train, kernel_size, channel,reuse=False):
        self.c2, self.c4, self.c8, self.c16 = 32, 64, 128, 256  # channel num: 64, 128, 256, 512

        self.output_dim = channel
        self.kernel_size = kernel_size
        self.channel=channel
        self.reuse = reuse
        self.build()

    def build(self):
    
        with tf.variable_scope('gen') as scope:
            if self.reuse:
                scope.reuse_variables()
    
            # Convolution, activation, bias, repeat! 
            conv1 = tf.layers.conv2d(input, self.c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    name='conv1')
            # Regularisation layer in every convolution.
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')

            act1 = lrelu(bn1, n='act1')
            act1 = tf.nn.dropout(act1, keep_prob=0.5)
            #TODO: Can we explain why act1 is being reassigned immediately? Same for all activation layers below...

            #Convolution, activation, bias, repeat! 
            conv2 = tf.layers.conv2d(act1, self.c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    name='conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
            act2 = lrelu(bn2, n='act2')
            act2 = tf.nn.dropout(act2, keep_prob=0.5)
            #Convolution, activation, bias, repeat! 
            conv3 = tf.layers.conv2d(act2, self.c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    name='conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
            act3 = lrelu(bn3, n='act3')
            act3 = tf.nn.dropout(act3, keep_prob=0.5)
            #Convolution, activation, bias, repeat! 
            conv4 = tf.layers.conv2d(act3, self.c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                    name='conv4')
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
            act4 = lrelu(bn4, n='act4')
            act4 = tf.nn.dropout(act4, keep_prob=0.5)
            #deconvolution, activation, bias, repeat! 
            conv5 = tf.layers.conv2d_transpose(act4, self.c8, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            name ='conv5')
            bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn5')
            act5 = tf.nn.relu(bn5, name='act5')
            #deconvolution, activation, bias, repeat! 
            conv6 = tf.layers.conv2d_transpose(act5, self.c4, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            name ='conv6')
            bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay =0.9, updates_collections=None, scope='bn6')
            act6 = tf.nn.relu(bn6, name='act6')
            #deconvolution, activation, bias, repeat! 
            conv7 = tf.layers.conv2d_transpose(act6, self.c2, kernel_size=[5,5], strides =[2,2], padding = "SAME", 
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


class Discriminator():
    def __init__(self,input, is_train, reuse=False):
        pass


# Examples for initialising GANs
mysimplegan = GAN("wgan")
mydiscogan = GAN(GANtype='DiscoGAN',kernel_size=[5,5])

print(mysimplegan.GANtype)


# Define placeholders for GAN
HEIGHT, WIDTH, CHANNEL = 64, 64, 3
BATCH_SIZE = 64
EPOCH = 100

is_train = tf.placeholder(tf.bool, name='is_train')
input_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL])
target_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL])
gen1 = Generator(input=input_image, is_train=is_train, kernel_size=[5,5], channel=3)

# Examples for Training Model
dir1 = os.path.join(os.getcwd(),'Male_Faces') 
dir2 = os.path.join(os.getcwd(),'Female_Faces')

#mydiscogan.train(input_dir = dir1, target_dir = dir2, batch_size = 256, n_epochs=20) #using default 64x64x3 image shape

