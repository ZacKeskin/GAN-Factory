import os
from GANfactory import GAN


""" Notes -
#
   1) Taxonomy
        We categorise the images into four types:
        
        i.   Real Input  (Data of the raw class we seek to change e.g. men)
        ii.  Real Target (Data of the class we change the input data into e.g. women)
        iii. Fake Input  (Generated data of the same category as the input i.e. reconstructed data from G1->D1->G2->)
        iv.  Fake Target (Generated data of the same category as the target i.e. the GAN-generated data)

        Hence:

        input_dir should contain the input data you would like the GAN to convert 
        target_dir should contain the target styles, to which you'd like the input data to be converted

    2) GAN Types
        There are three possible GANtypes accepted: 'Standard','RecLoss' or 'DiscoGAN'

    3) Usage
        Simply instantiate a GAN by assigning a new variable to GAN(GANtype)
        Use the GAN.train() method to begin training your model
"""


# Examples!

#Image height, width and channel (3 due to RGB)
HEIGHT, WIDTH, CHANNEL = 64, 64, 3
#Batch size of images
BATCH_SIZE = 16
EPOCH = 800
#test folder directory name
testfolder_dir = 'Test'
#image name for generated results.
resultname = 'result'
#input name for input image in test folder directory
input_name = 'input'
#target name for target image in test folder directory
target_name = 'target'
#input directory path (images are contained within the folder Male)
input_dir = os.path.join(os.getcwd(),'data','Male') 

#target directory path (images are contained within the folder female)
target_dir = os.path.join(os.getcwd(),'data', 'Female')

#sample images are created and stored within the folder GAN_Model_Training.
output_dir = os.path.join(os.getcwd(),'GAN_Model_Training')
##################################################################
#      Leave the GAN you would like to train uncommented         #
#                                                                #
#                                                                #
#                                                                #
#                                                                #
#                                                                #
##################################################################

#Standard GAN
#mystandardgan = GAN(GANtype="Standard",kernel_size=[5,5])
"""
mystandardgan.train(input_dir=input_dir, 
                target_dir=target_dir, 
                batch_size=BATCH_SIZE, 
                n_epochs=EPOCH,
                img_height=HEIGHT,
                img_width=WIDTH,
                img_channels=CHANNEL,
                output_dir=output_dir
                )
"""

"""
mystandardgan.test(height=HEIGHT,
                width=WIDTH,
                channel=CHANNEL,
                testfolder=testfolder_dir,
                resultname=resultname,
                input_image_name = input_name, 
                target_image_name = target_name
"""
"""
# GAN with Reconstruction Loss
myRLgan = GAN(GANtype="RecLoss",kernel_size=[5,5])

myRLgan.train(input_dir=input_dir, 
                target_dir=target_dir, 
                batch_size=BATCH_SIZE, 
                n_epochs=EPOCH,
                img_height=HEIGHT,
                img_width=WIDTH,
                img_channels=CHANNEL,
                output_dir=output_dir,
                checkpoint_after_epoch=1)
"""

"""
myRLgan.test(height=HEIGHT,
                width=WIDTH,
                channel=CHANNEL,
                testfolder=testfolder_dir,
                resultname=resultname,
                input_image_name = input_name, 
                target_image_name = target_name)
"""

# Discovery GAN

mydiscogan = GAN(GANtype='DiscoGAN',kernel_size=[4,4])

mydiscogan.train(input_dir=input_dir, 
                target_dir=target_dir, 
                batch_size=BATCH_SIZE, 
                n_epochs=EPOCH,
                img_height=HEIGHT,
                img_width=WIDTH,
                img_channels=CHANNEL,
                output_dir=output_dir,
                checkpoint_after_epoch=1)



mydiscogan.test(height=HEIGHT,
                width=WIDTH,
                channel=CHANNEL,
                testfolder=testfolder_dir,
                resultname=resultname,
                input_image_name = input_name, 
                target_image_name = target_name)
