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

HEIGHT, WIDTH, CHANNEL = 64, 64, 3
BATCH_SIZE = 64
EPOCH = 800

input_dir = os.path.join(os.getcwd(),'data','Male_Faces','train') 
target_dir = os.path.join(os.getcwd(),'data', 'Female_Faces','train')
output_dir = os.path.join(os.getcwd(),'GAN_Model_Training')

# Standard GAN
mystandardgan = GAN(GANtype="Standard",kernel_size=[5,5])
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
                checkpoint_after_epoch=50)
"""


# Discovery GAN
mydiscogan = GAN(GANtype='DiscoGAN',kernel_size=[5,5])

mydiscogan.train(input_dir=input_dir, 
                target_dir=target_dir, 
                batch_size=BATCH_SIZE, 
                n_epochs=EPOCH,
                img_height=HEIGHT,
                img_width=WIDTH,
                img_channels=CHANNEL,
                output_dir=output_dir,
                checkpoint_after_epoch=50)

# TBC: Testing

#mystandardgan.test(live_data_dir)
#myRLgan.test(live_data_dir)
#mydiscogan.test(live_data_dir)