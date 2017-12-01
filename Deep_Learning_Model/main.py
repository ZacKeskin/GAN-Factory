import os
import Discogan, mega
from Discogan import generator, generator2, discriminator, discriminator2

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
    
        
   
    def train(self, input_dir=os.path.join(os.getcwd(),'input'), batch_size=128, n_epochs=10):  #target_dir,optimser,loss, checkpoint_after_epoch
        if self.GANtype.upper() == "WGAN":
            self.input_data = mega.input_data(input_dir)
            #train
        elif self.GANtype.upper == "":
            #get data
            #train
            pass
        elif self.GANtype.upper == "DISCOGAN":
            self.input_data = Discogan.input_data(input_dir)
            #train

    def test(self):
        pass



# Examples for initialising GANs
simplegan = GAN("wgan")
discogan = GAN(GANtype='DiscoGAN',kernel_size=[5,5])
print(simplegan.GANtype)




# Examples for Training Model
dir1 = os.path.join(os.getcwd(),'Male_Faces')

discogan.train(input_dir = dir1)
