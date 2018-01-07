## UCL Deep Learning Group Project
Working Repository for Deep Learning Group Project

Team:
- Gideon Acquaah-Harrison (ID: 17107197)
- Sibi Chandar (ID: 15041329)
- Zac Keskin (ID: 16137321)
- Lena Petersen (ID: 17080280)


# Project 

- Comparing the performance of plain GANs, GANs with Reconstruction Loss and DiscoGANs in style transfer in human faces

- Style transfer to be investigated along lines of gender, age and emotion

- Training data to be a subset from the CelebA dataset http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# Collecting Data

- In order to train the model, we use data from CelebA

- We provide a simple tool (readText.py) to extract datasets of specific feature domains. This can be found in the 'CelebA Data Sort' folder within this repository.

- The instructions.txt file, in the same folder, provides detail instructions on how to use the tool

# Using the Code

- A sample implementation for each of the three GAN models is included in main.py

- To select a model, simply uncomment the relevant portions of code, or else use the GANFactory class to create your own

- A number of parameters are available when instantiating each of the GANs, including source and target directories, batch, epoch and kernel size and image dimensions. These are designed to enable quick customisation of model implementations

- With a GAN object, you may use the .train() method to begin learning the model. With saved weights (i.e. after training), you can test the trained implementation using .test(). This essentially runs through one more pass of the GAN with your selected input image, saving the output image in the desired directory.

- Further changes to the architecture are possible by adjusting the code in GANFactory.py


# Pre-Requisites

We assume the user has installed anaconda:
https://www.anaconda.com/download/#macos

Most packages required are included as part of anaconda. Additionally we require TensorFlow

You should be able to simply run:
``` 
pip install TensorFlow
```

(you may need sudo depending on your local permissions)



