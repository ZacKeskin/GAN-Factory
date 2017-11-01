## UCL Deep Learning Group Project
Working Repository for Deep Learning Group Project

Team:
- Gideon Acquah-Harrison
- Sibi Chandar
- Zac Keskin
- Lena Petersen


# Project Ideas

- Ideally, something interactive that will play well at the presentation.
- This is the link we discussed, as simple food for thought: [Deep Learning Ideas](https://elitedatascience.com/machine-learning-projects-for-beginners)
- Also [Kaggle for datasets and inspiration from their challenges](https://www.kaggle.com/datasets)
- Suggestion of using image or audio recognition. 
- Need to decide on GAN (/DiscoGAN), CNN, etc. - what are the benefits/challenges in each choice - I imagine it depends on the task we are trying to complete so need to agree functionality first.

 *Note we need to confirm with Nic whether our chosen project is sufficient and get his feedback.*
 *We also need to schedule >2 (but assuming the more, the better) meetings with TAs/Nic during the project*


# Project Plan, Backlog & Tracker

I will create an actions tracker to help alignment towards a high-level plan. I hope we can track to this and keep eachother updated, to understand where we are in real time. This will be a GoogleSheet document; link to follow soon.



# Marking Criteria

To update this section with information on how we are graded regarding individual paper, group paper, presentation etc.



# Pre-Requisites

The simplest way to align a common codebase is to install anaconda:
https://www.anaconda.com/download/#macos

Additionally you will need to install Keras. 

You should be able to simply run:
``` 
pip install TensorFlow
```
```
pip install Keras
```
(you may need sudo depending on your local permissions)

However, in case this is not sufficient then In the repo I have also included the instructions provided by Nic Lane. Realistically you will not need to follow these in detail, as Anaconda includes most of the dependencies. The final step includes copying and running a sample DL program, which imports MNIST data, trains a model on a large portion of it and then tests the remainder on the trained model.



# Code Set Up

I suggest we use Jupyter notebooks to build and share the code within out work. This is because it is a common and popular format for sharing scientific work in Python, but more importantly it has modular code interpretation - i.e. you can run certain parts of the code independently, so slow steps such as training or evaluating a simple model can be done once and kept in memory whilst you then try other things (rather than having to do this each time you make a change).

This is simple if you  `cd` into the cloned repo on your local machine and then type: `jupyter notebook` , and paste the link produced into your browser.

If you do that you will see some initial test code I've provided that i was looking into; the idea is to load the model trained on the MNIST data set (the HFD5 datafile provided in the /Models folder within the Repo and then see if it can correctly classify an image of a numeral (nine.jpg). I didn't get it to do that yet - but hopefully food for thought.

