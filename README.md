## UCL Deep Learning Group Project
Working Repository for Deep Learning Group Project

Team:
- Gideon Acquah-Harrison
- Sibi Chandar
- Zac Keskin
- Lena Petersen


# Project 

- Comparing the performance of WGANs, DiscoGANs, CycleGANs etc. in style transfer in human faces

- Style transfer to be investigated along lines of gender, age and emotion

- Training data to be a subset from the WIKI dataset https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

- *We also need to schedule >2 (but assuming the more, the better) meetings with TAs/Nic during the project*


# Project Plan, Backlog & Tracker

I have created an actions tracker to help alignment towards a high-level plan. I hope we can track to this and keep eachother updated, to understand where we are in real time. This comprises a GoogleSheet document, which can be found at https://docs.google.com/spreadsheets/d/1iw5dNL4kMnFnwyHyoqgyGfzPhCI-4PX7P7ThLXq8II4/edit?usp=sharing



# Marking Criteria

To update this section with information on how we are graded regarding individual paper, group paper, presentation etc.



# Pre-Requisites

The simplest way to align a common codebase is to install anaconda:
https://www.anaconda.com/download/#macos

Most packages required are included as part of anaconda. Additionally we require raw TensorFlow

You should be able to simply run:
``` 
pip install TensorFlow
```

(you may need sudo depending on your local permissions)





# Code Set Up

I suggest we use Jupyter notebooks to build and share the code within out work. This is because it is a common and popular format for sharing scientific work in Python, but more importantly it has modular code interpretation - i.e. you can run certain parts of the code independently, so slow steps such as training or evaluating a simple model can be done once and kept in memory whilst you then try other things (rather than having to do this each time you make a change).

This is simple if you  `cd` into the cloned repo on your local machine and then type: `jupyter notebook` , and paste the link produced into your browser.


