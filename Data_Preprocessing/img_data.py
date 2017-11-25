### This script extracts and classifies images based on the metadata provided and
### exports them to separate folders in order to train & test on the GANs

import numpy as np
import pandas as pd
from PIL import Image
import os
import string
import random

import custom_loadmat # load Matlab struct into Python 



# Choose parameters
choose_dataset = 'Male_Faces'   # Male_Faces or Female_Faces
faceyness = 5                 #Choose real value between 1 and 7 to be more or less lenient when selecting faces
crop_size = 0.05                 #Lenience when cropping - default is maximum crop minus 5%
train_test_ratio = 0.9
#TODO: We may also wish to introduce a minimum image size e.g 64x64... 

# Import Matlab Struct
current_directory = os.getcwd()
folderpath = os.path.join(current_directory, 'Data_Preprocessing/wiki') 
filepath = os.path.join(folderpath,'wiki.mat')

mat = custom_loadmat.loadmat(filepath)


# Convert to Pandas dataframe
fields = mat['wiki']
columns = fields.keys() 
df = pd.DataFrame.from_dict(fields)

max = df['face_score'].to_frame().max(axis=1)
min =df['face_score'].to_frame().min(axis=1)
mean = df['face_score'].to_frame().mean(axis=1)


# Filter into subsets by faceyness, sex, and other desired categories
male_faces = df.loc[   (df['face_score'] > faceyness)   &   (df['gender'] == 1)   &   (np.isnan(df['second_face_score']))  ]
female_faces = df.loc[ (df['face_score'] > faceyness)   &   (df['gender'] == 0)   &   (np.isnan(df['second_face_score']))  ]
#TODO: additional categories for age bins



# Crop and Save images to folder for analysis
if choose_dataset == 'Male_Faces':
    dataset = male_faces
elif choose_dataset == 'Female_Faces':
    dataset = female_faces

# Define stochastic boolean
def decision(probability): 
    return random.random() < probability

i=0

for tpl in dataset.itertuples():
    i+=1
    try:
        img = Image.open(os.path.join(folderpath,tpl[4]))
        
        face_borders = tpl[2].tolist() # (Left,top,right,bottom)
        
        #newTL = oldTL 
        crop_borders = (face_borders[0]-img.width  * crop_size,
                        face_borders[1]-img.height * crop_size, 
                        face_borders[2]+img.width  * crop_size,
                        face_borders[3]+img.height * crop_size)

        img2 = img.crop(crop_borders)
        
        # Assign % of data to training and validation sets
        if decision(train_test_ratio) == True:
            output_folder = os.path.join(current_directory, str(choose_dataset),"train")
        else:
            output_folder = os.path.join(current_directory, str(choose_dataset),"test")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        filename = img.filename.rsplit('/',1)[1][:-4]
        fullpath = os.path.join(output_folder, filename + '.jpg')
        
        img2.save(fullpath)
        print('Saved ' + str(i) + ' of ' + str(dataset.shape[0]) + ' images')

    except:
        pass
print('Dataset Complete')