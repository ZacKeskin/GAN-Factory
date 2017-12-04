### This script extracts and classifies images based on the metadata provided and
### exports them to separate folders in order to train & test on the GANs

import numpy as np
import pandas as pd
from PIL import Image
import os
import string
import random
from datetime import timedelta, datetime

import custom_loadmat # load Matlab struct into Python 



# Choose parameters
choose_dataset = 'Young_Females'   # Male_Faces/Female_Faces/Old_Males/Young_Males/Old_Females/Young_Females
faceyness = 4              #Choose real value between 1 and 7 to be more or less lenient when selecting faces
crop_size = 0.05                 #Lenience when cropping - default is maximum crop minus 5%
train_test_ratio = 0.99


# Import Matlab Struct
current_directory = os.getcwd()
folderpath = os.path.join(current_directory,'Data_Preprocessing','wiki') 
filepath = os.path.join(folderpath,'wiki.mat')

mat = custom_loadmat.loadmat(filepath)


# Convert to Pandas dataframe
fields = mat['wiki']
columns = fields.keys() 
df = pd.DataFrame.from_dict(fields)
# Calculate ages from data and categorise into 'bins'
df['Age'] = df['photo_taken'] - (df['dob']/365) 
df['bin'] = pd.cut(df['Age'], [0,10,20,30,40,50,60,70,80,90,100,110,120])
df['young_old'] = pd.cut(df['Age'], [0,15,35,65,120], labels=['Children', 'Young', 'Out of Scope','Old'])

max = df['face_score'].to_frame().max(axis=1)
min = df['face_score'].to_frame().min(axis=1)
mean = df['face_score'].to_frame().mean(axis=1)


# Filter into subsets by faceyness, sex, and other desired categories
male_faces = df.loc[   (df['face_score'] > faceyness)   &   (df['gender'] == 1)   &   (np.isnan(df['second_face_score']))  ]
old_male_faces = df.loc[   (df['face_score'] > faceyness)   &   (df['gender'] == 1)   &   (np.isnan(df['second_face_score'])) & (df['young_old'] == 'Old') ]
young_male_faces = df.loc[   (df['face_score'] > faceyness)   &   (df['gender'] == 1)   &   (np.isnan(df['second_face_score'])) & (df['young_old'] == 'Young') ]

female_faces = df.loc[ (df['face_score'] > faceyness)   &   (df['gender'] == 0)   &   (np.isnan(df['second_face_score']))  ]
old_female_faces = df.loc[   (df['face_score'] > faceyness)   &   (df['gender'] == 0)   &   (np.isnan(df['second_face_score'])) & (df['young_old'] == 'Old') ]
young_female_faces = df.loc[   (df['face_score'] > faceyness)   &   (df['gender'] == 0)   &   (np.isnan(df['second_face_score'])) & (df['young_old'] == 'Young') ]


# Crop and Save images to folder for analysis
if choose_dataset == 'Male_Faces':
    dataset = male_faces
elif choose_dataset == 'Female_Faces':
    dataset = female_faces
elif choose_dataset == 'Old_Males':
    dataset = old_male_faces
elif choose_dataset == 'Old_Females':
    dataset = old_female_faces
elif choose_dataset == 'Young_Males':
    dataset = young_male_faces
elif choose_dataset == 'Young_Females':
    dataset = young_female_faces

print(dataset)

# Define stochastic boolean
def decision(probability): 
    return random.random() < probability

i=0

for tpl in dataset.itertuples():
    
    try:
        img = Image.open(os.path.join(folderpath,tpl[4]))

        if os.stat(os.path.join(folderpath,tpl[4])).st_size  > (10000): # Ensure minimum filesize to prevent corrupted files
            face_borders = tpl[2].tolist() # (Left,top,right,bottom)
            i+=1
            #newTL = oldTL 
            crop_borders = (face_borders[0]-img.width  * crop_size,
                            face_borders[1]-img.height * crop_size, 
                            face_borders[2]+img.width  * crop_size,
                            face_borders[3]+img.height * crop_size
                            )

            img2 = img.crop(crop_borders)
            
            # Assign % of data to training and validation sets
            if decision(train_test_ratio) == True:
                output_folder = os.path.join(current_directory,'data', str(choose_dataset),"train")
            else:
                output_folder = os.path.join(current_directory, 'data', str(choose_dataset),"test")
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            filename = img.filename.rsplit('/',1)[1][:-4]
            fullpath = os.path.join(output_folder, filename + '.jpg')
            
            img2.save(fullpath)
            print('Saved ' + str(i) + ' of ' + str(dataset.shape[0]) + ' images')
        else:
            print(os.path.getsize(os.path.join(folderpath,tpl[4])))
    except:
        pass
print('Dataset Complete')