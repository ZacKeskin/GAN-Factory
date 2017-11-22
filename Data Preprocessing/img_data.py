import numpy as np
import pandas as pd
import custom_loadmat
from PIL import Image


# Import Matlab Struct
folderpath = '/Users/Zac/Education/UCL/MSc/E2. Introduction to Deep Learning/Group Project/UCL_Deep_Learning_Group/Data Preprocessing/wiki/'
filepath = folderpath + 'wiki.mat'
mat = custom_loadmat.loadmat(filepath)

# Convert to Pandas dataframe
fields = mat['wiki']
columns = fields.keys() 
df = pd.DataFrame.from_dict(fields)

max = df['face_score'].to_frame().max(axis=1)
min =df['face_score'].to_frame().min(axis=1)
mean = df['face_score'].to_frame().mean(axis=1)

#print(max) #,min,mean)
faceyness = 6.5

male_faces = df.loc[   (df['face_score'] > faceyness)   &   (df['gender'] == 1)   &   (np.isnan(df['second_face_score']))  ]
female_faces = df.loc[ (df['face_score'] > faceyness)   &   (df['gender'] == 0)   &   (np.isnan(df['second_face_score']))  ]

print(df.shape)

print(male_faces.shape)
print(female_faces.shape)

#print(female_faces.head)
#print(male_faces.tail())
for tpl in male_faces.itertuples():
    print(tpl[4])
    img = Image.open(folderpath+tpl[4])
    img.show()
    