from PIL import Image
import os

num_images_to_sort = 1000
attribute = 21 

current_directory = os.getcwd()
folderpath = os.path.join(current_directory, 'data')
output_folder_1 = os.path.join(current_directory, str('Male'))
output_folder_2 = os.path.join(current_directory, str('Female'))


ins = open( "textFile.txt", "r" )
data = [[n for n in line.split()] for line in ins]
for i in range(0, num_images_to_sort):
    img = Image.open('data/' + str(data[i][0]))
    if(data[i][attribute]=="-1"):
        img.save(output_folder_2 + '/' + str(data[i][0]))
    else:
        img.save(output_folder_1 + '/' + str(data[i][0]))


