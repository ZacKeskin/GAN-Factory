from PIL import Image
import os


current_directory = os.getcwd()
folderpath = os.path.join(current_directory, 'data')
output_folder_Male = os.path.join(current_directory, str('Male'))
output_folder_Female = os.path.join(current_directory, str('Female'))

ins = open( "textFile.txt", "r" )
data = [[n for n in line.split()] for line in ins]
for i in range(0, 10000):
    img = Image.open('data/' + str(data[i][0]))
    if(data[i][21]=="-1"):
        img.save(output_folder_Female + '/' + str(data[i][0]))
    else:
        img.save(output_folder_Male + '/' + str(data[i][0]))


