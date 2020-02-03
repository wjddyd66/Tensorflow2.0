import os
import shutil


image_dir = './JPEGImages/'
ano_dir ='./Annotations/'
preprocessing_dir = '../preprocessing_test'
preprocessing_image = preprocessing_dir+'/JPEGImages/'
preprocessing_ano = preprocessing_dir+'/Annotations/'

def make_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    else:
        pass
    
make_dir(preprocessing_dir)
make_dir(preprocessing_image)
make_dir(preprocessing_ano)

image_file = list(map(lambda x: x[:-4], os.listdir(image_dir)))
ano_file = list(map(lambda x: x[:-4], os.listdir(ano_dir)))

for image in image_file:
    for ano in ano_file:
        if image == ano:
            shutil.move(image_dir+image+'.jpg',preprocessing_image)
            shutil.move(ano_dir+ano+'.xml',preprocessing_ano)
