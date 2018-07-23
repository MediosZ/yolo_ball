import os
from PIL import Image
import numpy as np

def input_data(filename):
    with open(filename, "r") as f:
        str = f.read()
        file_list = str.split('\n')
        #print(len(file_list))
    images_path = [item.split(' ')[0] for item in file_list]
    paras = [item.split(' ')[1] for item in file_list]
    paras = [list(map(int, item.split(','))) for item in paras]
    #print(paras[0])
    images = [np.array(Image.open(one)) for one in images_path]
    #print(images[0])
    return images, paras

#input_data('train.txt')