import cv2
import numpy as np
import matplotlib as plt
import os

def image_wash(filename):
    img = plt.imread(filename)


path = "./"
file_list = os.listdir(path)
file_list_image = [file for file in file_list if file.endswith(".jpg")]
file_list_png = [file for file in file_list if file.endswith(".png")]
file_list_image += file_list_png
print(file_list_image)
print(file_list_png)