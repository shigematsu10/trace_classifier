import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

class InputImageProc:

    def __init__(self, path_name):
        self.path_name = path_name
    
    def change_gray(self):
        