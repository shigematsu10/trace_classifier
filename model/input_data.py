import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
#名前も自動で変更されたらbestだった、、、、

class InputImageProc:

    def __init__(self):
        self.draw_image_num = 180
        self.rain_image_num = 220
        self.all_image_num = self.draw_image_num + self.rain_image_num
    

    def change_gray(self):
        for i in range(self.draw_image_num):
            image_gray = cv2.imread(f'./images/red_draw/aug/draw{i}.jpeg', cv2.IMREAD_GRAYSCALE)
            #cv2.imshow("gray",image_gray)
            #cv2.waitKey(0)
            cv2.imwrite(f'./datasets/gray/draw{i}.jpeg', image_gray)
        
        for i in range(self.rain_image_num):
            image_gray = cv2.imread(f'./images/red_rain/aug/rain{i}.jpeg', cv2.IMREAD_GRAYSCALE)
            #cv2.imshow("gray",image_gray)
            #cv2.waitKey(0)
            cv2.imwrite(f'./datasets/gray/rain{i}.jpeg', image_gray)


    def create_tfds(self):
        label_name_list = ['draw', 'rain']
        dataset = np.empty((1, 150, 100), dtype='uint8')
        label_list = []
        labels = [0] * self.draw_image_num
        label_list.extend(labels)
        labels = [1] * self.rain_image_num
        label_list.extend(labels)
        label_list = np.array(label_list)
        print(label_list.shape)
        np.save('./datasets/label_list.npy', label_list)
        
        for name in label_name_list :
            if name == 'draw':
                img_num = self.draw_image_num
            elif name == 'rain':
                img_num = self.rain_image_num
            for i in range(img_num) :
                image_gray = cv2.imread(f'./datasets/gray/{name}{i}.jpeg',cv2.IMREAD_GRAYSCALE)
                image_gray = image_gray[np.newaxis, :, :]
                dataset = np.append(dataset, image_gray, axis=0)
                #print(dataset.shape)
        dataset = np.delete(dataset, 0, 0)
        print(dataset.shape)
        np.save('./datasets/dataset.npy', dataset)
    



    #def create_tfds(self, image):
    def __call__(self):
        self.change_gray()

