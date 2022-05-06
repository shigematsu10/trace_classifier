import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import random
#名前も自動で変更されたらbestだった、、、、

class InputImageProc:

    def __init__(self, dataset=None, label_list=None):
        #./images/red_draw/aug/draw{i}.jpegの数
        self.draw_image_num = 180
        #./images/red_rain/aug/rain{i}.jpegの数
        self.rain_image_num = 220

        self.all_image_num = self.draw_image_num + self.rain_image_num

        self.dataset = dataset
        self.label_list = label_list


    def change_gray(self):
        for i in range(self.draw_image_num):
            image_gray = cv2.imread(f'./images/red_draw/aug/draw{i}.jpeg', cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'./datasets/gray/draw{i}.jpeg', image_gray)
        
        for i in range(self.rain_image_num):
            image_gray = cv2.imread(f'./images/red_rain/aug/rain{i}.jpeg', cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'./datasets/gray/rain{i}.jpeg', image_gray)


#ここにtrain, valid, testに分けてnpyファイルを保存するコードをかく
    def create_dataset(self):
        label_name_list = ['draw', 'rain']
        dataset = np.empty((1, 150, 100), dtype='uint8')
        self.label_list = []
        labels = [0] * self.draw_image_num
        self.label_list.extend(labels)
        labels = [1] * self.rain_image_num
        self.label_list.extend(labels)
        self.label_list = np.array(label_list)
        print(self.label_list.shape)
        np.save('./datasets/label_list_origin.npy', self.label_list)
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
        self.dataset = np.delete(dataset, 0, 0)
        print(self.dataset.shape)
        np.save('./datasets/dataset_origin.npy', self.dataset)


    def split_train_valid_test(self):
        train_num = self.all_image_num * 0.8
        valid_test_num = self.all_image_num * 0.1
        train, valid_test = np.split(self.dataset, train_num, 0)
        valid, test = np.split(valid_test, valid_test_num, 0)
        return train, valid, test


    def shuffle_dataset(self, img, label):
        input_img = img.tolist()
        input_label = label.tolist()
        p = list(zip(input_img, input_label))
        random.shuffle(p)
        input_img, input_label = zip(*p)
        return input_img, input_label


    #def create_tfds(self, image):
    def __call__(self):
        self.change_gray()
        self.create_dataset()
        input_data, input_label = self.shuffle_dataset()

