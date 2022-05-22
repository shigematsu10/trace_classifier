import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import random

class InputImageProc:

    def __init__(self, dataset=None, label_list=None):
        #./images/red_draw/aug/draw{i}.jpegの数
        self.draw_image_num = 0
        #./images/red_rain/aug/rain{i}.jpegの数
        self.rain_image_num = 0

        self.dataset = dataset
        self.label_list = label_list

    def count_file(self):
        draw_dir = './images/red_draw/aug'
        rain_dir = './images/red_rain/aug'
        for file_draw in os.listdir(draw_dir):
            draw_path = os.path.join(draw_dir, file_draw)
            if os.path.isfile(draw_path):
                self.draw_image_num += 1
        self.draw_image_num -= 1
        for file_rain in os.listdir(rain_dir):
            rain_path = os.path.join(rain_dir, file_rain)
            if os.path.isfile(rain_path):
                self.rain_image_num += 1
        self.rain_image_num -= 1
        self.all_image_num = self.draw_image_num + self.rain_image_num
        print(f'rain_image num : {self.rain_image_num}')
        print(f'draw_image num : {self.draw_image_num}')
        print(f'all_image num : {self.all_image_num}')


    def create_dataset(self):
        label_name_list = ['draw', 'rain']
        
        self.label_list = []
        labels = [0] * self.draw_image_num
        self.label_list.extend(labels)
        labels = [1] * self.rain_image_num
        self.label_list.extend(labels)
        self.label_list = np.array(self.label_list)
        #print(self.label_list.shape)
        np.save('./datasets/label_list_origin.npy', self.label_list)

        self.dataset = np.empty((0, 150, 100), dtype='uint8')
        for name in label_name_list :
            if name == 'draw':
                img_num = self.draw_image_num
            elif name == 'rain':
                img_num = self.rain_image_num
            for i in range(img_num) :
                image_gray = cv2.imread(f'./datasets/gray/{name}{i}.jpeg',cv2.IMREAD_GRAYSCALE)
                image_gray = image_gray[np.newaxis, :, :]
                self.dataset = np.append(self.dataset, image_gray, axis=0)
                #print(dataset.shape)
        #print(self.dataset.shape)
        np.save('./datasets/dataset_origin.npy', self.dataset)


    def shuffle_dataset(self):
        dataset_list = self.dataset.tolist()
        dataset_label_list = self.label_list.tolist()
        p = list(zip(dataset_list, dataset_label_list))
        random.shuffle(p)
        dataset_list, dataset_label_list = zip(*p)
        self.dataset = np.array(dataset_list)
        self.label_list = np.array(dataset_label_list)


    def split_train_valid_test(self):
        train_num = int(self.all_image_num * 0.8)
        valid_test_num = int(self.all_image_num * 0.1)
        print(f'train image num : {train_num}')
        print(f'valid_test image num : {valid_test_num}*2')
        train, valid_test = np.split(self.dataset, [train_num], 0)
        train_label, valid_test_label = np.split(self.label_list, [train_num], axis=0)
        valid, test = np.split(valid_test, [valid_test_num], 0)
        valid_label, test_label = np.split(valid_test_label, [valid_test_num], axis=0)
        np.save('./datasets/train/dataset.npy', train)
        np.save('./datasets/train/label_list.npy', train_label)
        np.save('./datasets/valid/dataset.npy', valid)
        np.save('./datasets/valid/label_list.npy', valid_label)
        np.save('./datasets/test/dataset.npy', test)
        np.save('./datasets/test/label_list.npy', test_label)


    def __call__(self):
        self.count_file()
        self.create_dataset()
        self.shuffle_dataset()
        self.split_train_valid_test()
