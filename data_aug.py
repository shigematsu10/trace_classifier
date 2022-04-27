from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


def show_image(img):
    plt.figure(figsize=(5, 5))
    plt.xticks(color='None')
    plt.yticks(color='None')
    plt.tick_params(bottom=False, left =False)
    plt.imshow(img)
    plt.show()
    plt.close()

class Image_process:

    def __init__(self, path_name):
        self.path_name = path_name
        self.path_origin = f'./images/red_{path_name}/'
        self.path_aug = f'./images/red_{path_name}/aug/'


    def resize_and_gather(self):
        #path_name : draw or rain
        if self.path_name == 'draw' :
            image_num = 45
        elif self.path_name == 'rain' :
            image_num = 51

        dataset = np.empty((1, 150, 100, 3))
        for id in range(1, image_num) :
            image = image.load_img(self.path_origin + f'{self.path_name}{id}.jpeg', target_size=(150, 100))
            image = np.array(image)
            image = image[np.newaxis, :, :, :]
            dataset = np.append(dataset, image, axis=0)
        dataset = np.delete(dataset, 0, 0)
        return dataset

    def image_save(self):
        dataset = self.resize_and_gather()
        image_num = dataset.shape[0]
        for id in range(image_num):
            image = array_to_img(dataset[id][0], scale = False)
            image.save_img(self.path_aug + f'{self.path_name}{id}.jpeg')

    def rotation(img):
        datagen = image.ImageDataGenerator(rotation_range=40)
        img_rotation = datagen.flow(dataset, batch_size=1)


#<<<--------------------------------基本操作----------------------------------------------->>>
#読み込む時点でresizeも行う
img1 = image.load_img('./images/red_draw/draw0.jpeg', target_size=(150, 100))
img2 = image.load_img('./images/red_draw/draw1.jpeg', target_size=(150, 100))
#画像からndarrayに変換
img1 = np.array(img1)
img2 = np.array(img2)
#(縦，横，チャンネル数)の3次元データ
#print(img1.shape)


#Keras.ImageDataGeneratorは4次元データに対応＝複数枚の入力に対応(画像数，縦，横，チャンネル数)
img1 = img1[np.newaxis, :, :, :]
img2 = img2[np.newaxis, :, :, :]
#(画像数，縦，横，チャンネル数)の4次元データ
#print(img1.shape)
dataset = np.append(img1, img2, axis=0)
print(dataset.shape)

#画像変換方法定義
datagen = image.ImageDataGenerator(rotation_range=180)
#画像生成
dataset_aug = datagen.flow(dataset, batch_size=1)
#ndarrayから画像へ変換（引数の一つ目注意：縦，横，チャンネル数の3次元）
data1 = array_to_img(dataset_aug[0][0], scale = False)
data2 = array_to_img(dataset_aug[1][0], scale = False)
show_image(data1)
show_image(data2)

#画像保存
image.save_img('./sample.jpeg', data1)
image.save_img('./sample2.jpeg', data2)
#<<<---------------------------------------------------------------------------------------->>>

