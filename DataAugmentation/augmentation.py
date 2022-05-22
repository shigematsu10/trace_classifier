from turtle import color, width
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


class ImageProcess:

    def __init__(self, path_name, resize_original_save=True):
        self.path_name = path_name
        self.path_origin = f'./images/red_{path_name}/'
        self.path_aug = f'./images/red_{path_name}/aug/'
        self.resize_original_save = resize_original_save

        self.origin_draw_image_num = 0
        self.origin_rain_image_num = 0
        self.aug_draw_image_num = 0
        self.aug_rain_image_num = 0


    def count_file(self):
        origin_draw_dir = './images/red_draw'
        origin_rain_dir = './images/red_rain'
        aug_draw_dir = './images/red_draw/aug'
        aug_rain_dir = './imgaes/red_rain/aug'
        for origin_draw_dir in os.listdir(origin_draw_dir):
            origin_draw_path = os.path.join(origin_draw_dir)
            if os.path.isfile(origin_draw_path):
                self.origin_draw_image_num += 1
        self.origin_draw_image_num -= 1

        for origin_rain_dir in os.listdir(origin_rain_dir):
            origin_rain_path = os.path.join(origin_rain_dir)
            if os.path.isfile(origin_rain_path):
                self.origin_rain_image_num += 1
        self.origin_rain_image_num -= 1

        for file_draw in os.listdir(aug_draw_dir):
            draw_path = os.path.join(aug_draw_dir, file_draw)
            if os.path.isfile(draw_path):
                self.aug_draw_image_num += 1
        self.aug_draw_image_num -= 1

        for file_rain in os.listdir(aug_rain_dir):
            rain_path = os.path.join(aug_rain_dir, file_rain)
            if os.path.isfile(rain_path):
                self.aug_rain_image_num += 1
        self.aug_rain_image_num -= 1
        print(self.origin_draw_image_num)
        print(self.origin_rain_image_num)
        print(self.aug_draw_image_num)
        print(self.aug_rain_image_num)


    def resize_and_gather(self):
        self.aug_name = 'resize'
        self.dataset = np.empty((1, 150, 100, 3))
        for id in range(0, self.image_num) :
            if self.path_name == 'draw' :
                original_image = image.load_img(self.path_origin + f'{self.path_name}{id}.jpeg', target_size=(150, 100))
            elif self.path_name == 'rain':
                original_image = image.load_img(self.path_origin + f'{self.path_name}{id}.jpg', target_size=(150, 100))
            original_image = np.array(original_image)
            original_image = original_image[np.newaxis, :, :, :]
            self.dataset = np.append(self.dataset, original_image, axis=0)
        self.dataset = np.delete(self.dataset, 0, 0)
        if self.resize_original_save == True :
            print('saveします')
            self.image_save(self.dataset)
        else :
            pass


    def image_save(self, images):
        for id in range(self.image_num):
            if self.aug_name == 'resize' :
                image_to_save = array_to_img(images[id], scale = False)
            else :
                image_to_save = array_to_img(images[id][0], scale = False)
            image.save_img(self.path_aug + f'{self.path_name}_{self.aug_name}{id}.jpeg', image_to_save)


    def rotation(self):
        self.aug_name = 'rotation'
        datagen = image.ImageDataGenerator(rotation_range=40)
        img_rotation_np = datagen.flow(self.dataset, batch_size=1)
        self.image_save(img_rotation_np)
    

    def shift(self):
        self.aug_name = 'shft'
        datagen = image.ImageDataGenerator(width_shift_range=0.5)
        img_shift_np = datagen.flow(self.dataset, batch_size=1)
        self.image_save(img_shift_np)
    

    def zoom(self):
        self.aug_name = 'zoom'
        datagen = image.ImageDataGenerator(zoom_range=[0.5, 1.5])
        img_zoom_np = datagen.flow(self.dataset, batch_size=1)
        self.image_save(img_zoom_np)


    def change_gray(self):
        for i in range(self.draw_image_num):
            image_gray = cv2.imread(f'./images/red_draw/aug/draw{i}.jpeg', cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'./datasets/gray/draw{i}.jpeg', image_gray)
        
        for i in range(self.rain_image_num):
            image_gray = cv2.imread(f'./images/red_rain/aug/rain{i}.jpeg', cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(f'./datasets/gray/rain{i}.jpeg', image_gray)


    def __call__(self, aug_method):
        if aug_method == 'all':
            self.count_file()
            self.resize_and_gather()
            self.rotation()
            self.shift()
            self.zoom()
        elif aug_method == 'rotation':
            self.count_file()
            self.resize_and_gather()
            self.rotation()
        elif aug_method == 'shift':
            self.count_file()
            self.resize_and_gather()
            self.shift()
        elif aug_method == 'zoom':
            self.count_file()
            self.resize_and_gather()
            self.zoom()
        self.change_gray()
        self.count_file()


"""
#<<<--------------------------------基本操作----------------------------------------------->>>
def show_image(img):
    plt.figure(figsize=(5, 5))
    plt.xticks(color='None')
    plt.yticks(color='None')
    plt.tick_params(bottom=False, left =False)
    plt.imshow(img)
    plt.show()
    plt.close()

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
"""
