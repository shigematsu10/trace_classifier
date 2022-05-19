import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image

def show_image(img):
    plt.figure(figsize=(5, 5))
    plt.xticks(color='None')
    plt.yticks(color='None')
    plt.tick_params(bottom=False, left =False)
    plt.imshow(img)
    plt.show()
    plt.close()

if __name__ == '__main__' :
    print('学習データのサンプル表示')
    train_img1 = image.load_img('./images/red_draw/aug/draw10.jpeg', target_size=(150, 100))
    train_img2 = image.load_img('./images/red_rain/aug/rain10.jpeg', target_size=(150, 100))
    show_image(train_img1)
    show_image(train_img2)

    trained_cnn = keras.models.load_model('./model/ckpt/')
    #test用データセット作成の必要あり
    train_img = np.load('./datasets/test/dataset.npy')
    train_label = np.load('./datasets/test/label_list.npy')

