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

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

if __name__ == '__main__' :
    print('学習データのサンプル表示')
    train_img1 = image.load_img('./images/red_draw/aug/draw10.jpeg', target_size=(150, 100))
    train_img2 = image.load_img('./images/red_rain/aug/rain10.jpeg', target_size=(150, 100))
    show_image(train_img1)
    show_image(train_img2)

    trained_cnn = keras.models.load_model('./model/ckpt/')
    #test用データセット作成の必要あり
    test_img = np.load('./datasets/test/dataset.npy')
    test_label = np.load('./datasets/test/label_list.npy')
    #評価コード
    predictions = trained_cnn.predict(test_img)
    # print(predictions)
    # print(predictions.shape)
    class_names = ['draw', 'rain']

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_label, test_img)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_label)
    plt.show()
    plt.close()




