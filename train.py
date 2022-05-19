import numpy as np 
import tensorflow as tf 

from model.input_data import InputImageProc
from model.cnn import CNN

if __name__ == '__main__':
    train_img = np.load('./datasets/train/dataset.npy')
    train_label = np.load('./datasets/train/label_list.npy')
    input_proc = InputImageProc(dataset=train_img, label_list=train_label)
    #train_img, train_label = input_proc.shuffle_dataset()
    """
    valid = np.load('./datasets/valid/dataset.npy')
    valid_label = np.load('./datasets/valid/label_list.npy')
    valid_tfds = InputTfds(valid, valid_label)
    valid_tfds = valid_tfds.create_tfds()
    
    test = np.load('./datasets/test/dataset.npy')
    test_label = np.load('./datasets/test/label_list.npy')
    test_tfds = InputTfds(test, test_label)
    test_tfds = test_tfds.create_tfds()
    """
    create_model = CNN()
    cnn = create_model.cnn()
    cnn.fit(train_img, train_label, epochs=5)

    cnn.save('./model/ckpt/')