import numpy as np 
import tensorflow as tf 

from model.input_data import InputImageProc
from model.cnn import CNN

if __name__ == '__main__':
    input_proc = InputImageProc()
    input_proc()

    train_img = np.load('./datasets/train/dataset.npy')
    train_label = np.load('./datasets/train/label_list.npy')
    
    create_model = CNN()
    cnn = create_model.cnn()
    cnn.fit(train_img, train_label, epochs=5)
    cnn.save('./model/ckpt/')

    valid_img = np.load('./datasets/valid/dataset.npy')
    valid_label = np.load('./datasets/valid/label_list.npy')
    valid_loss, valid_acc = cnn.evaluate(valid_img, valid_label, verbose=2)