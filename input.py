import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    i=0
    im_gray = cv2.imread('./images/red_rain/rain1.jpg', cv2.IMREAD_GRAYSCALE)
    #print(im_gray.shape)
    cv2.imshow('gray', im_gray)
    cv2.waitKey(0)