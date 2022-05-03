from DataAugmentation.augmentation import ImageProcess

if __name__ == '__main__':
    path_name_list = ['rain']
    for path_name in path_name_list :
        img_proc = ImageProcess(path_name, resize_original_save=False)
        img_proc(aug_method='all')