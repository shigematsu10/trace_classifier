from DataAugmentation.augmentation import ImageProcess

if __name__ == '__main__':
    path_name_list = ['draw', 'rain']
    for path_name in path_name_list :
        img_proc = ImageProcess(path_name, resize_original_save=True)
        img_proc(aug_method='all')