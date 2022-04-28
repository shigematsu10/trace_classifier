from DataAugmentation.augmentation import Image_process

if __name__ == '__main__':
    path_name_list = ['rain']
    for path_name in path_name_list :
        img_proc = Image_process(path_name, resize_original_save=False)
        img_proc(aug_method='all')