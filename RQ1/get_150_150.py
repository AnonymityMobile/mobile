import os
import cv2


def get_path(path_dir_compile):
    path_list = []
    if os.path.isdir(path_dir_compile):
        for root, dirs, files in os.walk(path_dir_compile, topdown=True):
            for file in files:
                file_absolute_path = os.path.join(root, file)
                if file_absolute_path.endswith('.JPEG'):
                    path_list.append(file_absolute_path)
    return path_list


if __name__ == '__main__':
    path_list = get_path('/Users/Desktop/insect')

    idx = 0
    for i in path_list:
        pic_name = i.split('/')[-1]
        pic = cv2.imread(i)
        pic_resize = cv2.resize(pic, (150, 150), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite('/Users/Desktop/insect_150_150/'+pic_name, pic_resize)
        if idx % 1000 == 0:
            print(idx)
        idx += 1