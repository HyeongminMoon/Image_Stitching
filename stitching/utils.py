import numpy as np
from glob import glob
import os
import cv2

def split_list(input_lst, split_count, reverse=False):
    lst = []
    nd_arrays = np.array_split(input_lst, split_count)
    for arr in nd_arrays:
        lst.append(list(arr))
    
    if reverse:
        lst.reverse()
    
    return lst


def test_list(root='Sample[0-9]'):

    path = glob(root)
    folder_dict = {}
    row_count = 4
    for folder in path:
        folder_name = os.path.basename(folder)
        images = glob(folder+'/*.png')
        image_pos_lst = []
        image_neg_lst = []
        for image in images:
            if '[-]' in image or 'Sample4' in image or 'Sample3' in image:
                image_neg_lst.append(image)
                image_neg_lst.sort()
            elif '[+]' in image:
                image_pos_lst.append(image)
                image_pos_lst.sort()
                image_pos_lst.reverse()
        folder_dict[folder_name+'[+]'] = split_list(image_pos_lst, row_count)
        folder_dict[folder_name+'[-]'] = split_list(image_neg_lst, row_count, True)

    return folder_dict

# img1을 img2의 밝기에 맞춤
# pivot은 (x,y)
def hist_match(img1, img2, img1_pivot=None, img2_pivot=None, first=None):

    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img1_v = img1_hsv[:,:,2]

    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    img2_v = img2_hsv[:,:,2]

    source = img1_v
    template = img2_v
    
    if img1_pivot==None or img2_pivot==None:
        s_max = np.max(source[:50,:])
        if first:
            t_max = np.max(template[:50,600:])
        else:
            t_max = np.max(template[:50,:])

        v_factor = t_max / s_max
    else:
        s_x, s_y = img1_pivot
        t_x, t_y = img2_pivot

        s_mean = np.mean(source[s_y-25:s_y+25, s_x-25:s_x+25])
        t_mean = np.mean(template[t_y-25:t_y+25, t_x-25:t_x+25])
        
        v_factor = t_mean / s_mean

    img1_hsv[:,:,2] = img1_hsv[:,:,2]*v_factor
    img1_hsv = img1_hsv.astype(np.uint8)

    return cv2.cvtColor(img1_hsv, cv2.COLOR_HSV2BGR)
    
        