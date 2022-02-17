import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image

def center_crop(img):
    h, w = img.height, img.width

    if h > w:
        half = (h - w) // 2
        cropsize = (0, half, w, half + w)  # left, upper, right, lower
    elif w > h:
        half = (w - h) // 2
        cropsize = (half, 0, half + h, h)

    if h != w:
        img = img.crop(cropsize)

    img = img.resize((128, 128), Image.ANTIALIAS)

    return img

path_list = ['./train_vid/', './val_vid/']
save_path_list = ['./train', './val' ]

for i in range(2):
    path = path_list[i]
    save_path = save_path_list[i]

    dir_list = os.listdir(path)
    mp4s = [d for d in dir_list if '.mp4' in d]

    if not os.path.exists(f'{save_path}'):
        os.mkdir(f'{save_path}')

    for mp4 in tqdm(mp4s):
        if not os.path.exists(f'{save_path}/{mp4[:-4]}'):
            os.mkdir(f'{save_path}/{mp4[:-4]}')

        vidcap = cv2.VideoCapture(f'{path}/{mp4}')
        success, image = vidcap.read()
        count = 0
        while success:
            pil_image = Image.fromarray(image)
            image = np.array(center_crop(pil_image))
            cv2.imwrite(f"{save_path}/{mp4[:-4]}/frame{str(count).zfill(5)}.png", image)
            success, image = vidcap.read()
            count +=1
