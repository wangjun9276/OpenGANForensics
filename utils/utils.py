import cv2
import numpy as np
from PIL import Image


def read_txt(path):
    txt_content = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            # data = f.readline()
            txt_content.append(line)
        # print(data)

    return txt_content

class RandomResize(object):

    def __init__(self, resize_range, width):
        self.resize_range = resize_range
        self.width = width

    def __call__(self, img):
        self.scale_f = np.random.choice(self.resize_range)
        # img_h, img_w = img.shape[0], img.shape[1]
        img = np.array(img)
        img = cv2.resize(img, dsize=None, fx=self.scale_f, fy=self.scale_f, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, dsize=(self.width, self.width), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(np.uint8(img))
        return img