import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import csv


def read_txt(path):
    txt_content = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            # data = f.readline()
            txt_content.append(line)
        # print(data)

    return txt_content


def read_csv(path):
    img_path = []
    labels = []
    masks = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if line[1] == 'path':
                continue
            # data = f.readline()
            img_path.append(line[1])
            labels.append(line[2])
            #masks.append(line[3])

    return img_path, labels, masks

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



def calc_auroc(id_test_results, ood_test_results):
    # calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    # print(scores.shape)
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)
    fpr, tpr, thresholds = roc_curve(trues, scores, pos_label=1, drop_intermediate=False)

    return result, fpr, tpr