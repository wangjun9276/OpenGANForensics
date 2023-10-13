from PIL import Image
import torch.utils.data as data
import torch
import numpy as np
from utils.utils import read_txt, read_GAN_csv, read_SCL_csv
import cv2


class Custom_train_loader(data.Dataset):
    def __init__(self, data_path, label_path, img_size=64, transform=None, train='_train.csv', binary=False, mask=False, loc=False):
        self.data_path = data_path
        self.lable_list = read_txt(label_path)
        self.transform = transform
        self.train = train
        self.binary = binary
        self.img_size = img_size
        self.masks = mask
        self.loc = loc
        #print(self.masks)

        edit_label = 0
        self.valid_list = []
        self.valid_listlab = []
        for lable in self.lable_list:
            #print(len(imgs))
            if edit_label == 0:
                if self.loc:
                    self.img_list, _, self.mask_list = read_SCL_csv(data_path + lable + self.train)
                else:
                    self.img_list, _, self.mask_list = read_GAN_csv(data_path + lable + self.train)
                self.nm_per_sample = len(self.img_list)
                self.sub_label = list(np.ones(self.nm_per_sample) * edit_label)
            else:
                new_list, _, new_mask = read_csv(data_path + lable + self.train)
                self.nm_per_sample = len(new_list)
                self.img_list = self.img_list + new_list
                self.mask_list = self.mask_list + new_mask
                self.sub_label = self.sub_label + list(np.ones(self.nm_per_sample) * edit_label)
            edit_label += 1

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        img = np.array(img)
        
        if self.masks:
            #print(self.img_list[index])
            mask = cv2.imread(self.mask_list[index], 0)
            mask = np.expand_dims(mask, axis=2)
            
            mask = cv2.resize(mask, [self.img_size, self.img_size]) / 255
            if self.binary:
                mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)[1]
            mask = np.expand_dims(mask, axis=2)
            mask = torch.tensor(mask).permute(2, 0, 1).float()
        else:
            mask = torch.zeros([1, self.img_size, self.img_size]).float()
            
        if self.transform is not None:
            img = self.transform(img)

        
        label = self.sub_label[index]
        label = torch.tensor(label)

        # img = torch.tensor(np.float32(img) / 255.0).permute(2, 0, 1)

        return [img, mask], label

    def __len__(self):
        return len(self.img_list)


class Custom_test_loader(data.Dataset):
    def __init__(self, data_path, label_path, img_size=64, binary=False, transform=None):
        self.data_path = data_path
        self.lable_list = read_txt(label_path)
        self.transform = transform
        self.img_size = img_size
        self.binary = binary

        edit_label = 0
        for lable in self.lable_list:
            if edit_label == 0:
                self.img_list, _ = read_csv(data_path + lable + label_path[19:])
                self.nm_per_sample = len(self.img_list)
                self.sub_label = list(np.ones(self.nm_per_sample) * edit_label)
            else:
                new_list, _ = read_csv(data_path + lable + label_path[19:])
                self.nm_per_sample = len(new_list)
                self.img_list = self.img_list + new_list
                self.sub_label = self.sub_label + list(np.ones(self.nm_per_sample) * edit_label)
                # print(len(self.img_list), len(self.sub_label))
            edit_label += 1

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        mask = cv2.imread(self.img_list[index].replace('image', 'mask')[:-4] + '_mask2.png', 0)
        mask = cv2.resize(mask, [self.img_size, self.img_size]) / 255
        if self.binary:
            mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)[1]
        
        mask = np.expand_dims(mask, axis=2)
        mask = torch.tensor(mask).permute(2, 0, 1).float()
        img = np.array(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.sub_label[index]
        label = torch.tensor(label)
        
        # img = np.array(img)
        # img = torch.tensor(np.float32(img) / 255.0).permute(2, 0, 1)

        return [img, mask], label

    def __len__(self):
        return len(self.img_list)

