from PIL import Image
import torch.utils.data as data
import torch
import numpy as np
from utils.utils import read_txt
import cv2
import csv


class Custom_train_loader(data.Dataset):
    def __init__(self, data_path, label_path, img_size=64,transform=None, train=True, binary=False):
        self.data_path = data_path
        self.lable_list = read_txt(label_path)
        self.transform = transform
        self.train = train
        self.binary = binary
        self.img_size = img_size

        edit_label = 0
        self.valid_list = []
        self.valid_listlab = []
        for lable in self.lable_list:
            imgs = []
            labels = []
            for r in range(2):
                if r == 1:
                    new_prex = label_path[18:].replace('train', 'valid')
                    new_list = read_txt(data_path + lable + new_prex)
                else:
                    new_list = read_txt(data_path + lable + label_path[18:])
                ###
                rand_perm = np.random.permutation(np.arange(len(new_list)))
                new_list = [new_list[x1] for x1 in rand_perm]
                ###
                nm_per_sample = len(new_list)
                imgs += new_list
                labels += list(np.ones(nm_per_sample) * edit_label)
            #print(len(imgs))
            if edit_label == 0:
                #self.img_list = read_txt(data_path + lable + label_path[18:])
                self.img_list = imgs[:4000]
                self.nm_per_sample = len(self.img_list)
                self.sub_label = list(np.ones(self.nm_per_sample) * edit_label)
                ###
                self.valid_list += imgs[4000:]
                nm_per_sample = len(self.valid_list)
                self.valid_listlab += list(np.ones(nm_per_sample) * edit_label)
            else:
                #new_list = read_txt(data_path + lable + label_path[18:])
                new_list = imgs[:4000]
                self.nm_per_sample = len(new_list)
                self.img_list = self.img_list + new_list
                self.sub_label = self.sub_label + list(np.ones(self.nm_per_sample) * edit_label)
                ###
                self.valid_list += imgs[4000:]
                nm_per_sample = len(imgs[4000:])
                self.valid_listlab += list(np.ones(nm_per_sample) * edit_label)
                # print(len(self.img_list), len(self.sub_label))
            edit_label += 1
        #print(len(self.img_list), len(self.sub_label), len(self.valid_list), len(self.valid_listlab))
        if self.train:
            csv_file = 'valid_data2.csv'
            with open(csv_file, 'w') as csvfile:
                csvfile.write("Image_path,GANs\n")
                for indes in range(len(self.valid_list)):
                    csvfile.write("{},{}\n".format(self.valid_list[indes], self.valid_listlab[indes]))
                    csvfile.flush()
        else:
            self.img_list = []
            self.sub_label = []
            csvFile = open("valid_data2.csv", "r")
            reader = csv.reader(csvFile)
            for item in reader:
                if reader.line_num == 1:
                    continue
                self.img_list.append(item[0])
                self.sub_label.append(float(item[1]))
                #print(item[0], item[1])
        #print(len(self.img_list), len(self.sub_label))

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        img = np.array(img)

        mask = cv2.imread(self.img_list[index].replace('image', 'mask')[:-4] + '_mask2.png', 0)
        mask = np.expand_dims(mask, axis=2)

        if self.transform is not None:
            if self.train:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = np.array(transformed['mask'][:, :, 0])
            else:
                img = self.transform(img)
                mask = mask[:, :, 0]

        mask = cv2.resize(mask, [self.img_size, self.img_size]) / 255
        if self.binary:
            mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)[1]
        mask = np.expand_dims(mask, axis=2)
        mask = torch.tensor(mask).permute(2, 0, 1).float()

        label = self.sub_label[index]
        label = torch.tensor(label)

        img = torch.tensor(np.float32(img) / 255.0).permute(2, 0, 1)

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
        self.pa = []
        for lable in self.lable_list:
            if lable == 'none':
                pa = 1
            else:
                pa = 0
            if edit_label == 0:
                self.img_list = read_txt(data_path + lable + label_path[19:])
                self.nm_per_sample = len(self.img_list)
                self.sub_label = list(np.ones(self.nm_per_sample) * edit_label)
                self.pa += list(np.ones(self.nm_per_sample) * pa)
            else:
                new_list = read_txt(data_path + lable + label_path[19:])
                self.nm_per_sample = len(new_list)
                self.img_list = self.img_list + new_list
                self.sub_label = self.sub_label + list(np.ones(self.nm_per_sample) * edit_label)
                self.pa += list(np.ones(self.nm_per_sample) * pa)
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
        
        img = np.array(img)
        img = torch.tensor(np.float32(img) / 255.0).permute(2, 0, 1)

        return [img, mask], label, self.pa[index], self.img_list[index]

    def __len__(self):
        return len(self.img_list)

