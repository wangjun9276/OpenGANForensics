'''
ResNet + (Vit for classification), patching after CNN extraction, 61 ResNetPlusTrans2.py
'''

import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.cuda import is_available as is_available_cuda
from Networks.resnet import resnet50, resnet34, resnet18, resnet101, resnet152
from torch.nn import functional as F

import os
import torch.utils.data as data
import torch
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from Networks.vit import Transformer
import albumentations as A
import warnings
warnings.filterwarnings('ignore')

from utils.data_load import Custom_train_loader, Custom_test_loader
device = 'cuda' if is_available_cuda() else 'cpu'


class ResNetVit(nn.Module):

    def __init__(self, model, img_size=256, nodown=True, patch_height=32, patch_width=32, n_class=1, dim=256):
        super().__init__()
        self.nodown = nodown
        self.img_size = img_size
        self.dim = dim
        
        if model == 'resnet18':
            self.net = resnet18(pretrained=False)
            self.filters = [64, 64, 128, 256, 512]
        elif model == 'resnet34':
            self.net = resnet34(pretrained=False)
            self.filters = [64, 64, 128, 256, 512]
        elif model == 'resnet50':
            self.net = resnet50(pretrained=False)
            self.filters = [64, 256, 512, 1024, 2048]
        elif model == 'resnet101':
            self.net = resnet101(pretrained=False)
            self.filters = [64, 256, 512, 1024, 2048]
        elif model == 'resnet152':
            self.net = resnet152(pretrained=False)
            self.filters = [64, 256, 512, 1024, 2048]
        
        if self.nodown:
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=3, stride=1, padding=0, bias=False)
        else:
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=7, stride=2, padding=3, bias=False)
        
        self.firstbn = self.net.bn1
        self.firstrelu = self.net.relu
        self.firstmaxpool = self.net.maxpool
        self.encoder1 = self.net.layer1
        self.encoder2 = self.net.layer2
        self.encoder3 = self.net.layer3
        self.encoder4 = self.net.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --------------------------------------------------------------------------------
        # ----------------------- define transformer parameters --------------------------
        # --------------------------------------------------------------------------------
        # patch_dim = patch_height * patch_width * 3
        num_patches = (self.img_size // patch_height) * (self.img_size // patch_height)
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) c p1 p2', p1=patch_height, p2=patch_width)
        )
        self.to_image = nn.Sequential(
            Rearrange('b (p1 p2) h w -> b (p1 h) (p2 w)', 
            p1=self.img_size // patch_height, p2=self.img_size // patch_width)
        )
        self.patch_embedding = nn.Linear(self.filters[4]*patch_height*patch_width, self.dim)  # 2048 for 50, 512 for resnet34
        self.patch_decoding = nn.Linear(dim, self.filters[4])  # 2048 for 50, 512 for resnet34
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(0.5)

        self.transformer = Transformer(dim, depth=4, heads=8, dim_head=256, mlp_dim=256, dropout=0.5)

        self.fc = nn.Linear(self.dim, n_class)


    def forward_one(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        #print(f'x shape: {x.shape}')
        x = self.firstmaxpool(x)

        #print(f'x_ shape: {x_.shape}')
        x = self.encoder1(x)
        #print(f'e1 shape: {e1.shape}')
        x = self.encoder2(x)
        #print(f'e2 shape: {e2.shape}')
        x = self.encoder3(x)
        #print(f'e3 shape: {e3.shape}')
        x = self.encoder4(x)
        
        return x


    def forward(self, achor):## ResNetPlusTrans2.py,,, sever61
        # ---------------------------------------------------------------------------
        # ---------------------- feature extraction by ResNet -----------------------
        # ---------------------------------------------------------------------------
        # get tensor shape after blooking
        achor = self.forward_one(achor)
        #print(f'The output shape after ResNetNodown: {achor.shape}')
        x = self.to_patch(achor)
        #print(f'The output shape after patching: {x.shape}')
        (batch, num_block, c, blook_h, blook_w) = x.shape
        # ---------------------------------------------------------------------------
        # ---------------------- feature extraction by ResNet -----------------------
        # ---------------------------------------------------------------------------
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ---------------------------------------------------------------------------
        # ---------------------- start of Vit classification-------------------------
        # ---------------------------------------------------------------------------
        x = x.view(batch, num_block, -1)
        x = self.patch_embedding(x)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(num_block + 1)]
        x = self.dropout(x)
        
        #print(f'Input shape to the Transformer: {x.shape}')
        x = self.transformer(x)
        
        #print(f'Output shape after Transformer: {x.shape}')
        cls_x = x[:batch, 0, :]
        #cls_x = x.mean(dim=1)
        
        #print(f'Input shape for classification: {cls_x.shape}')
        y = self.fc(cls_x)
        # ---------------------------------------------------------------------------
        # ----------------------- end of Vit classification -------------------------
        # ---------------------------------------------------------------------------

        return y


def train(**kwargs):
    print('Data processing ..... ...... .....')
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.2),
            A.GaussianBlur(p=0.3),
        ]
    )
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize([256, 256]),
                                         ])

    train_data = Custom_train_loader('./data_ours/', './data_ours/labels_train_list.txt', img_size=224, binary=False,
                                     transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)

    valid_data = Custom_train_loader('./data_ours/', './data_ours/labels_valid_list.txt', img_size=224, binary=False,
                                     transform=test_transform, train=False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    print('Data processing done !!!')

    print('-------------- model initialization ----------------')
    model = ResNetVit(model=args.model, nodown=False, img_size=256, patch_width=4, patch_height=4, n_class=11, dim=256).to(device)
    if args.pretrain:
        print(f'Resuming the model training from path {args.weights_path}')
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.weights_path).items()})
    
    model = nn.DataParallel(model)
    
    criterion_CE = nn.CrossEntropyLoss(label_smoothing=0.0)
    criterion_l1 = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 90], gamma=0.5)

    Accuracy = []
    Loss = []
    Val_Accuracy = []
    Val_Loss = []

    patient = 15
    early_stop = 0
    best_loss = 0
    print('------------------ Training start ---------------------')
    for epoch in range(args.epoches):
        train_loss = 0.
        train_accuracy = 0.
        run_accuracy = 0.

        run_loss = 0.
        total = 0.
        model.train()
        iter = 0
        for datas, label in (train_loader):
            iter += 1
            inputs = datas[0].to(device)
            inputs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs)
            mask = datas[1].float().to(device)
            label = label.long().to(device)
            #print(f'The shape of input and label are {inputs.shape} and {label.shape}')

            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion_CE(outs, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            total += label.size(0)
            run_loss += loss.item()

            _, prediction = torch.max(outs, 1)
            run_accuracy += (prediction == label).sum().item()
            if iter % 500 == 0:
                print('epoch {} | iter {} | train accuracy: {:.4f}% | loss:  {:.4f}'.format(epoch, iter,
                                                                                            100 * run_accuracy / (
                                                                                                    label.size(
                                                                                                        0) * 500),
                                                                                            run_loss / 500))
                train_accuracy += run_accuracy
                train_loss += run_loss
                run_accuracy, run_loss = 0., 0.
        Loss.append(train_loss / len(train_loader))
        Accuracy.append(100 * train_accuracy / total)
        
        scheduler.step()

        # validation
        model.eval()
        print('Waitting for Val...')
        with torch.no_grad():
            accuracy = 0.
            total = 0.
            val_loss = 0.
            for datas, label in valid_loader:
                inputs = datas[0].to(device)
                inputs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs)
                mask = datas[1].float().to(device)
                label = label.long().to(device)

                out = model(inputs)
                loss = criterion_CE(out, label)

                _, prediction = torch.max(out, 1)
                total += label.size(0)
                accuracy += (prediction == label).sum().item()
                val_loss += loss.item()
            print('epoch {} | The Val acc is {:.4f}% | total loss is {:.4f}'.format(epoch, np.round(100. * accuracy / total, 4),
                                                                                                          val_loss/len(valid_loader)))
            val_loss = val_loss / len(valid_loader)
            val_acc = 100. * accuracy / total
            Val_Accuracy.append(val_acc)
            Val_Loss.append(val_loss)

        if not os.path.isdir('pkl20230111/{}_Vit2_S1_4_b{}'.format(args.model, args.batch)):
            os.makedirs('pkl20230111/{}_Vit2_S1_4_b{}'.format(args.model, args.batch), exist_ok=True)

        if val_acc > best_loss + 0.05:
            print(f'Saving Better Model with valid acc {val_acc} and loss {val_loss}')
            early_stop = 0
            best_loss = val_acc
            torch.save(model.state_dict(), './pkl20230111/{}_Vit2_S1_4_b{}/'
                                           '{}_Vit2_S1_4_b{}_valossACC_{}.pth'.format(
                args.model,
                args.batch,
                args.model,
                args.batch,
                np.round(best_loss, 4)))
        else:
            early_stop += 1

        if early_stop == patient:
            print('---------------------- Early stopping ---------------------')
            print(
                f'-------- the model reaches the maximun performance at epoch {epoch} -------')
            break

    print('------------------- Training done! -------------------------')


def test(**kwargs):
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize([256, 256]),
                                         ])
    test_data = Custom_test_loader('./data_ours/', './data_ours/labelsl_test_list.txt', img_size=224, binary=False,
                                   transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,
                                              num_workers=args.num_workers)  # args.batch
                                              
    print('Data processing done !!!')
    print('-------------- loading model ----------------')
    model = ResNetVit(model=args.model, nodown=True, img_size=256, patch_width=4, patch_height=4, n_class=11, dim=256).to(device)

    print(f'Resuming the model training from path {args.weights_path}')
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.weights_path).items()})

    # validation
    print('Waitting for test...')
    with torch.no_grad():
        accuracy = 0.
        total = 0
        for datas, label, orig, path in test_loader:
            inputs = datas[0].to(device)
            inputs = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(inputs)
            masks = datas[1].float().to(device)
            labels = label.long().to(device)

            out = model(inputs)

            _, prediction = torch.max(out, 1)
            total += labels.size(0)
            accuracy += (prediction == labels).sum().item()

        acc = 100. * accuracy / total

    print('The Test accuracy is {:.4f}% \n'.format(acc))
    print('------------------- Test done! -------------------------')


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script train the network.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', '-p', type=str, default='D:/Deepfake_dataset/GAN/ClimateGAN',
                        help='input folder with PNG and JPEG images')
    parser.add_argument('--weights_path', '-m', type=str, default='./pkl20221226/resnet50_Vit2_S1_4_b32/resnet50_Vit2_S1_4_b32_best_w0.01_0.99_valossACC_88.3409.pth',
                        help='weights path of the network')
    parser.add_argument('--epoches', '-e', type=int, default=50, help='epochs for training')
    parser.add_argument('--batch', '-b', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--num_workers', '-nw', type=int, default=4, help='num_workers for training')
    parser.add_argument('--classes', '-c', default=2, help='number of classes for training')
    parser.add_argument('--pretrain', '-pre', default=True, help='pretrained model weights for training')
    parser.add_argument('--model',  default='resnet50', help='pretrained model weights for training')
    args = parser.parse_args()

    #train(**vars(args))
    test(**vars(args))