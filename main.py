'''
Official PyTorch implementation of "Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture".

If it is useful, please cite our work

@inproceedings{wang2023open,
  title={Open Set Classification of GAN-based Image Manipulations via a ViT-based Hybrid Architecture},
  author={Wang, Jun and Alamayreh, Omran and Tondi, Benedetta and Barni, Mauro},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={953--962},
  year={2023}
}
'''

import os
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.cuda import is_available as is_available_cuda
import torch.utils.data as data
import torch
from Networks.model import ResNetVitLoc, ResNetVit
from core import train, test
import warnings
warnings.filterwarnings('ignore')

from utils.data_load import Custom_train_loader, Custom_test_loader
device = 'cuda' if is_available_cuda() else 'cpu'



def main_worker(**kwargs):
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([args.img_size, args.img_size]),
                                          transforms.ToTensor(),
                                          transforms.ColorJitter(contrast=0.05, hue=0.05, brightness=0.05),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                          ])
    test_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize([args.img_size, args.img_size]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

    train_data = Custom_train_loader(args.data_path, './data_ours/labels_train_list.txt',
                                     img_size=args.img_size,
                                     binary=False,
                                     transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.num_workers)

    valid_data = Custom_train_loader(args.data_path, './data_ours/labels_valid_list.txt',
                                     img_size=args.img_size,
                                     binary=False,
                                     transform=test_transform, train=False)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.num_workers)


    if args.loc:
        model = ResNetVitLoc(model=args.model, nodown=args.nodown, img_size=args.img_size, patch_width=args.patch_size,
                             patch_height=args.patch_size, n_class=args.classes,
                             dim=args.dim, depth=args.depth, heads=args.head,
                             droupout=args.drop).to(device)
    else:
        model = ResNetVit(model=args.model, nodown=args.nodown, img_size=args.img_size, patch_width=args.patch_size,
                          patch_height=args.patch_size, n_class=args.classes,
                             dim=args.dim, depth=args.depth, heads=args.head,
                             droupout=args.drop).to(device)

    if args.pretrain:
        print(f'Resuming the model training from path {args.weights_path}')
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.weights_path).items()})

    model = nn.DataParallel(model)

    criterion_CE = nn.CrossEntropyLoss(label_smoothing=0.0)
    criterion_l1 = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 90], gamma=0.5)

    if not os.path.isdir('{}/{}_Patch{}_Dim{}_b{}'.format(args.save_model, args.model, args.patch_size, args.dim,
                                                          args.batch)):
        os.makedirs('{}/{}_Patch{}_Dim{}_b{}'.format(args.save_model, args.model, args.patch_size, args.dim, args.batch),
                    exist_ok=True)

    patient = args.patient
    early_stop = 0
    best_acc = 0
    for epoch in range(args.epoches):
        train(epoch, train_loader, model, optimizer, criterion_CE, criterion_l1, device, args)
        scheduler.step()

        val_acc, val_loss = test(epoch, valid_loader, model, criterion_CE, criterion_l1, device, args)

        if val_acc > best_acc + 0.05:
            print(f'Saving Better Model with valid acc {val_acc} and loss {val_loss}')
            early_stop = 0
            best_acc = val_acc
            torch.save(model.state_dict(), './{}/{}_Patch{}_Dim{}_b{}/'
                                           '{}_Patch{}_Dim{}_b{}_best.pth'.format(
                args.save_model, args.model, args.patch_size, args.dim, args.batch,
                args.model, args.patch_size, args.dim, args.batch))
        else:
            early_stop += 1

        if early_stop == patient:
            print('---------------------- Early stopping ---------------------')
            print(
                f'-------- the model reaches the maximun performance at epoch {epoch} -------')
            break


if __name__ == '__main__':
    # ## parameters for hybrid network
    # lambda_locs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    # lambda_clss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    parser = argparse.ArgumentParser(description="This script train the network.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', '-p', type=str, default='./data_ours/',
                        help='input folder with paths for our set')
    parser.add_argument('--weights_path', '-m', type=str, default=None,
                        help='weights path of the network')
    parser.add_argument('--save_model', type=str, default='./save_models/',
                        help='path to save trained models')
    parser.add_argument('--epoches', '-e', type=int, default=50, help='epochs for training')
    parser.add_argument('--batch', '-b', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--num_workers', '-nw', type=int, default=4, help='num_workers for training')
    parser.add_argument('--classes', '-c', default=2, help='number of classes for training')
    parser.add_argument('--pretrain', '-pre', default=True, help='define to resume training with pretrained model or not')
    parser.add_argument('--model', type=str, default='resnet50', help='pretrained model weights for training')

    parser.add_argument('--loc', action='store_true', default=False, help='load hybrid network for loc and cls if True')
    parser.add_argument('--patch_size', type=int, default=4, help='patch size for Vit classifier')
    parser.add_argument('--depth', type=int, default=4, help='depth for Vit classifier')
    parser.add_argument('--head', type=int, default=8, help='heads for Vit classifier')
    parser.add_argument('--drop', type=float, default=0.5, help='dropout ratio for Vit classifier')
    parser.add_argument('--img_size', type=int, default=256, help='input size for the network')
    parser.add_argument('--dim',  type=int, default=256, help='dim for Vit classifier')
    parser.add_argument('--freq', type=int, default=50, help='print training info after every freq iterations')
    parser.add_argument('--lambda_locs', type=int, default=0.2, help='localization weight in hybrid network')
    parser.add_argument('--lambda_clss', type=int, default=0.8, help='classification weight in hybrid network')
    parser.add_argument('--patient', type=int, default=15, help='used for early stopping if no improvement')
    parser.add_argument('--nodown', action='store_true', default=True, help='no down in the network')
    args = parser.parse_args()


    main_worker(**vars(args))