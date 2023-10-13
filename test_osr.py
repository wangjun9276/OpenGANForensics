import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.cuda import is_available as is_available_cuda
import torch.utils.data as data
import torch
from Networks.model import ResNetVitLoc, ResNetVit
from core import test_osr
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
    #print(args.masks)
    known_data = Custom_train_loader(args.data_path, args.data_path+'labelsk_test_list.txt',
                                     img_size=args.img_size,
                                     binary=False,
                                     train='_test.csv',
                                     mask=args.masks,
                                     loc=args.loc,
                                     transform=train_transform)
    known_loader = torch.utils.data.DataLoader(known_data, batch_size=args.batch, shuffle=True,
                                               num_workers=args.num_workers)

    unnknown_data = Custom_train_loader(args.data_path, args.data_path+'labelsu_test_list.txt',
                                     img_size=args.img_size,
                                     binary=False,
                                     train='_test.csv',
                                     mask=args.masks,
                                     loc=args.loc,
                                     transform=test_transform)
    unknown_loader = torch.utils.data.DataLoader(unnknown_data, batch_size=args.batch, shuffle=True,
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
    test_osr(known_loader, unknown_loader, model, device, args)



if __name__ == '__main__':
    # ## parameters for hybrid network
    # lambda_locs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
    # lambda_clss = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    parser = argparse.ArgumentParser(description="This script train the network.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', '-p', type=str, default='./data/',
                        help='input folder with paths for our set')
    parser.add_argument('--weights_path', '-m', type=str, default=None,
                        help='weights path of the network')
    parser.add_argument('--save_model', type=str, default='./save_models/',
                        help='path to save trained models')
    parser.add_argument('--epoches', '-e', type=int, default=50, help='epochs for training')
    parser.add_argument('--batch', '-b', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--num_workers', '-nw', type=int, default=4, help='num_workers for training')
    parser.add_argument('--classes', '-c', type=int, default=11, help='number of classes for training')
    parser.add_argument('--pretrain', action='store_true', default=False, help='define to resume training with pretrained model or not')
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
    parser.add_argument('--nodown', action='store_true', default=False, help='no down in the network')
    parser.add_argument('--masks', action='store_true', default=False, help='if true, read masks (specify for facial attribute edit classification task)')
    args = parser.parse_args()


    main_worker(**vars(args))
