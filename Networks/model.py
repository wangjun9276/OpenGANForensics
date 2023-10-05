from torch import nn
from Networks.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from Networks.vit import Transformer
from torch.nn import functional as F


class ResNetVit(nn.Module):

    def __init__(self, model, img_size=256, nodown=True, patch_height=32,
                 patch_width=32, n_class=1, dim=256,
                 depth=4, heads=8, droupout=0.5):
        super().__init__()
        self.nodown = nodown
        self.img_size = img_size
        self.dim = dim
        self.depth=depth
        self.heads=heads
        self.dropouts=droupout

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
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=3, stride=1, padding=0,
                                       bias=False)
        else:
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=7, stride=2, padding=3,
                                       bias=False)

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
        self.patch_embedding = nn.Linear(self.filters[4] * patch_height * patch_width,
                                         self.dim)  # 2048 for 50, 512 for resnet34
        self.patch_decoding = nn.Linear(dim, self.filters[4])  # 2048 for 50, 512 for resnet34
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.dropouts)

        self.transformer = Transformer(dim, depth=self.depth, heads=self.heads,
                                       dim_head=self.dim, mlp_dim=self.dim, dropout=self.dropouts)

        self.fc = nn.Linear(self.dim, n_class)

    def forward_one(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        # print(f'x shape: {x.shape}')
        x = self.firstmaxpool(x)

        # print(f'x_ shape: {x_.shape}')
        x = self.encoder1(x)
        # print(f'e1 shape: {e1.shape}')
        x = self.encoder2(x)
        # print(f'e2 shape: {e2.shape}')
        x = self.encoder3(x)
        # print(f'e3 shape: {e3.shape}')
        x = self.encoder4(x)

        return x

    def forward(self, achor):  ## ResNetPlusTrans2.py,,, sever61
        # ---------------------------------------------------------------------------
        # ---------------------- feature extraction by ResNet -----------------------
        # ---------------------------------------------------------------------------
        # get tensor shape after blooking
        achor = self.forward_one(achor)
        # print(f'The output shape after ResNetNodown: {achor.shape}')
        x = self.to_patch(achor)
        # print(f'The output shape after patching: {x.shape}')
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

        # print(f'Input shape to the Transformer: {x.shape}')
        x = self.transformer(x)

        # print(f'Output shape after Transformer: {x.shape}')
        cls_x = x[:batch, 0, :]
        # cls_x = x.mean(dim=1)

        # print(f'Input shape for classification: {cls_x.shape}')
        y = self.fc(cls_x)
        # ---------------------------------------------------------------------------
        # ----------------------- end of Vit classification -------------------------
        # ---------------------------------------------------------------------------

        return y


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class ResNetVitLoc(nn.Module):

    def __init__(self, model, img_size=256, nodown=True, patch_height=32,
                 patch_width=32, n_class=1, dim=256,
                 depth=4, heads=8, droupout=0.5):
        super().__init__()
        self.nodown = nodown
        self.img_size = img_size
        self.dim = dim
        self.depth=depth
        self.heads=heads
        self.dropouts=droupout

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
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=3, stride=1, padding=0,
                                       bias=False)
        else:
            self.firstconv = nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=7, stride=2, padding=3,
                                       bias=False)

        self.firstbn = self.net.bn1
        self.firstrelu = self.net.relu
        self.firstmaxpool = self.net.maxpool
        self.encoder1 = self.net.layer1
        self.encoder2 = self.net.layer2
        self.encoder3 = self.net.layer3
        self.encoder4 = self.net.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.FCN_head = FCNHead(self.filters[4], 1)

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
        self.patch_embedding = nn.Linear(self.filters[4] * patch_height * patch_width,
                                         self.dim)  # 2048 for 50, 512 for resnet34
        self.patch_decoding = nn.Linear(dim, self.filters[4])  # 2048 for 50, 512 for resnet34
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.dropouts)

        self.transformer = Transformer(dim, depth=self.depth, heads=self.heads,
                                       dim_head=self.dim, mlp_dim=self.dim, dropout=self.dropouts)

        self.fc = nn.Linear(self.dim, n_class)

    def forward_one(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        # print(f'x shape: {x.shape}')
        x = self.firstmaxpool(x)

        # print(f'x_ shape: {x_.shape}')
        x = self.encoder1(x)
        # print(f'e1 shape: {e1.shape}')
        x = self.encoder2(x)
        # print(f'e2 shape: {e2.shape}')
        x = self.encoder3(x)
        # print(f'e3 shape: {e3.shape}')
        x = self.encoder4(x)

        return x

    def forward(self, achor):  ## ResNetPlusTrans2.py,,, sever61
        # ---------------------------------------------------------------------------
        # ---------------------- feature extraction by ResNet -----------------------
        # ---------------------------------------------------------------------------
        # get tensor shape after blooking
        achor = self.forward_one(achor)
        # print(f'The output shape after ResNetNodown: {achor.shape}')
        x = self.to_patch(achor)
        # print(f'The output shape after patching: {x.shape}')
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

        # print(f'Input shape to the Transformer: {x.shape}')
        x = self.transformer(x)

        # print(f'Output shape after Transformer: {x.shape}')
        cls_x = x[:batch, 0, :]
        # cls_x = x.mean(dim=1)

        # print(f'Input shape for classification: {cls_x.shape}')
        y = self.fc(cls_x)
        # ---------------------------------------------------------------------------
        # ----------------------- end of Vit classification -------------------------
        # ---------------------------------------------------------------------------
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # ---------------------------------------------------------------------------
        # -------------------------- part of localization ---------------------------
        # ---------------------------------------------------------------------------
        loc_x = self.FCN_head(achor)
        # print(f'Output shape after FCN: {loc_x.shape}')
        loc_x = F.interpolate(loc_x, [self.img_size, self.img_size], mode="bilinear", align_corners=False)
        loc_x = F.sigmoid(loc_x)
        # print(loc_x.shape)

        return y, loc_x
