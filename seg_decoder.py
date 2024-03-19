import torch
import torch.nn as nn
import einops
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, kernel_size=0, padding=0, stride=1):
        super(ConvModule, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class SAMAggregatorNeck(nn.Module):
    def __init__(
            self,
            # in_channels=[1280]*32,          
            # in_channels=[384]*12,       #vits
            # in_channels=[192]*12,       #vitt
            in_channels=[384]*12,          #vits_rellis_3d
            inner_channels=128,    
            selected_channels = range(1, 12, 1),        
            out_channels=256,
            up_sample_scale=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.selected_channels = selected_channels
        self.up_sample_scale = up_sample_scale

        self.down_sample_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.down_sample_layers.append(
                nn.Sequential(
                    ConvModule(
                        in_channels[idx],
                        inner_channels,
                        kernel_size=1,
                    ),
                    ConvModule(
                        inner_channels,
                        inner_channels,
                        kernel_size=3,
                        padding=1,
                        stride=1,
                    ),
                )
            )
        self.fusion_layers = nn.ModuleList()
        for idx in self.selected_channels:
            self.fusion_layers.append(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        self.up_layers = nn.ModuleList()
        self.up_layers.append(
            nn.Sequential(
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                ),
                ConvModule(
                    inner_channels,
                    inner_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )
        self.up_layers.append(
            ConvModule(
                inner_channels,
                out_channels,
                kernel_size=1,
            )
        )

        self.up_sample_layers = nn.ModuleList()
        assert up_sample_scale == 4
        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.up_sample_layers.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )

        self.up_sample_layers.append(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    

    def forward(self, inputs):
        image_embedding, inner_states = inputs        
        ######################
      
        inner_states = [einops.rearrange(inner_states[idx], 'b h w c -> b c h w') for idx in self.selected_channels]
        inner_states = [layer(x) for layer, x in zip(self.down_sample_layers, inner_states)]

        x = None
        for inner_state, layer in zip(inner_states, self.fusion_layers):
            if x is not None:
                inner_state = x + inner_state
            x = inner_state + layer(inner_state)
        x = self.up_layers[0](x) + x
        img_feats_0 = self.up_layers[1](x)

        img_feats_1 = self.up_sample_layers[0](img_feats_0) + self.up_sample_layers[1](img_feats_0)

        img_feats_2 = self.up_sample_layers[2](img_feats_1) + self.up_sample_layers[3](img_feats_1)

        return img_feats_2, img_feats_1, img_feats_0, image_embedding




class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)

        return x





class SegHead(nn.Module):
    def __init__(self,):
        super(SegHead, self).__init__()
        self.in_channels = [256, 256, 256, 256]

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = 256

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_ch=embedding_dim * 4,
            out_ch=embedding_dim,
            kernel_size=1,
        )
        
        ### sam_l
        # self.neck_net = SAMAggregatorNeck(in_channels=[1024]*24, selected_channels = range(4, 24, 2))

        ### sam_h
        self.neck_net = SAMAggregatorNeck()



        self.linear_pred = nn.Conv2d(embedding_dim, 2, kernel_size=1)

    def forward(self, inputs):
        x = self.neck_net(inputs)

        c1, c2, c3, c4 = x
        # print('c4.shape:',c4.shape)
        # print('c1.shape:',c1.shape)
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.linear_pred(x)

        return x



class SegHeadUpConv(nn.Module):
    def __init__(self):
        super(SegHeadUpConv, self).__init__()

        self.UpConv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 2, kernel_size=1)
        )

    def forward(self, inputs):
        # image_embedding, inner_states = inputs
        image_embedding = inputs 
        x = self.UpConv(image_embedding)
        x = self.seg_head(x)

        return x
