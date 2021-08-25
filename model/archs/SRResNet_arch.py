import functools
import paddle.nn as nn
import paddle.nn.functional as F
import models.archs.arch_util as arch_util
import paddle

class MSRResNet(nn.Layer):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2D(nf, nf * 9, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.upconv2 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

class MSRResNet_scSE(nn.Layer):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet_scSE, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        # TODO check
        basic_block = functools.partial(arch_util.ResidualBlock_noBN_scSE, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2D(nf, nf * 9, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.upconv2 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class MSRResNet_Bottleneck_Inverted_Residual(nn.Layer):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet_Bottleneck_Inverted_Residual, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        # TODO check
        basic_block = functools.partial(arch_util.ResidualBlock_noBN_Bottleneck_Inverted_Residual, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2D(nf, nf * 9, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.upconv2 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

class MSRResNet_Bottleneck_Inverted_Residual_scSE(nn.Layer):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet_Bottleneck_Inverted_Residual_scSE, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2D(in_nc, nf, 3, 1, 1, bias_attr=True)
        # TODO check ResidualBlock_noBN_Bottleneck_Inverted_Residual_scSE
        basic_block = functools.partial(arch_util.ResidualBlock_noBN_Bottleneck_Inverted_Residual_scSE, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2D(nf, nf * 9, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.upconv2 = nn.Conv2D(nf, nf * 4, 3, 1, 1, bias_attr=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2D(nf, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, out_nc, 3, 1, 1, bias_attr=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

