import math

import torch

def make_anchors(x, strides, offset=0.5):
    """
    Generate anchors from features
    """
    assert x is not None
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset  # shift x
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def pad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(torch.nn.Module):
    def __init__(self, conv, in_ch, out_ch, k=1, s=1, p=None, d=1, g=1, bn=0, act=0):
        super().__init__()
        self.bn   = bn
        self.act  = act
        if type(conv) is torch.nn.Conv2d:
            self.conv = conv(in_ch, out_ch,
                          k, stride=s, padding=pad(k, p, d), 
                          dilation=d, groups=g, bias=False)
        else:
            self.conv = conv(in_ch, out_ch,
                            k, stride=s, padding=pad(k, p, d), 
                            dilation=d, groups=g)

        if self.bn:
            self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.03)
        if self.act:
            self.relu = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.bn:
            out = self.norm(out) 
        if self.act:
            out = self.relu(out)
        return out


class Residual(torch.nn.Module):
    def __init__(self, conv, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(conv, ch, ch, 3),
                                         Conv(conv, ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class CSP(torch.nn.Module):
    def __init__(self, conv, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(conv, in_ch, out_ch // 2)
        self.conv2 = Conv(conv, in_ch, out_ch // 2)
        self.conv3 = Conv(conv, (2 + n) * out_ch // 2, out_ch)
        self.res_m = torch.nn.ModuleList(Residual(conv, out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, conv, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(conv, in_ch, in_ch // 2)
        self.conv2 = Conv(conv, in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, conv, width, depth):
        super().__init__()
        p1 = [Conv(conv, width[0], width[1], 3, 2)]
        p2 = [Conv(conv, width[1], width[2], 3, 2),
              CSP(conv, width[2], width[2], depth[0])]
        p3 = [Conv(conv, width[2], width[3], 3, 2),
              CSP(conv, width[3], width[3], depth[1])]
        p4 = [Conv(conv, width[3], width[4], 3, 2),
              CSP(conv, width[4], width[4], depth[2])]
        p5 = [Conv(conv, width[4], width[5], 3, 2),
              CSP(conv, width[5], width[5], depth[0]),
              SPP(conv, width[5], width[5])]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, conv, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(conv, width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(conv, width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(conv, width[3], width[3], 3, 2)
        self.h4 = CSP(conv, width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(conv, width[4], width[4], 3, 2)
        self.h6 = CSP(conv, width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))
        return h2, h4, h6


class DFL(torch.nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, conv, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c1 = max(filters[0], self.nc)
        c2 = max((filters[0] // 4, self.ch * 4))
        self.flag = (type(conv) is torch.nn.Conv2d)

        self.dfl = DFL(self.ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(conv ,x, c1, 3),
                                                           Conv(conv, c1, c1, 3),
                                                           conv(c1, self.nc, 1)) for x in filters)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(conv, x, c2, 3),
                                                           Conv(conv, c2, c2, 3),
                                                           conv(c2, 4 * self.ch, 1)) for x in filters)

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.box[i](x[i]), self.cls[i](x[i])), 1)
        if self.training:
            return x
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.ch * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        if not self.flag:
            return
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, conv, width, depth, num_classes):
        super().__init__()
        self.net = DarkNet(conv, width, depth)
        self.fpn = DarkFPN(conv, width, depth)

        img_dummy = torch.zeros(1, 3, 256, 256)
        self.head = Head(conv, num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))



def yolo_v8_n(conv, num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(conv, width, depth, num_classes)


def yolo_v8_s(conv, num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(conv ,width, depth, num_classes)


def yolo_v8_m(conv, num_classes: int = 80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(conv, width, depth, num_classes)


def yolo_v8_l(conv, num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(conv, width, depth, num_classes)


def yolo_v8_x(conv, num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(conv, width, depth, num_classes)