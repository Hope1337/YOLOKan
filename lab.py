from kan_convs import KANConv2DLayer
from utils.lib import *
from models import yolo_v8_x, yolo_v8_s, yolo_v8_l, yolo_v8_m, yolo_v8_n
from utils.measure import time_run


model1  = yolo_v8_m(KANConv2DLayer, 80)
model2  = yolo_v8_m(torch.nn.Conv2d, 80)
images = torch.randn(8, 3, 640, 640)

print(time_run(model2, images))
print(time_run(model1, images))

