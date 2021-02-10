import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.ab_norm = 110.
 
        self.model1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64))
        
        self.model2=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128))
        
        self.model3=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256))
        
        self.model4=nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        
        self.model5=nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,dilation=2,stride=1,padding=2),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,dilation=2,stride=1,padding=2),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,dilation=2,stride=1,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        
        self.model6=nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,dilation=2,stride=1,padding=2),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,dilation=2,stride=1,padding=2),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,dilation=2,stride=1,padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        
        self.model7=nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512))
        
        self.model8=nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,313,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(313)
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(313, 2, kernel_size=1, padding=0, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
    def unnormalize_ab(self, in_ab):
        return in_ab*self.ab_norm
    
    
    def forward(self, x):
        input_1 = x
        conv1_2 = self.model1(input_1)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.out(self.softmax(conv8_3))
        color_ab = self.upsample4(out_reg)

        return self.softmax(conv8_3), self.unnormalize_ab(color_ab)

def Colormodel():
    return Model