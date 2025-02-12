import torch
import torch.nn as nn
import torch.nn.functional as F

class MToIE(nn.Module):
    def __init__(self):
        super(MToIE, self).__init__()
        self.mns = MainNetworkStructure(3, 16)

    def forward(self, x, T):
        Fout = self.mns(x, T)
        return Fout
    
class TaskSpecificModule(nn.Module):
    def __init__(self, task_type, channel):
        super(TaskSpecificModule, self).__init__()
        if task_type == 1:
            self.task_layer = nn.Conv2d(channel*4, channel*4, kernel_size=3, padding=1)
        elif task_type == 2:
            self.task_layer = nn.Conv2d(channel*4, channel*4, kernel_size=3, padding=1)
        elif task_type == 3:
            self.task_layer = nn.Conv2d(channel*4, channel*4, kernel_size=3, padding=1)
    def forward(self, x):
        return self.task_layer(x)

class MainNetworkStructure(nn.Module):
    def __init__(self, inchannel, channel):
        super(MainNetworkStructure, self).__init__()
        self.en = Encoder(channel)
        self.de = Decoder(channel)

        self.task_specific_modules = nn.ModuleDict({
            '1':TaskSpecificModule(1,channel),
            '2':TaskSpecificModule(1,channel),
            '3':TaskSpecificModule(1,channel)
        })

        self.ne1 = NodeEncoder(channel)
        self.ne2 = NodeEncoder(channel)
        self.ne3 = NodeEncoder(channel)

        self.nd1 = NodeDecoder(channel)
        self.nd2 = NodeDecoder(channel)
        self.nd3 = NodeDecoder(channel)

        self.conv_in   = nn.Conv2d(3, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2t1 = nn.Conv2d(channel*8, channel*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_out  = nn.Conv2d(channel, 3, kernel_size=1, stride=1, padding=0, bias=False)

        self.attn1 = SelfAttention(channel * 4)
        self.attn2 = SelfAttention(channel * 4)

        self.alpha_param1 = nn.Parameter(torch.tensor(0.3), requires_grad=True)
        self.alpha_param2 = nn.Parameter(torch.tensor(0.3), requires_grad=True)

        self.gamma = nn.Parameter(torch.zeros(1))            
                        
    def forward(self, x, T):
    
        x_in = self.conv_in(x)       
        x_e3, x_e2, x_e1 = self.en(x_in) 

        if T == 1:
            x_e31 = self.task_specific_modules[str(T)]
            x_e32 = x_e31(x_e3)
            x_ne = self.ne1(x_e32)
            x_nd = self.nd1(x_ne,x_e3)

            x_ne3_2 = self.ne3(self.attn1(x_nd, x_e32))
            x_nd = self.nd3(x_ne3_2,x_e3)
            
        elif T == 2:
            x_e31 = self.task_specific_modules[str(T)]
            x_e32 = x_e31(x_e3)
            x_ne = self.ne2(x_e32)
            x_nd = self.nd2(x_ne,x_e3)

            x_ne3_2 = self.ne3(self.attn2(x_nd,x_e32)) 
            x_nd = self.nd3(x_ne3_2,x_e3)
            
        elif T == 3:                 
            x_e31 = self.task_specific_modules[str(T)]
            x_e32 = x_e31(x_e3)
            
            x_neh = self.ne1(x_e32)
            x_ndh = self.nd1(x_neh,x_e3)

            x_nel = self.ne2(x_e32)
            x_ndl = self.nd2(x_nel,x_e3)
      
            alpha = torch.sigmoid(self.alpha_param1)
            x_ne3_2 = self.ne3(alpha * self.attn1(x_ndh, x_e32) + 
                            (1-alpha) * self.attn2(x_ndl, x_e32))
            x_nd = self.nd3(x_ne3_2,x_e3)
            	    
        x_de = self.de(x_nd, x_e3, x_e2, x_e1)
        
        x_out = self.conv_out(x_de)
        
        return x_out

class Encoder(nn.Module):
    def __init__(self, channel):
        super(Encoder, self).__init__()
        self.e1 = BB(channel)
        self.e2 = BB(channel * 2)
        self.e3 = BB(channel * 4)
        self.conv_e1te2 = nn.Conv2d(channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_e2te3 = nn.Conv2d(2 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        e1out = self.e1(x)
        e2out = self.e2(self.conv_e1te2(self.maxpool(e1out)))
        e3out = self.e3(self.conv_e2te3(self.maxpool(e2out)))
        return e3out, e2out, e1out

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.d1 = BB(channel * 4)
        self.d2 = BB(channel * 2)
        self.d3 = BB(channel)
        
        self.conv_d1td2 = nn.Conv2d(4 * channel, 2 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_d2td3 = nn.Conv2d(2 * channel, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def _upsample(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def forward(self, x, e3out, e2out, e1out):
        d1out = self.d1(x + e3out)
        d2out = self.d2(self._upsample(self.conv_d1td2(d1out), e2out) + e2out)
        d3out = self.d3(self._upsample(self.conv_d2td3(d2out), e1out) + e1out)
        return d3out
 
class NodeEncoder(nn.Module):
    def __init__(self, channel):
        super(NodeEncoder, self).__init__()
        self.m1 = MRFE(channel * 4)
        self.m2 = MRFE(channel * 8)
        self.conv_e = nn.Conv2d(4 * channel, 8 * channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        xm1 = self.m1(x)
        xm2 = self.m2(self.conv_e(self.maxpool(xm1)))
        return xm2

class NodeDecoder(nn.Module):
    def __init__(self, channel):
        super(NodeDecoder, self).__init__()
        self.m3 = MRFE(channel * 8)
        self.m4 = MRFE(channel * 4)
        self.conv_d = nn.Conv2d(8 * channel, 4 * channel, kernel_size=1, stride=1, padding=0, bias=False)

    def _upsample(self, x,y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def forward(self, xm2,xu):
        xm3 = self.m3(xm2)
        xm4 = self.m4(self._upsample(self.conv_d(xm3),xu))
        return xm4

class SelfAttention(nn.Module):
    def __init__(self, channel):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(channel, channel // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, ref):
        
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(ref).view(batch_size, -1, width * height)

        energy = torch.bmm(proj_query, proj_key) / (C//4) ** 0.5
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma *out + x        
        
        return out

class BB(nn.Module):
    def __init__(self, channel):
        super(BB, self).__init__()
        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.PReLU(channel)
        self.norm = nn.GroupNorm(num_channels=channel, num_groups=1)

    def forward(self, x):
        x_1 = self.act(self.norm(self.conv_1(x)))
        x_2 = self.act(self.norm(self.conv_2(x_1)))
        x_out = self.act(self.norm(self.conv_3(x_2)) + x)
        return x_out

class MRFE(nn.Module):
    def __init__(self, channel):
        super(MRFE, self).__init__()
        self.dw1 = nn.Conv2d(channel, channel, 3, 1, 1, groups=channel)
        self.dw3 = nn.Conv2d(channel, channel, 3, 1, 3, groups=channel, dilation=3)
        self.dw5 = nn.Conv2d(channel, channel, 3, 1, 5, groups=channel, dilation=5)
        
        self.pw = nn.Conv2d(channel*3, channel, 1)
        self.norm = nn.GroupNorm(4, channel)
        self.relu = nn.PReLU()
        
    def forward(self, x):
        x1 = self.dw1(x)
        x3 = self.dw3(x)
        x5 = self.dw5(x)
        out = self.pw(torch.cat([x1, x3, x5], 1))
        out = self.relu(self.norm(out))
        return x + out
