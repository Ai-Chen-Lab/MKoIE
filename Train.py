import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import cv2
import scipy.misc
from MToIE import *
from makedataset import Dataset
import utils_train
from Test_SSIM_A import *
from torchvision.models import vgg16

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"

class FocusedLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(FocusedLoss, self).__init__()
        
        vgg = vgg16(pretrained=True).features[:16]
        vgg = vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.cuda(gpu_id)
        
        self.lambda_recon = 0.8  
        self.lambda_percep = 0.2 
        
    def get_vgg_features(self, x, layers=[3, 8, 15]):
        features = []
        for i, module in enumerate(self.vgg):
            x = module(x)
            if i in layers:
                features.append(x)
        return features
    
    def reconstruction_loss(self, pred, target):
        l1_loss = F.l1_loss(pred, target)
        char_eps = 1e-6
        char_loss = torch.mean(torch.sqrt((pred - target) ** 2 + char_eps))
        return 0.5 * l1_loss + 0.5 * char_loss
    
    def perceptual_loss(self, pred, target):
        pred_features = self.get_vgg_features(pred)
        target_features = self.get_vgg_features(target)
        
        percep_loss = 0
        for pf, tf in zip(pred_features, target_features):
            percep_loss += F.mse_loss(pf, tf)
        return percep_loss / len(pred_features)
    
    def forward(self, pred, target):
        recon_loss = self.reconstruction_loss(pred, target)
        percep_loss = self.perceptual_loss(pred, target)
        
        total_loss = (self.lambda_recon * recon_loss + 
                     self.lambda_percep * percep_loss)
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'percep_loss': percep_loss.item()
        }

def load_checkpoint(checkpoint_dir):
    if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
        model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
        print('loading existing model ......', checkpoint_dir + 'checkpoint.pth.tar')
        net = MToIE()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch']
    else:
        net = MToIE()
        device_ids = [0]
        model = nn.DataParallel(net, device_ids=device_ids).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cur_epoch = 0
    
    return model, optimizer, cur_epoch

def save_checkpoint(state, epoch, is_best, PSNR, SSIM, filename='checkpointA.pth.tar'):
    torch.save(state, checkpoint_dir + 'checkpointA_%d_%.4f_%.4f.pth.tar'%(epoch,PSNR,SSIM))
    if is_best:
        shutil.copyfile(checkpoint_dir + 'checkpointA.pth.tar',checkpoint_dir + 'model_best.pth.tar')
        
def adjust_learning_rate(optimizer, epoch, lr_update_freq):
    if not epoch % lr_update_freq and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
            print(param_group['lr'])
    return optimizer

def train_psnr(train_in, train_out):
    psnr = utils_train.batch_psnr(train_in, train_out, 1.)
    return psnr
    
def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

if __name__ == '__main__': 
    checkpoint_dir = './checkpoint/'
    test_dir = './testdata/'
    result_dir = './result/'
    testfiles = os.listdir(test_dir)   
    
    print('> Loading dataset ...')
    dataset = Dataset(trainrgb=True, trainsyn=True, shuffle=True)
    loader_dataset = DataLoader(dataset=dataset, num_workers=0, batch_size=20, shuffle=True)
    count = len(loader_dataset)
    
    lr_update_freq = 60
    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir)
    
    focused_loss = FocusedLoss(gpu_id=0).cuda()
    
    for epoch in range(cur_epoch, 200):
        optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
        learnrate = optimizer.param_groups[-1]['lr']
        model.train()
            
        for i, data in enumerate(loader_dataset, 0):
            try:
                img_c = torch.zeros(data[:,0:3,:,:].size())    
                img_l = torch.zeros(data[:,0:3,:,:].size())      
                Type = np.random.randint(1,4)    
                    
                for nx in range(data.shape[0]):    
                    img_c[nx,:,:,:] = data[nx,0:3,:,:]           
                                        
                for nxx in range(data.shape[0]):  
                    if Type == 1: 
                        img_l[nxx] = data[nxx,3:6,:,:] 
                    elif Type == 2:   
                        img_l[nxx] = data[nxx,6:9,:,:]
                    else:    
                        img_l[nxx] = data[nxx,9:12,:,:]                      
                                            
                input_var = Variable(img_l.cuda(), volatile=True)  
                target_final = Variable(img_c.cuda(), volatile=True)            
                                
                eout = model(x=input_var, T=Type)
                
                total_loss, loss_dict = focused_loss(eout, target_final)
                
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Warning: Loss is NaN or Inf, skipping batch")
                    continue
                    
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                SN1_psnr = train_psnr(target_final, eout)
                
                print("[Epoch %d][Type %d][%d/%d] lr:%f total_loss: %.4f recon_loss: %.4f "
                      "percep_loss: %.4f PSNR_train: %.4f" 
                      %(epoch+1, Type, i+1, count, learnrate, total_loss.item(), 
                        loss_dict['recon_loss'], loss_dict['percep_loss'], SN1_psnr))
                
            except Exception as e:
                print(f"Error during training iteration: {str(e)}")
                continue

        model.eval()

        with torch.no_grad():
            for f in range(len(testfiles)):
                try:
                    img = cv2.imread(test_dir + '/' + testfiles[f])
                    h, w, c = img.shape
                    img_ccc = cv2.resize(img, (512,512)) / 255.0
                    img_h = hwc_to_chw(img_ccc)
                    input_var = torch.from_numpy(img_h.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
                 
                    s = time.time()
                    e_out = model(input_var, T=3)
                    e_out = chw_to_hwc(e_out.squeeze().cpu().detach().numpy())              
                    e = time.time()           
                    e_out = cv2.resize(e_out, (w,h))
                                             
                    cv2.imwrite(result_dir + '/' + testfiles[f], np.clip(e_out*255, 0.0, 255.0))
                except Exception as e:
                    print(f"Error during testing: {str(e)}")
                    continue

            try:
                ps, ss = C_PSNR_SSIM()
                    
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, epoch, is_best=0, PSNR=ps, SSIM=ss)
                
            except Exception as e:
                print(f"Error during checkpoint saving: {str(e)}")
