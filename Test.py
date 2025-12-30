import torch
import torch.nn as nn

import numpy as np
from thop import profile
import cv2
import time
import os
from MTOIE import *
import utils_train


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def load_checkpoint(checkpoint_dir,IsGPU):
    
	if IsGPU == 1:
		model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
		net = KPTNet()
		device_ids = [0]
		model = nn.DataParallel(net, device_ids=device_ids).cuda()
		model.load_state_dict(model_info['state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(model_info['optimizer'])
		cur_epoch = model_info['epoch']

	return model, optimizer,cur_epoch

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(train_in,train_out):
	
	psnr = utils_train.batch_psnr(train_in,train_out,1.)
	return psnr


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])
	

if __name__ == '__main__': 	
	checkpoint_dir = './checkpoint/'
	test_dir = './testdata/'
	result_dir = './output/' 
	testfiles = os.listdir(test_dir)
    
	IsGPU = 1    #GPU is 1, CPU is 0

	print('> Loading dataset ...')

	lr_update_freq = 30
	model,optimizer,cur_epoch = load_checkpoint(checkpoint_dir,IsGPU)

	if IsGPU == 1:
		for f in range(len(testfiles)):
			model.eval()
			with torch.no_grad():
				img_c = cv2.imread(test_dir + '/' + testfiles[f])
				h,w,c = img_c.shape
				img_cc = img_c / 255.0
                
				img_l = hwc_to_chw(np.array(img_cc).astype('float32'))
				input_var = torch.from_numpy(img_l.copy()).type(torch.FloatTensor).unsqueeze(0).cuda()
				s = time.time()
				E_out = model(input_var,T=3)
				e = time.time()   
				print(input_var.shape)       
				print('Time:%.4f'%(e-s))    
				E_out = chw_to_hwc(E_out.squeeze().cpu().detach().numpy())	               
				cv2.imwrite(result_dir + '/' + testfiles[f][:-4] + '_KPTNetC.png',np.clip(E_out*255,0.0,255.0))
