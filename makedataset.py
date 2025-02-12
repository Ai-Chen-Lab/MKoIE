import os
import os.path
import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata

class Dataset(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, trainrgb=True,trainsyn = True, shuffle=False):
		super(Dataset, self).__init__()
		self.trainrgb = trainrgb
		self.trainsyn = trainsyn
		self.train_haze	 = 'Train_GT2LV_GT.h5'
		
		if self.trainrgb:
			if self.trainsyn:
				h5f = h5py.File(self.train_haze, 'r')
			else:
				h5f = h5py.File(self.train_real_rgb, 'r')				 
		else:
			if self.trainsyn:				 
				h5f = h5py.File(self.train_syn_gray, 'r')
			else:
				h5f = h5py.File(self.train_real_gray, 'r')			  
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		if self.trainrgb:
			if self.trainsyn:
				h5f = h5py.File(self.train_haze, 'r')
			else:
				h5f = h5py.File(self.train_real_rgb, 'r')				 
		else:
			if self.trainsyn:				 
				h5f = h5py.File(self.train_syn_gray, 'r')
			else:
				h5f = h5py.File(self.train_real_gray, 'r')			  
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)

class Dataset_RainMaps(udata.Dataset):
	r"""Implements torch.utils.data.Dataset
	"""
	def __init__(self, trainrgb=True,trainsyn = True, shuffle=False):
		super(Dataset_RainMaps, self).__init__()
		self.trainrgb = trainrgb
		self.trainsyn = trainsyn
		self.train_haze	 = 'RainMaps.h5'
		
		if self.trainrgb:
			if self.trainsyn:
				h5f = h5py.File(self.train_haze, 'r')
			else:
				h5f = h5py.File(self.train_real_rgb, 'r')				 
		else:
			if self.trainsyn:				 
				h5f = h5py.File(self.train_syn_gray, 'r')
			else:
				h5f = h5py.File(self.train_real_gray, 'r')			  
		self.keys = list(h5f.keys())
		if shuffle:
			random.shuffle(self.keys)
		h5f.close()

	def __len__(self):
		return len(self.keys)

	def __getitem__(self, index):
		if self.trainrgb:
			if self.trainsyn:
				h5f = h5py.File(self.train_haze, 'r')
			else:
				h5f = h5py.File(self.train_real_rgb, 'r')				 
		else:
			if self.trainsyn:				 
				h5f = h5py.File(self.train_syn_gray, 'r')
			else:
				h5f = h5py.File(self.train_real_gray, 'r')			  
		key = self.keys[index]
		data = np.array(h5f[key])
		h5f.close()
		return torch.Tensor(data)
    
def data_augmentation(image, mode):
	r"""Performs dat augmentation of the input image

	Args:
		image: a cv2 (OpenCV) image
		mode: int. Choice of transformation to apply to the image
			0 - no transformation
			1 - flip up and down
			2 - rotate counterwise 90 degree
			3 - rotate 90 degree and flip up and down
			4 - rotate 180 degree
			5 - rotate 180 degree and flip
			6 - rotate 270 degree
			7 - rotate 270 degree and flip
	"""
	out = np.transpose(image, (1, 2, 0))
	if mode == 0:
		# original
		out = out
	elif mode == 1:
		# flip up and down
		out = np.flipud(out)
	elif mode == 2:
		# rotate counterwise 90 degree
		out = np.rot90(out)
	elif mode == 3:
		# rotate 90 degree and flip up and down
		out = np.rot90(out)
		out = np.flipud(out)
	elif mode == 4:
		# rotate 180 degree
		out = np.rot90(out, k=2)
	elif mode == 5:
		# rotate 180 degree and flip
		out = np.rot90(out, k=2)
		out = np.flipud(out)
	elif mode == 6:
		# rotate 270 degree
		out = np.rot90(out, k=3)
	elif mode == 7:
		# rotate 270 degree and flip
		out = np.rot90(out, k=3)
		out = np.flipud(out)
	else:
		raise Exception('Invalid choice of image transformation')
	return np.transpose(out, (2, 0, 1))

def img_to_patches(img,win,stride,Syn=True):
	
	chl,raw,col = img.shape
	chl = int(chl)
	num_raw = np.ceil((raw-win)/stride+1).astype(np.uint8)
	num_col = np.ceil((col-win)/stride+1).astype(np.uint8) 
	count = 0
	total_process = int(num_col)*int(num_raw)
	img_patches = np.zeros([chl,win,win,total_process])
	if Syn:
		for i in range(num_raw):
			for j in range(num_col):			   
				if stride * i + win <= raw and stride * j + win <=col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, stride*j : stride*j + win]				 
				elif stride * i + win > raw and stride * j + win<=col:
					img_patches[:,:,:,count] = img[:,raw-win : raw,stride * j : stride * j + win]		   
				elif stride * i + win <= raw and stride*j + win>col:
					img_patches[:,:,:,count] = img[:,stride*i : stride*i + win, col-win : col]
				else:
					img_patches[:,:,:,count] = img[:,raw-win : raw,col-win : col]				
				count +=1		   
		
	return img_patches


def readfiles(filepath):
	'''Get dataset images names'''
	files = os.listdir(filepath)
	return files

def normalize(data):

	return np.float32(data/255.0)

def get_dark_channel(image, size=7):
    """
    Calculate the dark channel of the image.
    """
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def get_light_channel(image, size=7):
    """
    Calculate the dark channel of the image.
    """
    b, g, r = cv2.split(image)
    min_img = cv2.max(cv2.max(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.dilate(min_img, kernel)
    return dark_channel

def guided_filter(I, p, r=15, eps=1e-3):
    """
    Perform guided filter on the image.
    I: guidance image (should be a grayscale/single channel image)
    p: filtering input image (should be a grayscale/single channel image)
    r: radius of the guided filter
    eps: regularization parameter
    """
    I = I.astype(np.float64)
    p = p.astype(np.float64)
    
    mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    
    q = mean_a * I + mean_b
    return q

def concatenateimgs(cimg,himg,limg,hlimg,dcp,lcp,depth):
	c,w,h = cimg.shape
	conimg = np.zeros((c*4+3,w,h))
	conimg[0:c,:,:] = cimg
	conimg[c:2*c,:,:] = himg
	conimg[2*c:3*c,:,:] = limg
	conimg[3*c:4*c,:,:] = hlimg
	
	conimg[4*c:4*c+1,:,:] = dcp
	conimg[4*c+1:4*c+2,:,:] = lcp
	conimg[4*c+2:4*c+3,:,:] = depth
		
	return conimg

def concatenate1imgs(cimg,dcp,lcp,depth):
	c,w,h = cimg.shape
	conimg = np.zeros((c+3,w,h))
	conimg[0:c,:,:] = cimg	
	conimg[c:c+1,:,:] = dcp
	conimg[c+1:c+2,:,:] = lcp
	conimg[c+2:c+3,:,:] = depth
		
	return conimg
		
def TrainGT(cimg_filepath,himg_filepath,limg_filepath,hlimg_filepath, depth_filepath, patch_size, stride):
	'''synthetic ImageEdge images'''
	train_haze = 'Train_GT2LV_GT.h5'
	img_files = readfiles(cimg_filepath)
	count = 0
		
	with h5py.File(train_haze, 'w') as h5f:
		for i in range(len(img_files)):
			filename    = img_files[i]
            
			clearimg     = normalize(cv2.imread(cimg_filepath + '/' + filename))
			hazeimg      = normalize(cv2.imread(himg_filepath + '/' + filename))
			lowimg       = normalize(cv2.imread(limg_filepath + '/' + filename))
			hazelowimg   = normalize(cv2.imread(hlimg_filepath + '/' + filename))
			imgdepth     = normalize(cv2.imread(depth_filepath + '/' + filename,0))
						
			grayclearimg    = normalize(cv2.imread(cimg_filepath + '/' + filename,0))

			dark_channel = guided_filter(grayclearimg, get_dark_channel(clearimg))#.transpose((1, 0))
			light_channel = guided_filter(grayclearimg, get_light_channel(clearimg))#.transpose((1, 0))
			
			clearimg   = clearimg.transpose(2, 0, 1)
			hazeimg    = hazeimg.transpose(2, 0, 1)
			lowimg     = lowimg.transpose(2, 0, 1)
			hazelowimg = hazelowimg.transpose(2, 0, 1)
			
			print(dark_channel.shape,clearimg.shape)
						
			clearimgs   = concatenateimgs(clearimg,hazeimg,lowimg,hazelowimg,dark_channel,light_channel,imgdepth)
			
			img_patches = img_to_patches(clearimgs, win=patch_size, stride=stride)
   
			for nx in range(img_patches.shape[3]):
				data = data_augmentation(img_patches[:, :, :, nx].copy(), 0)
				h5f.create_dataset(str(count), data=data)
				count += 1
			i += 1
		print(data.shape)
	h5f.close()	

def STrainGT(cimg_filepath, depth_filepath, patch_size, stride):
	'''synthetic ImageEdge images'''
	train_haze = 'Train_GT2LV_GT.h5'
	img_files = readfiles(cimg_filepath)
	count = 0
		
	with h5py.File(train_haze, 'w') as h5f:
		for i in range(len(img_files)):
			filename    = img_files[i]
			
			print(filename)
            
			clearimg     = normalize(cv2.imread(cimg_filepath + '/' + filename))
			depth        = normalize(cv2.imread(depth_filepath + '/' + filename,0))						
			grayclearimg = normalize(cv2.imread(cimg_filepath + '/' + filename,0))

			dark_channel = guided_filter(grayclearimg, get_dark_channel(clearimg))#.transpose((1, 0))
			light_channel = guided_filter(grayclearimg, get_light_channel(clearimg))#.transpose((1, 0))
			
			clearimg   = clearimg.transpose(2, 0, 1)
			
			print(dark_channel.shape,clearimg.shape)
						
			clearimgs   = concatenate1imgs(clearimg,dark_channel,light_channel,depth)
			
			img_patches = img_to_patches(clearimgs, win=patch_size, stride=stride)

   
			for nx in range(img_patches.shape[3]):
				data = data_augmentation(img_patches[:, :, :, nx].copy(), 0)
				h5f.create_dataset(str(count), data=data)
				count += 1
			i += 1
		print(data.shape)
	h5f.close()
