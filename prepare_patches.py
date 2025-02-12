from makedataset import *
import argparse

if __name__ == "__main__":
		
	parser = argparse.ArgumentParser(description="Building the training patch database")
	
	parser.add_argument("--rgb", action='store_true',default = True, help='prepare RGB database instead of grayscale')
	parser.add_argument("--patch_size", "--p", type=int, default=256, help="Patch size")
	parser.add_argument("--stride", "--s", type=int, default=224, help="Size of stride")
	args = parser.parse_args()

	TrainGT('./clear/','./haze/','./low/','./low_haze/','./Depth/',args.patch_size,args.stride)
