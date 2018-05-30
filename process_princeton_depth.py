#!/usr/bin/env python
""" process_princeton_depth.py:
Script to process depth imgs from the princeton dataset and convert them from 16 to 8 bit.
raw depth files 16 bit images with values in mm. Capped at 4m to match Kinect v1 (NYU dataset)
also resized to be square and same size (128, 128) as images used in CycleGAN demo.
"""

__author__ = "Kyle M Lindgren"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "May 23, 2018"

# import
import cv2
import numpy as np
from glob import glob
import os, errno


ds_dir = '/media/kylelindgren/Data/pbrs_princeton_depth'
#out_dir = '/media/kylelindgren/Data/princeton_processed_128'
out_dir = '/media/kylelindgren/Data/princeton_processed_480'

def extract_sams():
	ds_dir_sub = sorted(glob(ds_dir + '/*'))
	ds_dir_num = int(len(ds_dir_sub))

	idx = 0
	for idx_folder in range(ds_dir_num):
	# for idx_folder in range(5):
		folder_imgs = sorted(glob(ds_dir_sub[idx_folder] + '/*'))
		n_imgs = int(len(folder_imgs))

		for i in range(n_imgs):
			idx_fld_name = int(idx / 1000)
			out_fld = out_dir + '/' + str(idx_fld_name) + 'K/'
			try:
				os.makedirs(out_fld)
			except OSError as e:
				if e.errno != errno.EEXIST:
					raise

			img = cv2.imread(folder_imgs[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

			img[img > 4000] = 0  # out of range = black in NYU, 4000mm -> kinect range ~ 4m
			img = (255*(img/4000)).astype('uint8')

			img = img[:, 80:560]
			#img = cv2.resize(img, dsize=(128, 128))
			
			cv2.imwrite(out_fld + str(idx) + '.png', img)
			idx = idx + 1

if __name__ == '__main__':
	extract_sams()