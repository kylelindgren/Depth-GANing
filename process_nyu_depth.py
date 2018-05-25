#!/usr/bin/env python
""" process_nyu_depth.py:
Script to extract imgs from the nyu dataset and format for compatibility with cyclegan.
"""

__author__ = "Kyle M Lindgren"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "May 24, 2018"

# import
import cv2
import numpy as np
from glob import glob
import os, errno


ds_dir = '/media/kylelindgren/Data/NYU_Depth_V2'
out_dir = '/media/kylelindgren/Data/nyu_processed'

def extract_sams():
	ds_dir_sub = sorted(glob(ds_dir + '/*'))
	ds_dir_num = int(len(ds_dir_sub))

	idx = 0
	for idx_folder in range(ds_dir_num):
		ds_fld = sorted(glob(ds_dir_sub[idx_folder] + '/*'))
		n_flds = int(len(ds_fld))

		for idx_sub_folder in range(n_flds):
			ds_sub_fld = sorted(glob(ds_fld[idx_sub_folder] + '/*.pgm'))
			n_imgs = int(len(ds_sub_fld))

			for i in range(n_imgs):
				idx_fld_name = int(idx / 1000)
				out_fld = out_dir + '/' + str(idx_fld_name) + 'K/'
				try:
					os.makedirs(out_fld)
				except OSError as e:
					if e.errno != errno.EEXIST:
						raise

				img = cv2.imread(ds_sub_fld[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
				if img is not None:
					img = (1*(img/256)).astype('uint8')
					img = cv2.resize(img[:, 80:560], dsize=(128, 128))
					# cv2.imshow('img', img)
					# cv2.waitKey(50)
					
					cv2.imwrite(out_fld + str(idx) + '.png', img)
					idx = idx + 1

if __name__ == '__main__':
	extract_sams()