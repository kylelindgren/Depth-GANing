#!/usr/bin/env python
""" preprocess_img_size.py:
Set image size to match NN size, without distorting.
"""

__author__ = "Kyle M Lindgren"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "May 28, 2018"


# import
import numpy as np
import cv2
from glob import glob


# ds_dir = '/home/kylelindgren/Documents/ee595/project/middlebury/test/'
ds_dir = '/home/kylelindgren/Documents/ee595/project/middlebury/noised/'
out_dir = '/home/kylelindgren/Documents/ee595/project/middlebury/processed/'

if __name__ == '__main__':
	img_size = 480
	imgs = sorted(glob(ds_dir + '*'))
	n_imgs = int(len(imgs))

	for i in range(n_imgs):
		img_new = np.zeros((img_size, img_size), dtype='uint8')

		img = cv2.imread(imgs[i], cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
		if(len(img.shape) > 2):
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_r, img_c = img.shape

		off_r, off_c = 0, 0
		if img_r < img_size:
			off_r = (img_size - img_r) / 2
		if img_c < img_size:
			off_c = (img_size - img_c) / 2
		img_new[off_r:(img.shape[0]+off_r), off_c:(img.shape[1]+off_c)] = img

		cv2.imshow('img', img_new)
		cv2.waitKey(1000)
		
		cv2.imwrite(out_dir + str(i) + '.png', img_new)

