import cv2 as cv
import os
import re
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='source image path')
parser.add_argument('--load_mat_file', dest='load_mat_file', help='MAT file name', type=str, default='kernel_blur MAT file')
parser.add_argument('--fold_AB', dest='fold_AB', help='input directory for image integrate', type=str, default='all data path')

args = parser.parse_args()
split = os.listdir(args.fold_A)
split = sorted(split)
kernel_blur = os.listdir(args.load_mat_file)
kernel_blur.sort()
kernel_path_list = []
for k in kernel_blur:
	kernel_path = os.path.join(args.load_mat_file, k)
	kernel_path_list.append(kernel_path)

for n,sp in enumerate(split):
	src_path = os.path.join(args.fold_A, sp)
	src_img = cv.imread(src_path, -1)
	if os.path.exists(kernel_path_list[n]):
		from scipy.io import loadmat
		kernel_blur = loadmat(kernel_path_list[n])
		kernel_blur = kernel_blur.get('kernel_blur')
		kernel_blur = np.transpose(kernel_blur, (2,0,1))
	else:
		print('%d file not exists'%(kernel_path_list[n]))
	mat_file_count = kernel_path_list[n]
	mat_file_count = re.split(r'/',mat_file_count)[-1]
	mat_file_count = re.split(r'\.',mat_file_count)[0]
	for i in range(kernel_blur.shape[0]):
		kernelBlur = kernel_blur[i]
		dst = cv.filter2D(src_img, -1, kernel=kernelBlur)
		first_name = re.split(r'\.', sp)[0]
		end_name = first_name + 'blur' + mat_file_count + '_' + str(i) + '.bmp'
		end_name = os.path.join(args.fold_AB, end_name)
		cv.imwrite(end_name, dst)

