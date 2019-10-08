import argparse
import torch as t
import torch
import torch.nn as nn
import numpy as np
import re
import torchvision as tv
from torch.autograd import Variable
import torch.nn.functional as F
import os
from PIL import Image
import cv2
from collections import OrderedDict
import time


class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--data_path', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
		self.parser.add_argument('--validata_path', required=True, help='path to validate images (should have subfolders trainA, trainB, valA, valB, etc)')
		self.parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
		self.parser.add_argument('--num_workers', default=2, type=int, help='# threads for loading data')
		self.parser.add_argument('--image_size', type=int, default=225, help='then crop to this size')
		self.parser.add_argument('--max_epoch', type=int, default=200, help='# epoch count')
		self.parser.add_argument('--lr1', type=int, default=0.00001, help='# learn rate')
		self.parser.add_argument('--beta1', type=int, default=0.5, help='# adam optimize beta1 parameter')
		self.parser.add_argument('--gpu', action='store_true', default=False, help='# use pgu')
		self.parser.add_argument('--vis', default=True, help='# wheather to use visdom visulizer')
		self.parser.add_argument('--env', type=str, default='net', help='# use gpu train')
		self.parser.add_argument('--plot_every', type=int, default=10, help='# print error')
		self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
		self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
		self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
		self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
		self.parser.add_argument('--model', type=str, default='Train', help='chooses which model to use. test or train')	
		self.parser.add_argument('--load_model', type=str, default=None, help='# train or test load .pth file')
	
	def parse(self):
                if not self.initialized:
                        self.initialize()
                self.opt = self.parser.parse_args()
                args = vars(self.opt)

                print('------------ Options -------------')
                for k, v in sorted(args.items()):
                        print('%s: %s' % (str(k), str(v)))
                print('-------------- End ----------------')
                return self.opt


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class MyDataset(torch.utils.data.Dataset):
	def __init__(self, validata=False):
		self.opt = opt
		if  validata:
			self.root = opt.validata_path
			self.dir_A = os.path.join(opt.validata_path)
		else:
			self.root = opt.data_path
			self.dir_A = os.path.join(opt.data_path)
		self.A_paths = make_dataset(self.dir_A)

		self.A_paths = sorted(self.A_paths)

		self.transform = tv.transforms.Compose([
						tv.transforms.Resize(opt.image_size),
						tv.transforms.CenterCrop(opt.image_size),
						tv.transforms.ToTensor(),
						tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
						])

	def __getitem__(self, index):
		A_path = self.A_paths[index]

		A_img = Image.open(A_path).convert('L')

		A_img = self.transform(A_img)

		all_name = re.split(r'/', A_path)[-1]	
		target = re.split(r'blur', all_name)[-1]	
		target = re.split(r'\.', target)[0]
		
		return A_img, target

	def __len__(self):
		return len(self.A_paths)	


class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            if in_channels != out_channels:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                        nn.BatchNorm2d(out_channels)
                        )
            else:
                self.downsample = None

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out
 
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.first = nn.Sequential(
		nn.Conv2d(1, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1)
                )
        self.layer1 = self.make_layer(64, 64, 3, 1) 
        self.layer2 = self.make_layer(64, 128, 4, 2) 
        self.layer3 = self.make_layer(128, 256, 6, 2) 
        self.layer4 = self.make_layer(256, 512, 3, 2) 
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, block_num, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x

 
class Visualizer():
        def __init__(self, opt):
                self.display_id = opt.display_id
                self.win_size = opt.display_winsize
                self.name = 'experiment_name'
                if self.display_id:
                        import visdom
                        self.vis = visdom.Visdom(env=opt.env, port=opt.display_port)
                        self.display_single_pane_ncols = opt.display_single_pane_ncols

        def plot_current_errors(self, epoch, count_ratio, opt, errors):
                if not hasattr(self, 'plot_data'):
                        self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
                self.plot_data['X'].append(epoch + count_ratio)
                self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
                self.vis.line(
                        X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
                        Y=np.array(self.plot_data['Y']),
                        opts={
                                'title': self.name + ' loss over time',
                                'legend': self.plot_data['legend'],
                                'xlabel': 'epoch',
                                'ylabel': 'loss'},
                        win=self.display_id)

        def print_current_errors(self, epoch, i, errors, t):
                message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
                for k, v in errors.items():
                        message += '%s: %.8f ' % (k, v)
                print(message)


def train():
	if opt.vis:
		vis  = Visualizer(opt)
	dataset = MyDataset()
	dataloader = t.utils.data.DataLoader(dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.num_workers,
                                         drop_last=True
                                         )
	net = Net(1521)
	if opt.load_model:
		map_location = lambda storage, loc: storage
		net.load_state_dict(t.load(opt.load_model, map_location=map_location))
	optimizer = t.optim.Adam(net.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
	criterion = t.nn.MSELoss()
	old_lr = opt.lr1

	if opt.gpu:
		net.cuda()
		criterion.cuda()

	for epoch in range(1,opt.max_epoch+1):
		epoch_iter = 0
		for ii,(img, labels)  in enumerate(dataloader):
			iter_start_time = time.time()
			epoch_iter += opt.batch_size
			inputs = Variable(img)
			optimizer.zero_grad()
			outputs = net(inputs.cuda())
			target = np.zeros((opt.batch_size, 1521))
			for l in range(0,opt.batch_size):
				kernel_count = labels[l]
				mat = re.split(r'_', kernel_count)[0]
				kernel_blur_number = re.split(r'_', kernel_count)[1]
				from scipy.io import loadmat
				kernel_blur = loadmat('./kernel_blur/%d.mat'%(int(mat)))
				kernel_blur = kernel_blur.get('kernel_blur')
				kernel_blur = np.transpose(kernel_blur, (2,0,1))
				kernelBlur = kernel_blur[int(kernel_blur_number)]				
				tmp = kernelBlur.reshape((1,1521), order='C')
				target[l] = tmp
			target = torch.Tensor(target)
			pre_outputs = Variable(target).cuda()
			loss = torch.sqrt(criterion(outputs, pre_outputs))
			loss.backward()
			optimizer.step()
			
			if opt.vis and (ii+1)%opt.plot_every == 0:
				errors = get_current_errors(loss)
				ti = (time.time() - iter_start_time) / opt.batch_size
				vis.print_current_errors(epoch, epoch_iter, errors, ti)
			if opt.display_id > 0 and (ii+1)%100 == 0:
				load_rate = float(epoch_iter)/dataset.__len__()
				vis.plot_current_errors(epoch, load_rate, opt, errors)	
		
		
		if epoch%1 == 0:
			t.save(net.state_dict(), 'checkpoints/net_%s.pth' % str(epoch))

			net_validata = Net(1521).cuda()
			map_location = lambda storage, loc: storage
			pth = 'checkpoints/net_%s.pth' % str(epoch)
			net_validata.load_state_dict(t.load(pth, map_location=map_location))
			dataset_validata = MyDataset(True)
			dataloader_validata = t.utils.data.DataLoader(dataset_validata,
                                         	batch_size=100,
                                         	shuffle=True,
                                         	num_workers=opt.num_workers,
                                         	drop_last=True
                                         	)
			for ii,(img, labels)  in enumerate(dataloader_validata):
				inputs_validata = Variable(img)
				outputs_validata = net_validata(inputs_validata.cuda())
				target_validata = np.zeros((100, 1521))
				for l in range(0, 100):
					validata_kernel_count = labels[l]
					validata_mat = re.split(r'_', validata_kernel_count)[0]
					validata_kernel_blur_number = re.split(r'_', validata_kernel_count)[1]
					from scipy.io import loadmat
					validata_kernel_blur = loadmat('./kernel_blur/%d.mat'%(int(validata_mat)))
					validata_kernel_blur = validata_kernel_blur.get('kernel_blur')
					validata_kernel_blur = np.transpose(validata_kernel_blur, (2,0,1))
					validata_kernelBlur = validata_kernel_blur[int(validata_kernel_blur_number)]
					tmp_validata = validata_kernelBlur.reshape((1,1521), order='C')
					target_validata[l] = tmp_validata
				target_validata = torch.Tensor(target_validata)
				pre_outputs_validata = Variable(target_validata).cuda()
				loss_validata = torch.sqrt(criterion(outputs_validata, pre_outputs_validata))
				print('==================================================')
				print('\033[1;35m epoch:%d Validate_dataset_loss:%.10f \033[0m!'%(epoch, loss_validata))
				print('==================================================')
				with open('validata.txt','a') as f: 
					vdl = 'epoch:%d Validate_dataset_loss:%.10f'%(epoch, loss_validata)
					f.write(vdl + '\n')
					f.close()
				break

		if epoch > 100:
			lrd = opt.lr1 / 100
			lr = old_lr - lrd
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
			print('update learning rate: %f -> %f' % (old_lr, lr))
			old_lr = lr


def test():
	net = Net(1521).eval()
	map_location = lambda storage, loc: storage	
	net.load_state_dict(t.load(opt.load_model, map_location=map_location))
	dataset = MyDataset()
	criterion = t.nn.MSELoss()
	dataloader = t.utils.data.DataLoader(dataset,
					batch_size=1,
					shuffle=False,
					num_workers=1,
					drop_last=False
					)
	for ii, (img, labels) in enumerate(dataloader):
		inputs = Variable(img)
		output = net(inputs)
		real_kernel = labels[0]
		real_mat = re.split(r'_', real_kernel)[0]
		real_number = re.split(r'_', real_kernel)[1]
		from scipy.io import loadmat
		real_kernel_blur = loadmat('./kernel_blur/%d.mat'%(int(real_mat)))
		real_kernel_blur = real_kernel_blur.get('kernel_blur')
		real_kernel_blur = np.transpose(real_kernel_blur, (2,0,1))
		real_kernelBlur = real_kernel_blur[int(real_number)]
		kernelBlur = output.data
		kernelBlur = kernelBlur.resize_(39,39)
		kernelBlur = kernelBlur.numpy()
		import scipy.io as io
		io.savemat('./test_kernel_blur/real_kernel' + str(labels[0]),{'name': real_kernelBlur})
		io.savemat('./test_kernel_blur/generate_kernel' + str(labels[0]),{'name': kernelBlur})

def gaussian_kernel_2d(kernel_size, sigma):
	kx = cv2.getGaussianKernel(kernel_size, sigma)
	ky = cv2.getGaussianKernel(kernel_size, sigma)
	return np.multiply(kx, np.transpose(ky))

def get_current_errors(loss):
	return OrderedDict([('NetLoss', loss.data)])

opt = BaseOptions().parse()
if opt.model == 'Train':
        train()
else:
        test()











































































































