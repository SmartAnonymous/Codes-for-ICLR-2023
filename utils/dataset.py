import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

class myDataset(Dataset):
	def __init__(self, path, transform, multi_cats=False, pre_load = False, cats = 1000):
		self.transform = transform
		self.multi_cats = multi_cats
		self.pre_load = pre_load

		if self.multi_cats == False:
			imgs_path = sorted(os.listdir(path))
			self.paths = []
			self.imgs = []
			for i in range(len(imgs_path)):
				if self.pre_load == False:
					self.paths.append(path + "/" + imgs_path[i])
				else:
					self.imgs.append(self.load_img(path + "/" + imgs_path[i]))
		else:
			cat_labels = sorted(os.listdir(path))
			self.paths = []
			self.imgs = []
			self.labels = []
			for l in tqdm(range(cats)):
				cat_path = path + cat_labels[int(1000 / cats) * l]
				imgs_path = sorted(os.listdir(cat_path))
				for i in range(len(imgs_path)):
					if self.pre_load == False:
						self.paths.append(cat_path + "/" + imgs_path[i])
					else:
						self.imgs.append(self.load_img(cat_path + "/" + imgs_path[i]))
					self.labels.append(cat_labels[int(1000 / cats) * l])
		if pre_load == True:
			print("Data pre-loaded!")

	def __getitem__(self, index):
		if self.pre_load == False:
			if self.multi_cats == False:
				return self.load_img(self.paths[index])
			else:
				return self.load_img(self.paths[index]), self.labels[index]
		else:
			if self.multi_cats == False:
				return self.imgs[index]
			else:
				return self.imgs[index], self.labels[index]


	def __len__(self):
		if self.pre_load == False:
			return len(self.paths)
		else:
			return len(self.imgs)

	def load_img(self, path):
		return self.transform(Image.open(path).convert('RGB'))

class ImageNet_val(Dataset):
	def __init__(self, imgs_dir, cats_dir):
		self.imgs = np.load(imgs_dir)
		self.cats = np.load(cats_dir)
		assert self.imgs.shape[0] == self.cats.shape[0]

	def __getitem__(self, index):
		return self.imgs[index], self.cats[index]

	def __len__(self):
		return self.imgs.shape[0]