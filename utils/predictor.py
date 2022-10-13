import os
import numpy as np
import time
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from .dataset import *


class predictor:

	def __init__(self, device_name="cpu", model_name="vit_base_patch16_224", log_dir=None):
		self.device = torch.device(device_name)
		self.model = timm.create_model(model_name, pretrained=True)
		self.model.to(self.device)
		self.model.eval()
		config = resolve_data_config({}, model=self.model)
		self.transform = create_transform(**config)
		print("Model loaded on " + device_name)

		if log_dir != None:
			self.writer = SummaryWriter(log_dir=log_dir)
		else:
			self.writer = None

		self.cat_ids = []
		self.cat_labels = []
		self.cat_names = []
		self.cat_list = {}
		with open("/home/hxy/proj-grad/utils/imagenet_ids.txt", "r") as f:
			categories = [s.strip() for s in f.readlines()]
		for k in range(len(categories)):
			cat, label, name, _ = categories[k].split()
			self.cat_ids.append(int(cat))
			self.cat_labels.append(label)
			self.cat_names.append(name)
			self.cat_list[label] = int(cat)
		f.close()


	def predict_class(self, class_dir, class_id, zero_lines_list=[[]], batch_size=50, log=False):
		dataset = myDataset(class_dir, self.transform)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
		corrects = []

		for i in range(len(zero_lines_list)):
			correct = 0
			for inputs in dataloader:
				inputs = inputs.to(self.device)
				with torch.no_grad():
					out, attns, xs = self.model(inputs, zero_lines = zero_lines_list[i])
				probabilities = torch.nn.functional.softmax(out, dim = 1)
				top_prob, top_catid = torch.topk(probabilities, 1)
				correct += np.sum(np.array(top_catid.cpu()).reshape((batch_size)) == class_id)

			corrects.append(correct)
			if self.writer != None and log == True:
				self.writer.add_scalar(str(class_id), correct, i)

		return corrects

	def predict_dir(self, imgs_dir):
		dataset = myDataset(imgs_dir, self.transform)
		dataloader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=8, pin_memory=True)

		for inputs in dataloader:
			inputs = inputs.to(self.device)
			with torch.no_grad():
				out, attns, xs = self.model(inputs)
			probabilities = torch.nn.functional.softmax(out, dim = 1)
			top_prob, top_catid = torch.topk(probabilities, 1)

		return attns, np.array(top_catid.cpu()).reshape((dataset.__len__()))


	def predict_one(self, img_path, zero_lines=[], tensor=None):
		if tensor == None:
			img = Image.open(img_path).convert('RGB')
			tensor = self.transform(img).unsqueeze(0).to(self.device)
		with torch.no_grad():
			out, attns, xs = self.model(tensor, zero_lines = zero_lines)
			# out = self.model(tensor)#, zero_lines_list = zero_lines)
		probabilities = torch.nn.functional.softmax(out, dim = 1)
		top_prob, top_catid = torch.topk(probabilities, 1)

		return top_prob.item(), top_catid.item(), attns, xs

	def predict_one_CNN(self, img_path, top_k = [], tensor=None):
		if tensor == None:
			img = Image.open(img_path).convert('RGB')
			tensor = self.transform(img).unsqueeze(0).to(self.device)
		with torch.no_grad():
			out, x_maps = self.model(tensor, top_k)
		probabilities = torch.nn.functional.softmax(out, dim = 1)
		top_prob, top_catid = torch.topk(probabilities, 1)

		return top_prob.item(), top_catid.item(), x_maps

	def predict_one_Swin(self, img_path, top_k = [], tensor=None):
		if tensor == None:
			img = Image.open(img_path).convert('RGB')
			tensor = self.transform(img).unsqueeze(0).to(self.device)
		with torch.no_grad():
			out, attns = self.model(tensor, top_k)
		probabilities = torch.nn.functional.softmax(out, dim = 1)
		top_prob, top_catid = torch.topk(probabilities, 1)

		return top_prob.item(), top_catid.item(), attns


	def predict_val(self, val_path, zero_lines_list=[[]], log_title="Val", batch_size=50, num_classes=1000):
		dataset = myDataset(val_path, self.transform, multi_cats = True, cats = num_classes) # , pre_load = True)
		# dataset = ImageNet_val(val_path[0], val_path[1])
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
		# print("Data Loaded!")
		print("Image number: ", dataset.__len__())

		corrects = []
		for i in tqdm(range(len(zero_lines_list))):
			correct = 0
			k_classes = 0
			for inputs, labels in tqdm(dataloader, leave = False):
				inputs = inputs.to(self.device)
				with torch.no_grad():
					#out, attns, xs = self.model(inputs, zero_lines = zero_lines_list[i])
					out = self.model(inputs, zero_lines_list = zero_lines_list[i])
				probabilities = torch.nn.functional.softmax(out, dim = 1)
				top_prob, top_catid = torch.topk(probabilities, 1)
				for t in range(len(top_catid)):
					if self.cat_labels[top_catid[t].item()] == labels[t]:
						correct += 1

				# k_classes += 1
				# if k_classes >= num_classes:
				# 	break

			corrects.append(correct)
			if self.writer != None:
				self.writer.add_scalar(log_title, correct, i)

		return corrects