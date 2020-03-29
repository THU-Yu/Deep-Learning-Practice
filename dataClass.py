import torch.utils.data as Data
import torch
import numpy as np

class mydataset(Data.Dataset):
	def __init__(self, tensor, label):
		l = len(tensor)
		data = []
		for i in range(l):
			data.append((tensor[i], label[i]))
		self.dataset = data
		
	def __getitem__(self, index):
		tensor, label = self.dataset[index]
		return tensor,label

	def __len__(self):
		return len(self.dataset)
