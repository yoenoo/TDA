import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader

def _load_mnist(name, root="./data", train=True, normalize=True, flatten=False, subset_labels=None, **kwargs):
	if name == "MNIST":
		mnist = datasets.MNIST(root=root, train=train, download=True, transform=ToTensor())
	elif name == "FashionMNIST":
		mnist = datasets.FashionMNIST(root=root, train=train, download=True, transform=ToTensor())
	else:
		raise ValueError(f"invalid name: {name}")

	images, labels = mnist.data, mnist.targets

	if labels is not None:
		idx = torch.isin(labels, torch.tensor(subset_labels))
		images = images[idx]
		labels = labels[idx]

	if normalize:
		max_pixel = torch.max(images)
		images = images.to(dtype=torch.float32) / max_pixel

	if flatten:
		bs = len(images)
		images = images.reshape(bs, -1)

	dataset = TensorDataset(images, labels)
	dataloader = DataLoader(dataset, **kwargs)
	return dataloader 

def load_mnist(batch_size: int, flatten=False, subset_labels=None):
	train_loader = _load_mnist(name="MNIST", train=True, shuffle=True, flatten=flatten, 
														 batch_size=batch_size, subset_labels=subset_labels)
	test_loader  = _load_mnist(name="MNIST", train=False, shuffle=False, flatten=flatten, 
														 batch_size=batch_size, subset_labels=subset_labels)
	return train_loader, test_loader 


def load_fashion_mnist(batch_size: int, flatten=False, subset_labels=None): 
	train_loader = _load_mnist(name="FashionMNIST", train=True, shuffle=True, flatten=flatten, 
														 batch_size=batch_size, subset_labels=subset_labels)
	test_loader  = _load_mnist(name="FashionMNIST", train=False, shuffle=False, flatten=flatten, 
														 batch_size=batch_size, subset_labels=subset_labels)
	return train_loader, test_loader 
