import torch
import numpy as np 
import torchvision
import matplotlib.pyplot as plt 
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import trange

from datautils import load_mnist, load_fashion_mnist


def load_data(name, pos_class, neg_class):
	subset_labels = (pos_class, neg_class)
	if name == "MNIST":
		train_loader, test_loader = load_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)	
	elif name == "FashionMNIST":
		train_loader, test_loader = load_fashion_mnist(batch_size=32, flatten=True, subset_labels=subset_labels)	
	else:
		raise ValueError(f"invalid name: {name}")
	return train_loader.dataset.tensors, test_loader.dataset.tensors

def create_model(**kwargs):
	# C = 1.0 / (self.num_train_examples * self.weight_decay)
	return LogisticRegression(
		# C=C,
		tol=1e-8,
		fit_intercept=False,
		solver="lbfgs",
		warm_start=True,
		max_iter=1000,
		**kwargs,
	)
	
def train_model(model, x_train, y_train, x_test, y_test):
	model.fit(x_train, y_train)

	y_pred = model.predict(x_train)
	train_acc = accuracy_score(y_train, y_pred) 

	y_pred = model.predict(x_test)
	test_acc = accuracy_score(y_test, y_pred)
	print(f"train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

def get_chunk_stats(y_train, indices, target_label, k=10):
	chunks = np.array_split(indices, k)
	props = []
	for chunk in chunks:
		labels = y_train[chunk].tolist()
		vals = Counter(labels)[target_label] / len(labels)
		props.append(vals)

	return props

def sigmoid(x): return 1 / (1 + torch.exp(-x))

def show_examples(examples, pos_class, neg_class, title, save_as):
	nrows = len(examples)
	ncols = len(examples[0][1]) + 1
	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, constrained_layout=True)
	fig.suptitle(title, fontsize=10)
	for row, (test_idx, train_idxs, ifs) in enumerate(examples):
		axes[row][0].imshow(x_test[test_idx].reshape(28,28), cmap="gray")

		class_label = pos_class if y_test[test_idx] == 0 else neg_class
		axes[row][0].set_title(f"Query: {class_label}", fontsize=6)
		
		axes[row][0].axis("off")
		# axes[row][0].set_xticks([])
		# axes[row][0].set_yticks([14], labels=[_label], fontsize=8)

		for i, train_idx in enumerate(train_idxs, start=1):
			axes[row][i].imshow(x_train[train_idx].reshape(28,28), cmap="gray")
			axes[row][i].axis("off")
			class_label = pos_class if y_train[train_idx] == 0 else neg_class
			axes[row][i].set_title(f"Label={class_label}\nIF={ifs[i-1]:.4f}", fontsize=6)

	plt.savefig(save_as)
	return axes

def show_chunk_stats(chunk_stats, save_as):
	n = len(chunk_stats[0][1:][0])
	idxs = len(chunk_stats)

	_, ax = plt.subplots()
	labels = ["best_pos", "worst_pos", "best_neg", "worst_neg"]
	for i, idx in enumerate(range(idxs)):
		ax.plot(np.arange(n), chunk_stats[idx][1:][0], label=labels[i])

	ax.legend()
	plt.savefig(save_as)
	return ax

from empiricaldist import Surv

def plot_left_tail(samples, ax=None):
	shift = torch.max(samples)
	left_tail = shift - samples
	surv = Surv.from_seq(left_tail)
	surv.replace(0, np.nan, inplace=True)

	if ax is not None:
		surv.plot(ax=ax)
	else:
		ax = surv.plot()
			
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel("Influence Function")
	ax.set_ylabel("CCDF (log)")

	#labels = np.array([-0.1, -1e-2, -1e-3])
	#locs = shift - labels
	#ax.set_xticks(locs, labels)

	return ax

def plot_right_tail(samples, ax=None):
	shift = torch.min(samples)
	right_tail = samples - shift
	surv = Surv.from_seq(right_tail)
	surv.replace(0, np.nan, inplace=True)
	
	if ax is not None:
		surv.plot(ax=ax)
	else:
		ax = surv.plot()
			
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel("Influence Function")
	ax.set_ylabel("CCDF (log)") 
	return ax


if __name__ == "__main__":
	"""
	# pos_class, neg_class = 0, 1
	# pos_class, neg_class = 1, 0

	# pos_class, neg_class = 1, 2
	pos_class, neg_class = 0, 8
	# pos_class, neg_class = 2, 3
	# pos_class, neg_class = 7, 4
	# pos_class, neg_class = 1, 7
	# pos_class, neg_class = 7, 8

	(x_train, y_train), (x_test, y_test) = load_data("MNIST", pos_class, neg_class)
	"""

	# pos_class = 3 # Dress
	# neg_class = 5 # Sandal
	pos_class = 5 # Sandal
	neg_class = 9 # Ankle boot
	(x_train, y_train), (x_test, y_test) = load_data("FashionMNIST", pos_class, neg_class)

	# transform labels
	y_train = torch.where(y_train==pos_class, 0., 1.)
	y_test = torch.where(y_test==pos_class, 0., 1.)

	model = create_model()
	train_model(model, x_train, y_train, x_test, y_test) 
	W = torch.tensor(model.coef_.T, dtype=torch.float)
	z = x_train @ W
	h = sigmoid(z)

	d_ii = (h*(1-h)).squeeze()
	D = torch.diag(d_ii)

	#y_target = torch.where(y_test==pos_class, 0., 1.)
	y_target = y_test
	y_pred = torch.tensor(model.predict_proba(x_test)[:,1]).to(torch.float)
	# yhat = torch.tensor(model.predict(x_test)).to(float)
	# yhat = torch.tensor([1. if y == neg_class else 0. for y in yhat])
	test_loss = torch.nn.functional.binary_cross_entropy(y_pred, y_target, reduction="none")

	# by label
	#_test_loss_pos_class = np.where(y_test==pos_class, test_loss, np.nan)
	_test_loss_pos_class = np.where(y_test==0, test_loss, np.nan)
	best_idx_pos = np.nanargmin(_test_loss_pos_class)
	worst_idx_pos = np.nanargmax(_test_loss_pos_class)
	print(best_idx_pos, worst_idx_pos)
	print(y_test[best_idx_pos], y_test[worst_idx_pos])
	print(test_loss[best_idx_pos], test_loss[worst_idx_pos])

	#_test_loss_neg_class = np.where(y_test==neg_class, test_loss, np.nan)
	_test_loss_neg_class = np.where(y_test==1, test_loss, np.nan)
	best_idx_neg  = np.nanargmin(_test_loss_neg_class)
	worst_idx_neg = np.nanargmax(_test_loss_neg_class)
	print(best_idx_neg, worst_idx_neg)
	print(y_test[best_idx_neg], y_test[worst_idx_neg])
	print(test_loss[best_idx_neg], test_loss[worst_idx_neg])

	# damps = 10 ** np.linspace(-5,2,8)
	damps = [1e-2]
	
	# for visualization
	harmful_examples = defaultdict(list)
	helpful_examples = defaultdict(list)
	chunk_stats = defaultdict(list)
	quantile_stats = defaultdict(list)

	# tail plot
	_, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,4))
	is_inverted = False

	for damp in damps:
		print(f"damp: {damp}")
		hess = (x_train.T @ D @ x_train) / len(x_train) + damp * torch.eye(len(W))
		inv_hess = torch.inverse(hess)

		eigvals = np.linalg.eigvalsh(hess.cpu().numpy())
		print(f"hessian min eigval {np.min(eigvals).item()}")
		print(f"hessian max eigval {np.max(eigvals).item()}")

		test_samples = {"best_pos": best_idx_pos, "worst_pos": worst_idx_pos,
										"best_neg": best_idx_neg, "worst_neg": worst_idx_neg}
		for test_idx_label, test_idx in test_samples.items():
		# test_loss_indices = torch.sort(test_loss).indices
		# for test_idx in [test_loss_indices[0], test_loss_indices[500], test_loss_indices[1000], test_loss_indices[-1]]:
		# for test_idx in list(test_loss_indices[:10]) + list(test_loss_indices[-10:]):
			xt = x_test[test_idx].reshape(1,-1)
			yt = y_test[test_idx] * 2 - 1

			influences = []
			for i in trange(len(x_train)):
				xtt = x_train[i].reshape(1,-1)
				ytt = y_train[i] * 2 - 1

				score = (yt * ytt * sigmoid(-yt * xt @ W) * sigmoid(-ytt * xtt @ W) * xt @ inv_hess @ xtt.T) / len(x_train)
				#score = yt * ytt * sigmoid(-yt * xt @ W) * sigmoid(-ytt * xt @ W) * xtt @ inv_hess @ xtt.T ## THIS IS WRONG BUT WORKS MUCH BETTER? (only for certain scenarios)
				influences.append(score) ## order = indices

			influences = torch.tensor(influences)
			values, indices = torch.sort(influences)

			## cdf
			normalized_values = values / (max(values) + 1e-12)
			plot_left_tail(normalized_values, ax=ax1); ax1.set_title("Left Tail") 
			if not is_inverted:
				ax1.invert_xaxis()
				is_inverted = True
			plot_right_tail(normalized_values, ax=ax2); ax2.set_title("Right Tail")

			"""
			# ccdf on influence values
			p = torch.linspace(0, 1, len(influences))
			_mask = values > 0 # only right tail
			#plt.loglog(values[_mask] / torch.max(values), (1-p)[_mask], label=test_idx)
			plt.loglog(values[_mask], (1-p)[_mask], label=test_idx)

			# ccdf on deltas 
			deltas = torch.tensor([values[i]-values[i-1] for i in range(1,len(values))])
			p = torch.linspace(0, 1, len(deltas))
			plt.loglog(deltas / max(deltas), 1-p, label=test_idx)
			"""

			harmful_examples[damp].append([test_idx, indices[:10].tolist(), values[:10].tolist()])							 # harmful
			helpful_examples[damp].append([test_idx, indices[-10:].tolist()[::-1], values[-10:].tolist()[::-1]]) # helpful

			# quantiles
			quantiles = torch.quantile(values, q=torch.linspace(0,1,steps=101)).tolist()
			quantile_stats[damp].append([test_idx, quantiles])

			# chunk stats
			target_label = y_test[test_idx].item()
			cstats = get_chunk_stats(y_train, indices, target_label, k=100) 
			chunk_stats[damp].append([test_idx, cstats])
 

	# same ccdf plot 
	ax1.legend(list(test_samples.keys()))
	ax2.legend(list(test_samples.keys()))
	plt.savefig(f"./experiment_results/tail_plot_FashionMNIST_{pos_class}_{neg_class}.png")

	_ = show_chunk_stats(chunk_stats[1e-2], save_as=f"./experiment_results/chunk_stats_FashionMNIST_{pos_class}_{neg_class}.png")
	_ = show_examples(harmful_examples[1e-2], pos_class, neg_class, title="FashionMNIST Harmful Samples", save_as=f"./experiment_results/examples_FashionMNIST_harmful_{pos_class}_{neg_class}.png")
	_ = show_examples(helpful_examples[1e-2], pos_class, neg_class, title="FashionMNIST Helpful Samples", save_as=f"./experiment_results/examples_FashionMNIST_helpful_{pos_class}_{neg_class}.png")

	# TODO: plot side by side
	plt.legend()
	plt.show()
