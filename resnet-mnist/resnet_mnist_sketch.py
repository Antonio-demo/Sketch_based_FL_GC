import time
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import csv
from Fed import countsketch_2, rand_hashing

"""
import wandb

wandb.init(project="my-tran-project", entity="zhangpan")
wandb.config = {
	"learning_rate": 0.001,
	"epochs": 100,
	"batch_size": 128
}

"""



# train process
print("\nLOAD DATA\n")

DEVICE = torch.device("cuda"if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor())

##################################################### Using a GPU #####################################################

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

print("\nUSING", device)
if torch.cuda.is_available():
	num_dev = torch.cuda.current_device()
	print(torch.cuda.get_device_name(num_dev), "\n")

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
	                 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1
	
	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
	
	def forward(self, x):
		residual = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			residual = self.downsample(x)
		
		out += residual
		out = self.relu(out)
		
		return out


class ResNet(nn.Module):
	
	def __init__(self, block, layers, num_classes, grayscale):
		self.inplanes = 64
		if grayscale:
			in_dim = 1
		else:
			in_dim = 3
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
		                       bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)
	
	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:  # 看上面的信息是否需要卷积修改，从而满足相加条件
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)
		
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		
		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		# because MNIST is already 1x1 here:
		# disable avg pooling
		# x = self.avgpool(x)
		
		x = x.view(x.size(0), -1)
		logits = self.fc(x)
		
		probas = F.softmax(logits, dim=1)
		return logits, probas


def resnet18(num_classes):
	"""Constructs a ResNet-18 model."""
	model = ResNet(block=BasicBlock,
	               layers=[2, 2, 2, 2],
	               num_classes=num_classes,
	               grayscale=True)
	return model


net = resnet18(10)

NUM_EPOCHS = 100
model = resnet18(num_classes=10)
model = model.to(DEVICE)

# 原先这里选用SGD训练，但是效果很差，换成Adam优化就好了
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
valid_loader = test_loader


def compute_accuracy_and_loss(model, data_loader, device):
	correct_pred, num_examples = 0, 0
	cross_entropy = 0.
	for i, (features, targets) in enumerate(data_loader):
		features = features.to(device)
		targets = targets.to(device)
		logits, probas = model(features)
		cross_entropy += F.cross_entropy(logits, targets, reduction='sum').item()
		_, predicted_labels = torch.max(probas, 1)
		num_examples += targets.size(0)
		correct_pred += (predicted_labels == targets).sum().item()
	# print(correct_pred)
	return correct_pred / num_examples * 100, cross_entropy / num_examples



train_acc_list, valid_acc_list = [], []
train_loss_list, valid_loss_list = [], []


model.train()
# copy weights
w_global = model.state_dict()
# print(w_glob)

print(w_global.keys())
print(len(w_global.keys()))

count = 0

for weight in w_global.keys():
	name = w_global[f'{weight}']
	print(name.shape)
	num_params = name.numel()
	count += num_params

print(count)


start_sketch_time = time.time()

conv1_weight = w_global['conv1.weight'].reshape(64,49)

#print(conv1_weight.shape)#64*1*7*7

hash_idx, rand_sgn = rand_hashing(49, 2)
countsketch_x = countsketch_2(conv1_weight, hash_idx, rand_sgn)
#print(countsketch_x.shape)#64*24


b = torch.zeros((64,25)).to(DEVICE)
#print(b.shape)#64*1*4*7
conuntsketch_cat_x = torch.cat((countsketch_x,b),1)

count_x = conuntsketch_cat_x.reshape(64,1,7,7)
#print(cat_conuntsketch_x.shape)#64*49

w_global['conv1.weight'] = count_x

end_sketch_time = time.time()

sketch_compute_time = end_sketch_time - start_sketch_time

print(f"sketch compute time:{sketch_compute_time} s")


# start_time = time.time()
# for epoch in range(NUM_EPOCHS):
#
# 	for batch_idx, (features, targets) in enumerate(train_loader):
# 		### PREPARE MINIBATCH
# 		features = features.to(DEVICE)
# 		targets = targets.to(DEVICE)
# 		### FORWARD AND BACK PROP
# 		logits, probas = model(features)
# 		cost = F.cross_entropy(logits, targets)
# 		optimizer.zero_grad()
# 		cost.backward()
# 		### UPDATE MODEL PARAMETERS
# 		optimizer.step()
#
# 		### LOGGING
# 		if batch_idx % 50 == 0:
# 			acc = (logits.argmax(1) == targets).float().mean().item()
# 			print("Train Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
# 				epoch + 1, NUM_EPOCHS, batch_idx, len(train_loader), acc, cost.item()))
#
# 	elapsed_1 = (time.time() - start_time)
# 	print(f'Training Time (minutes)=: {elapsed_1 / 60:.4f} min')
#
#
# 	# no need to build the computation graph for backprop when computing accuracy
# 	model.eval()
#
# 	valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
# 	print('Test Epochs[{}/{}]--Loss on Test {:.4}--Accuracy on Test {:.4}'.
# 	      format(epoch + 1, NUM_EPOCHS, valid_loss, valid_acc))
# 	elapsed_2 = (time.time() - start_time - elapsed_1) / 60
#
#
# 	print(f'Predit Time (minutes)=: {elapsed_2:.4f} min')
#
# 	wandb.log({"Train loss": cost.item(), 'epoch': epoch + 1})
# 	wandb.log({"Test loss": valid_loss, 'epoch': epoch + 1})
# 	wandb.log({"Train acc": acc, 'epoch': epoch + 1})
# 	wandb.log({"Test acc": valid_acc, 'epoch': epoch + 1})
# 	wandb.log({"Train time": elapsed_1 / 60 , 'epoch': epoch + 1})
# 	wandb.log({"Predit Time": elapsed_2 , 'epoch': epoch + 1})
#
#
# 	# Optional
# 	#wandb.watch(model)
# 	#model.load_state_dict(w_glob)
#
# elapsed = (time.time() - start_time) / 60
# print(f'Total Training Time: {elapsed:.2f} min')













