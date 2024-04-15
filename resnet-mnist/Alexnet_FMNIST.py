import csv
import math
import torch.nn as nn
import torch
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
"""
import wandb
wandb.init(project="my-tran-project", entity="zhangpan")


wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 128
}

"""

class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		#这里全连接层的输出个数比LeNet中的大数倍，使用丢弃层Dropout来缓解过拟合
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=256 * 5 * 5, out_features=4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(in_features=4096, out_features=4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			#输出层。由于这里使用Fashion-MNIST,所以类别数为10,
			nn.Linear(in_features=4096, out_features=10),
		)

	def forward(self, img, labels=None):
		feature = self.conv(img)
		logits = self.fc(feature)
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss(reduction='mean')
			loss = loss_fct(logits, labels)
			return loss, logits
		else:
			return logits

def load_dataset(batch_size, resize=None):
	#加载数据集
	trans = []
	if resize:
		#将输入的28*28的图片，resize成224*224的形状
		trans.append(torchvision.transforms.Resize(size=resize, interpolation=InterpolationMode.BICUBIC))
	trans.append(torchvision.transforms.ToTensor())
	transform = torchvision.transforms.Compose(trans)
	#生成训练集和测试集
	mnist_train = torchvision.datasets.FashionMNIST(root='data',train=True, download=True,
	                                                transform=transform)
	mnist_test = torchvision.datasets.FashionMNIST(root='data', train=False, download=True,
	                                               transform=transform)
	train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True,num_workers=1)
	test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=1)
	return mnist_test, train_loader, test_loader


class MyModel:
	def __init__(self, batch_size=128, epochs=100, learning_rate=0.001):
		self.batch_size = batch_size
		self.epochs = epochs
		self.lr = learning_rate
		self.model = AlexNet()

	def train(self):
		cuda = torch.cuda.is_available()
		device = torch.device('cuda' if cuda else 'cpu')

		print("\nUSING", device)
		if cuda:
			num_dev = torch.cuda.current_device()
			print(torch.cuda.get_device_name(num_dev), "\n")

		mnist_test, train_loader, test_loader = load_dataset(batch_size=self.batch_size, resize=224)
		#num_training_steps = len(train_loader) * self.epochs
		optimizer = torch.optim.Adam([{"params":self.model.parameters(),
		                               "initial_lr":self.lr}])
		scheduler = CosineAnnealingLR(optimizer, T_max=20)
		#scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=300, num_training_steps=num_training_steps,
		                                            #num_cycles=2, last_epoch=self.epochs)

		self.model.to(device)

		#writer = SummaryWriter(log_dir='runs/result_3')
		f1 = open('./tables/alexnet_fmint_train_loss_acc.csv', 'w')
		csv_writer1 = csv.writer(f1)

		f2 = open('./tables/alexnet_fmint_test_loss_acc.csv', 'w')
		csv_writer2 = csv.writer(f2)


		for epoch in range(self.epochs):
			for i, (inputs, labels) in enumerate(train_loader):
				inputs, labels = inputs.to(device), labels.to(device)
				loss, logits = self.model(inputs, labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if i % 50 == 0:
					accuracy = (logits.argmax(1)==labels).float().mean().item()
					print("Train Epochs[{}/{}]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
						epoch + 1, self.epochs, i, len(train_loader), accuracy, loss.item()))
					#writer.add_scalar('Training/Accuracy', accuracy, epoch+1)
					#writer.add_scalar('Training/Loss', loss.item(), epoch+1)
					#line1 = [epoch+1,loss.item(),accuracy]
					#csv_writer1.writerow(line1)

			scheduler.step()
			#writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], epoch+1)
			#self.model.eval()#切换到评价模式
			lr = scheduler.get_last_lr()[0]
			test_acc, test_loss = self.test_img(self.model, mnist_test, self.batch_size, device)
			#test_acc = self.evaluate(test_loader, self.model, device)
			print("Test Epochs[{}/{}]--Loss on Test {:.4}--Accuracy on Test {:.4}".format(epoch+1, self.epochs,
			                                        test_loss,test_acc))
			#line2 = [epoch+1, test_loss, test_acc]
			#csv_writer2.writerow(line2)
			#wandb.log({"Train loss": loss.item(), 'epoch':epoch})
			#wandb.log({"Test loss": test_loss, 'epoch':epoch})
			#wandb.log({"Learning Rate": lr, 'epoch':epoch})
			#wandb.log({"Train acc": accuracy, 'epoch':epoch})
			#wandb.log({"Test acc": test_acc, 'epoch':epoch})

		f1.close()
		f2.close()



	@staticmethod
	def evaluate(data_loader, model, device):
		with torch.no_grad():
			acc_sum, n = 0.0, 0.0
			for x,y in data_loader:
				x, y = x.to(device), y.to(device)
				logits = model(x)
				acc_sum += (logits.argmax(1)==y).float().sum().item()
				n += len(y)
			return acc_sum / n

	@staticmethod
	def test_img(net, dataset, batch_size, device):
		net.eval()
		test_loss = 0
		correct = 0
		data_loader = DataLoader(dataset, batch_size)
		for idx, (data, target) in enumerate(data_loader):
			data, target = data.to(device), target.to(device)
			log_probs = net(data)
			#sum up batch loss
			test_loss += F.cross_entropy(log_probs, target,reduction='sum').item()
			#get the index of the max log-probability
			y_pred = log_probs.data.max(1, keepdim=True)[1]
			correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum().item()

		test_loss /= len(data_loader.dataset)
		accuracy = 100.00 * correct / len(data_loader.dataset)

		#print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
				#test_loss, correct, len(data_loader.dataset), accuracy))
		return accuracy, test_loss



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5,last_epoch=-1):
	def lr_lambda(current_step):
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
	return  LambdaLR(optimizer, lr_lambda, last_epoch)




if __name__ == '__main__':
	model = MyModel()
	model.train()



































