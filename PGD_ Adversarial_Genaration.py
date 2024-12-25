import torch
import torch.nn as nn   
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import DeepConvNet
import matplotlib.pyplot as plt
import os
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
import random
import HELPER
from torch.utils.data import Dataset

# 定义 PGD 对抗样本生成函数
def pgd_attack(model, images, labels, eps, alpha, iters):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        loss,grads_dict = model.loss(images,labels)
        # print(grads_dict['W1'][1].shape)
        # print(grads_dict['x'].shape)
        input_grad=grads_dict['x']
        #images.grad=input_grad

        # 计算损失
        # loss = torch.nn.functional.cross_entropy(outputs, labels)
        # model.zero_grad()  # 清零梯度
        # loss.backward()
        

        # 使用梯度更新图像
        input_grad_sign=input_grad.sign()
        adv_images = images + alpha * input_grad_sign
        # 确保对抗样本在合法范围内
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        print('Iter:',i,'Loss:',loss.item())

    return images

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('naturally_trained_model.pth', map_location=device)
model.device = device


# 定义对抗训练参数
epsilon = 0.1  # 允许的最大扰动
alpha = 0.001   # 每步的扰动
num_iter = 10 # 迭代次数
num_epochs = 5
batch_size = 64
learning_rate = 0.001



class Cifar10DataLoader:
    def __init__(self, cuda=True, show_examples=True, bias_trick=False, flatten=True, validation_ratio=0.2, dtype=torch.float32):
        self.cuda = cuda
        self.show_examples = show_examples
        self.bias_trick = bias_trick
        self.flatten = flatten
        self.validation_ratio = validation_ratio
        self.dtype = dtype

    def _extract_tensors(self, dset, num=None, x_dtype=torch.float32):
        """
        Extract the data and labels from a CIFAR10 dataset object
        and convert them to tensors.

        Input:
        - dset: A torchvision.datasets.CIFAR10 object
        - num: Optional. If provided, the number of samples to keep.
        - x_dtype: Optional. data type of the input image

        Returns:
        - x: `x_dtype` tensor of shape (N, 3, 32, 32)
        - y: int64 tensor of shape (N,)
        """
        x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)
        y = torch.tensor(dset.targets, dtype=torch.int64)
        if num is not None:
            if num <= 0 or num > x.shape[0]:
                raise ValueError(
                    "Invalid value num=%d; must be in the range [0, %d]"
                    % (num, x.shape[0])
                )
            x = x[:num].clone()
            y = y[:num].clone()
        return x, y

    def cifar10(self, num_train=None, num_test=None, x_dtype=torch.float32):
        """
        Return the CIFAR10 dataset, automatically downloading it if necessary.
        This function can also subsample the dataset.

        Inputs:
        - num_train: [Optional] How many samples to keep from the training set.
        If not provided, then keep the entire training set.
        - num_test: [Optional] How many samples to keep from the test set.
        If not provided, then keep the entire test set.
        - x_dtype: [Optional] Data type of the input image

        Returns:
        - x_train: `x_dtype` tensor of shape (num_train, 3, 32, 32)
        - y_train: int64 tensor of shape (num_train, 3, 32, 32)
        - x_test: `x_dtype` tensor of shape (num_test, 3, 32, 32)
        - y_test: int64 tensor of shape (num_test, 3, 32, 32)
        """
        download = not os.path.isdir("cifar-10-batches-py")
        dset_train = CIFAR10(root=".", download=download, train=True)
        dset_test = CIFAR10(root=".", train=False)
        x_train, y_train = self._extract_tensors(dset_train, num_train, x_dtype)
        x_test, y_test = self._extract_tensors(dset_test, num_test, x_dtype)

        return x_train, y_train, x_test, y_test

    def preprocess_cifar10(self):
        """
        Returns a preprocessed version of the CIFAR10 dataset,
        automatically downloading if necessary. We perform the following steps:

        (0) [Optional] Visualize some images from the dataset
        (1) Normalize the data by subtracting the mean
        (2) Reshape each image of shape (3, 32, 32) into a vector
            of shape (3072,)
        (3) [Optional] Bias trick: add an extra dimension of ones to the data
        (4) Carve out a validation set from the training set

        Inputs:
        - cuda: If true, move the entire dataset to the GPU
        - validation_ratio: Float in the range (0, 1) giving the
        fraction of the train set to reserve for validation
        - bias_trick: Boolean telling whether or not to apply the bias trick
        - show_examples: Boolean telling whether or not to visualize
        data samples
        - dtype: Optional, data type of the input image X

        Returns a dictionary with the following keys:
        - 'X_train': `dtype` tensor of shape (N_train, D) giving
        training images
        - 'X_val': `dtype` tensor of shape (N_val, D) giving val images
        - 'X_test': `dtype` tensor of shape (N_test, D) giving test images
        - 'y_train': int64 tensor of shape (N_train,) giving training labels
        - 'y_val': int64 tensor of shape (N_val,) giving val labels
        - 'y_test': int64 tensor of shape (N_test,) giving test labels

        N_train, N_val, and N_test are the number of examples in the train,
        val, and test sets respectively. The precise values of N_train and
        N_val are determined by the input parameter validation_ratio. D is
        the dimension of the image data;
        if bias_trick is False, then D = 32 * 32 * 3 = 3072;
        if bias_trick is True then D = 1 + 32 * 32 * 3 = 3073.
        """
        X_train, y_train, X_test, y_test = self.cifar10(x_dtype=self.dtype)

        # Move data to the GPU
        if self.cuda:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()

        # 0. Visualize some examples from the dataset.
        if self.show_examples:
            classes = [
                "plane", "car", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
            samples_per_class = 12
            samples = []
            HELPER.reset_seed(0)
            for y, cls in enumerate(classes):
                plt.text(-4, 34 * y + 18, cls, ha="right")
                (idxs,) = (y_train == y).nonzero(as_tuple=True)
                for i in range(samples_per_class):
                    idx = idxs[random.randrange(idxs.shape[0])].item()
                    samples.append(X_train[idx])
            img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
            plt.imshow(HELPER.tensor_to_image(img))
            plt.axis("off")
            plt.show()

        # 1. Normalize the data: subtract the mean RGB (zero mean)
        mean_image = X_train.mean(dim=(0, 2, 3), keepdim=True)
        X_train -= mean_image
        X_test -= mean_image

        # 2. Reshape the image data into rows
        if self.flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        # 3. Add bias dimension and transform into columns
        if self.bias_trick:
            ones_train = torch.ones(X_train.shape[0], 1, device=X_train.device)
            X_train = torch.cat([X_train, ones_train], dim=1)
            ones_test = torch.ones(X_test.shape[0], 1, device=X_test.device)
            X_test = torch.cat([X_test, ones_test], dim=1)

        # 4. Take the validation set from the training set
        num_training = int(X_train.shape[0] * (1.0 - self.validation_ratio))
        num_validation = X_train.shape[0] - num_training

        X_val, y_val = X_train[num_training:], y_train[num_training:]
        X_train, y_train = X_train[:num_training], y_train[:num_training]

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
        }


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


#数据
loader = Cifar10DataLoader(show_examples=False, dtype=torch.float32,flatten=False)
data_set = loader.preprocess_cifar10()
X_train,y_train = data_set['X_train'],data_set['y_train']
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#将数据分为多个批次，每个批次包含batch_size个数据


# 定义优化器和损失函数
parameters={}
parameters = model.params
parameters_list=list(parameters.values())#不加values的话list处理的是key
#这里并没有用到，了解一下list函数

#非张量检测
# non_tensor_element = None

# for element in parameters_list:
#     if not torch.is_tensor(element):
#         non_tensor_element = element
#         break

# print("唯一的非张量变量是:", non_tensor_element)




# 训练模型并生成对抗样本
adv_examples = []
adv_labels = []
epoch=0
for epoch in range(num_epochs):
    dataset_num=0
    for data, target in train_loader:#遍历每个批次
        data, target = data.to(device), target.to(device)
        
        print('epoch:%d ,dataset_num:%d'%(epoch,dataset_num))
        dataset_num+=1
        # 生成对抗样本
        adv_data = pgd_attack(model, data, target, epsilon, alpha, num_iter)

        # 将对抗样本和对应的标签保存到列表中
        adv_examples.append(adv_data.cpu())
        adv_labels.append(target.cpu())
        #训练时注意要把对抗样本和原始样本混合在一起训练
        # # 混合对抗样本和原始数据进行训练
        # mixed_data = torch.cat([data, adv_data])
        # mixed_target = torch.cat([target, target])

        # # 前向传播
        # optimizer.zero_grad()
        # output = model(mixed_data)
        # loss = criterion(output, mixed_target)

        # # 反向传播和优化
        # loss.backward()
        # optimizer.step()

       
    epoch+=1

# 将对抗样本和标签保存到文件
adv_examples = torch.cat(adv_examples)
adv_labels = torch.cat(adv_labels)
torch.save((adv_examples, adv_labels), 'cifar10_adv_examples.pth')
print('Adversarial examples saved as cifar10_adv_examples.pth')


