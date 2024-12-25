import torch
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


# 定义设备（GPU or CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载自然训练的模型
natural_model = torch.load('naturally_trained_model.pth', map_location=device)
natural_model.device = device
#加载pgd训练的模型
pgd_moodel = torch.load('pgd_trained_model.pth', map_location=device)
pgd_moodel.device = device


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


#加载自然数据
loader = Cifar10DataLoader(show_examples=False, dtype=torch.float32,flatten=False)
data_set = loader.preprocess_cifar10()
X_train,y_train = data_set['X_train'],data_set['y_train']
#加载对抗数据
adv_examples, adv_labels = torch.load('cifar10_adv_examples.pth')
adv_examples = adv_examples.to(device)
adv_labels = adv_labels.to(device)
#拼接数据
adv_ratio =0.2
#控制对抗样本比例
num_adv_examples = int(adv_examples.shape[0] * adv_ratio)

# Select a subset of adversarial examples
adv_examples_subset = adv_examples[:num_adv_examples]#从adv_examples中选择从第0个元素到第num_adv_examples-1个元素
adv_labels_subset = adv_labels[:num_adv_examples]

# Concatenate the original and adversarial examples
mixed_data = torch.cat((X_train, adv_examples_subset), 0)
mixed_labels = torch.cat((y_train, adv_labels_subset), 0)



test_dataset = CustomDataset(mixed_data, mixed_labels)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)#将数据分为多个批次，每个批次包含batch_size个数据



# 定义评估函数
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估时关闭梯度计算，提高速度，节省内存
        for images, labels in test_loader:#test_loder的每个batch是一个（inputs,labels）的元组，inputs 是形状为 (batch_size, 3, 32, 32) 的张量，labels 是形状为 (batch_size,) 的张量
            images, labels = images.to(device), labels.to(device)
            outputs = model.loss(images)  # 不传递 y 参数，确保模式为 'test'
           
            _, predicted = torch.max(outputs, 1)#torch.max()返回两个值，第一个是最大值，第二个是最大值的索引，这里1表示按行取最大值，cfar10共有十类，故每行有十个值，选出最大值的索引即选出种类
            total += labels.size(0)#labels.size(0)返回的是labels的行数,表示的是 batch_size
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# 评估模型
natural_accuracy = evaluate_model(natural_model, test_loader)
pgd_accuracy = evaluate_model(pgd_moodel, test_loader)
print(f"Natural accuracy: {natural_accuracy:.2f}%")
print(f"PGD accuracy: {pgd_accuracy:.2f}%")

accuracies = [natural_accuracy, pgd_accuracy]
labels = ['Natural Accuracy', 'PGD Accuracy']

# 创建条形图
plt.figure(figsize=(8, 4))
plt.bar(labels, accuracies, color=['blue', 'red'])
plt.xlabel('Model Type')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Model Accuracies')
plt.ylim([0, 100])  # 设置y轴的范围，使得图标更易读
plt.show()