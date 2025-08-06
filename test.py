import torchvision.transforms as T
import torchvision.transforms.v2 as TV2


# train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
#                                       transforms.RandomHorizontalFlip(),
#                                       transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                      ])



# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = TV2.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])


# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transforms = TV2.Compose([
    # 几何变换
    T.RandomResizedCrop(224, scale=(0.75, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=90),  # 保守的旋转角度
    # 颜色变换
    T.ColorJitter(brightness=0.15,  saturation=0.15),
    # 自动对比度变换
    T.RandomAutocontrast(p=0.3),

    # 转换和标准化
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    # 张量级变换，随机遮挡
    # TV2.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])


train_path = "processed_dataset_first/train"
val_path = "processed_dataset_first/val"
test_path = "processed_dataset_first/test"

from torchvision import datasets

train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_path, transform=test_transform)
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
# 打印训练集，验证集，测试集，样本数量，类别个数，类别名称
print("训练集样本数量：", len(train_dataset))
print("验证集样本数量：", len(val_dataset))
print("测试集样本数量：", len(test_dataset))
print("训练集类别个数：", len(train_dataset.classes))
print("训练集类别名称：", train_dataset.classes)
print("验证集类别个数：", len(val_dataset.classes))
print("验证集类别名称：", val_dataset.classes)
print("测试集类别个数：", len(test_dataset.classes))
print("测试集类别名称：", test_dataset.classes)

print(train_dataset[0][0].shape)

print(train_dataset.class_to_idx)
# 字典key value给反过来
idx_to_labels = {v: k for k, v in train_dataset.class_to_idx.items()}
print(idx_to_labels)


# import numpy as np
# np.save('idx_to_labels.npy', idx_to_labels)
# np.save('labels_to_idx.npy', train_dataset.class_to_idx)

# 部署时加载映射关系
import numpy as np
idx_to_labels_test = np.load('idx_to_labels.npy', allow_pickle=True).item()
labels_to_idx_test = np.load('labels_to_idx.npy', allow_pickle=True).item()
print(idx_to_labels_test)
print(labels_to_idx_test)

from torch.utils.data import DataLoader
batch_size = 32
 
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)

from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler

import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torch import device
import torch

model = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1) # 载入预训练模型

# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)

print("修改后的全连接层：", model.fc)

# 优化器，只优化最后的全连接层
optimizer = optim.Adam(model.fc.parameters())
# 学习率降低策略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss() 

# 训练轮次 Epoch
EPOCHS = 30



for name, param in model.named_parameters():
    print(name, param.requires_grad)
    #只打印第一个参数的详细信息
    if name == 'conv1.weight':
        print(param)

# 获取第一批数据
first_batch = next(iter(test_loader))
images, labels = first_batch

# import torch

# x = torch.randn(2, 3, requires_grad=True, device='cuda:0')
# print(type(x))
# # 中文解释也打印
# print("原始数据：", x.data)
# print("梯度：", x.grad)
# print("创建该张量的函数：", x.grad_fn)
# print("是否需要梯度：", x.requires_grad)
# print("是否为叶子节点：", x.is_leaf)
# print("设备：", x.device)
# print("数据类型：", x.dtype)
# print("形状：", x.shape)
# print("维度数：", x.ndim)


# from tqdm import tqdm
# outputs = 123
# labels1 = 456
# loss1 = 789
# for images, labels in tqdm(train_loader):
#     images = images.to(device)
#     labels = labels.to(device)
#     outputs = model(images)
#     loss = criterion(outputs, labels)
#     loss1 = loss
    
#     break


# print(loss)
# print(outputs.grad_fn)

# print(outputs[0])
# print(labels1[0])
# print(labels1.shape)

#模型结构
# print(model)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
def train_one_batch(images, labels):
    '''
    运行一个 batch 的训练，返回当前 batch 的训练日志
    '''
    
    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images) # 输入模型，执行前向预测
    loss = criterion(outputs, labels) # 计算当前 batch 中，每个样本的平均交叉熵损失函数值
    
    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 获取当前 batch 的标签类别和预测类别
    #在第一个维度上取最大值，获得最大值的索引，获得当前 batch 所有图像的预测类别
    _, preds = torch.max(outputs, 1) 
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.cpu().numpy()
    
    log_train = {}
    # 当前epoch
    log_train['epoch'] = epoch
    # 当前batch
    log_train['batch'] = batch_idx
    # 计算分类评估指标
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    log_train['train_precision'] = precision_score(labels, preds, average='macro')
    log_train['train_recall'] = recall_score(labels, preds, average='macro')
    log_train['train_f1-score'] = f1_score(labels, preds, average='macro')
    
    return log_train


def evaluate_val_set():
    '''
    在整个测试集上评估，返回分类评估指标日志
    '''

    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in val_loader: # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1) # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels) # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    log_test = {}
    log_test['epoch'] = epoch
    
    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    
    return log_test


def evaluate_test_set():
    '''
    在整个测试集上评估，返回分类评估指标日志
    '''

    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in test_loader: # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images) # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1) # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels) # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    log_test = {}
    log_test['epoch'] = epoch
    
    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    
    return log_test
