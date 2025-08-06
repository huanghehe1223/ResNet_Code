import torchvision.transforms as T
import torchvision.transforms.v2 as TV2
from torchvision import datasets

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
    TV2.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])


train_path = "processed_dataset_first/train"
val_path = "processed_dataset_first/val"
test_path = "processed_dataset_first/test"


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