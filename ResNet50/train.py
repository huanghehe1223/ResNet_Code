import torchvision.transforms as T
import torchvision.transforms.v2 as TV2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet34_Weights
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import os
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime

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

# 数据路径
train_path = "processed_dataset_first/train"
val_path = "processed_dataset_first/val"
test_path = "processed_dataset_first/test"

# 加载数据集
train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_path, transform=test_transform)
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

# 打印数据集信息
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

# 保存映射关系
np.save('idx_to_labels.npy', idx_to_labels)
np.save('labels_to_idx.npy', train_dataset.class_to_idx)

# 数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 模型配置
model = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)  # 载入预训练模型

# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

print("修改后的全连接层：", model.fc)

# 优化器，只优化最后的全连接层
optimizer = optim.Adam(model.fc.parameters())
# 学习率降低策略
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练轮次 Epoch
EPOCHS = 100

# 文件保存后缀
suffix = ""  # 可以修改为任何你想要的后缀，如 "_v2" 或 "_experiment1"

def train_one_batch(images, labels, epoch, batch_idx):
    '''
    运行一个 batch 的训练，返回当前 batch 的训练日志
    '''
    
    model.train()
    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值
    
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
    log_train['train_precision'] = precision_score(labels, preds, average='macro', zero_division=0)
    log_train['train_recall'] = recall_score(labels, preds, average='macro', zero_division=0)
    log_train['train_f1-score'] = f1_score(labels, preds, average='macro', zero_division=0)
    
    return log_train

def evaluate_val_set(epoch):
    '''
    在整个验证集上评估，返回分类评估指标日志
    '''
    model.eval()
    
    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in val_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个验证集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    log_val = {}
    log_val['epoch'] = epoch
    
    # 计算分类评估指标
    log_val['val_loss'] = np.mean(loss_list)
    log_val['val_accuracy'] = accuracy_score(labels_list, preds_list)
    log_val['val_precision'] = precision_score(labels_list, preds_list, average='macro', zero_division=0)
    log_val['val_recall'] = recall_score(labels_list, preds_list, average='macro', zero_division=0)
    log_val['val_f1-score'] = f1_score(labels_list, preds_list, average='macro', zero_division=0)
    
    return log_val

def evaluate_test_set(epoch):
    '''
    在整个测试集上评估，返回分类评估指标日志
    '''
    model.eval()
    
    loss_list = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
        
    log_test = {}
    log_test['epoch'] = epoch
    
    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro', zero_division=0)
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro', zero_division=0)
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro', zero_division=0)
    
    # 添加真实标签和预测标签，用于详细分析
    log_test['y_true'] = labels_list
    log_test['y_pred'] = preds_list
    
    return log_test

def detailed_test_evaluation():
    '''
    详细的测试集评估，包含逐个类别的指标和混淆矩阵
    '''
    print("\n" + "="*60)
    print("详细测试集评估")
    print("="*60)
    
    # 获取测试结果
    test_log = evaluate_test_set(0)
    y_true = test_log['y_true']
    y_pred = test_log['y_pred']
    
    # 加载类别映射
    idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
    class_names = [idx_to_labels[i] for i in range(len(idx_to_labels))]
    
    print(f"总体指标:")
    print(f"准确率 (Accuracy): {test_log['test_accuracy']:.4f}")
    print(f"精确率 (Precision-macro): {test_log['test_precision']:.4f}")
    print(f"召回率 (Recall-macro): {test_log['test_recall']:.4f}")
    print(f"F1分数 (F1-score-macro): {test_log['test_f1-score']:.4f}")
    print(f"损失 (Loss): {test_log['test_loss']:.4f}")
    
    print("\n" + "-"*60)
    print("逐类别详细评估报告:")
    print("-"*60)
    
    # 生成详细的分类报告
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=False
    )
    print(report)
    
    # 计算逐类别指标（数值形式，便于后续处理）
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    print("\n" + "-"*60)
    print("各类别样本数统计:")
    print("-"*60)
    unique, counts = np.unique(y_true, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"{class_names[idx]}: {count} 个样本")
    
    # 绘制混淆矩阵
    print("\n正在绘制混淆矩阵...")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 设置图形大小
    plt.figure(figsize=(12, 10))
    
    # 绘制混淆矩阵热力图
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Sample Count'})
    
    plt.title('Test Set Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # 旋转x轴标签以防重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存混淆矩阵图
    plt.savefig(f'confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存为 confusion_matrix{suffix}.png")
    
    # 显示图形
    plt.show()
    
    # 计算并显示归一化混淆矩阵（按行归一化，显示召回率）
    print("\n正在绘制归一化混淆矩阵（召回率）...")
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.3f', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Recall'})
    
    plt.title('Test Set Normalized Confusion Matrix (Recall)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存归一化混淆矩阵图
    plt.savefig(f'confusion_matrix_normalized{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"归一化混淆矩阵已保存为 confusion_matrix_normalized{suffix}.png")
    
    plt.show()
    
    # 保存详细评估结果
    evaluation_results = {
        'overall_metrics': {
            'accuracy': test_log['test_accuracy'],
            'precision_macro': test_log['test_precision'],
            'recall_macro': test_log['test_recall'],
            'f1_score_macro': test_log['test_f1-score'],
            'loss': test_log['test_loss']
        },
        'class_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names
    }
    
    # 保存为numpy文件
    np.save(f'detailed_test_results{suffix}.npy', evaluation_results)
    print(f"\n详细评估结果已保存为 detailed_test_results{suffix}.npy")
    
    return evaluation_results

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    '''
    绘制训练过程中的loss和accuracy曲线
    '''
    epochs = range(1, len(train_losses) + 1)
    
    # 创建2x1的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制Loss曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制Accuracy曲线
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'training_curves{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Training curves saved as training_curves{suffix}.png")
    
    # 显示图像
    plt.show()
    
    # 单独绘制Loss曲线（更大的图）
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    plt.title('Training and Validation Loss Curves', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'loss_curves{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Loss curves saved as loss_curves{suffix}.png")
    plt.show()
    
    # 单独绘制Accuracy曲线（更大的图）
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    plt.title('Training and Validation Accuracy Curves', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'accuracy_curves{suffix}.png', dpi=300, bbox_inches='tight')
    print(f"Accuracy curves saved as accuracy_curves{suffix}.png")
    plt.show()
    
    # 打印训练统计信息
    print("\n" + "="*50)
    print("Training Statistics")
    print("="*50)
    print(f"Total epochs trained: {len(train_losses)}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final training accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")
    print(f"Best training loss: {min(train_losses):.4f} (epoch {train_losses.index(min(train_losses))+1})")
    print(f"Best training accuracy: {max(train_accuracies):.4f} (epoch {train_accuracies.index(max(train_accuracies))+1})")
    print(f"Best validation loss: {min(val_losses):.4f} (epoch {val_losses.index(min(val_losses))+1})")
    print(f"Best validation accuracy: {max(val_accuracies):.4f} (epoch {val_accuracies.index(max(val_accuracies))+1})")

# 训练过程
def train_model():
    # 早停机制
    best_val_accuracy = 0.0
    best_val_loss = float('inf')  # 监测最小loss
    # 保存最佳模型权重
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 7
    patience_counter = 0
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # 记录训练开始时间
    train_start_time = time.time()
    start_datetime = datetime.datetime.now()
    
    print("开始训练...")
    print(f"训练开始时间：{start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总共 {EPOCHS} 个 epoch，早停耐心值：{patience}")
    print("早停监测指标：验证集loss")
    print("模型保存策略：保存准确率最佳的模型")
    
    # 逐个epoch训练,每个epoch逐个batch训练,每个batch训练后更新一次学习率
    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0
        
        total_batches = len(train_loader)
        half_point = total_batches // 2
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 训练一个batch
            log_train = train_one_batch(images, labels, epoch, batch_idx)
            
            running_loss += log_train['train_loss']
            running_accuracy += log_train['train_accuracy']
            num_batches += 1
            
            # 只在50%和100%时显示进度和日志
            if batch_idx == half_point:  # 50%进度
                progress = (batch_idx + 1) / total_batches * 100
                avg_loss = running_loss / num_batches
                avg_acc = running_accuracy / num_batches
                print(f"  训练进度 {progress:.0f}% ({batch_idx+1}/{total_batches}) - "
                      f"当前平均Loss: {avg_loss:.4f}, 当前平均Acc: {avg_acc:.4f}")
            elif batch_idx == total_batches - 1:  # 100%进度
                progress = 100
                avg_loss = running_loss / num_batches
                avg_acc = running_accuracy / num_batches
                print(f"  训练进度 {progress:.0f}% ({batch_idx+1}/{total_batches}) - "
                      f"最终平均Loss: {avg_loss:.4f}, 最终平均Acc: {avg_acc:.4f}")
        
        # 整个epoch训练结束，计算平均训练指标
        # 计算平均训练指标
        epoch_train_loss = running_loss / num_batches
        epoch_train_accuracy = running_accuracy / num_batches
        
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        
        # 验证阶段
        print("在验证集上评估...")
        log_val = evaluate_val_set(epoch)
        val_losses.append(log_val['val_loss'])
        val_accuracies.append(log_val['val_accuracy'])
        
        # 更新学习率
        lr_scheduler.step()
        
        # 打印本轮训练结果
        print(f'训练 - Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_accuracy:.4f}')
        print(f'验证 - Loss: {log_val["val_loss"]:.4f}, Accuracy: {log_val["val_accuracy"]:.4f}')
        print(f'验证 - Precision: {log_val["val_precision"]:.4f}, Recall: {log_val["val_recall"]:.4f}, F1-score: {log_val["val_f1-score"]:.4f}')
        
        # 保存最佳模型 - 基于准确率
        if log_val['val_accuracy'] > best_val_accuracy:
            best_val_accuracy = log_val['val_accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"新的最佳验证准确率: {best_val_accuracy:.4f}")
            
            # 保存最佳准确率模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'val_loss': log_val['val_loss'],
            }, f'best_model{suffix}.pth')
            print("最佳准确率模型已保存")
        
        # 早停机制 - 基于loss
        if log_val['val_loss'] < best_val_loss:
            best_val_loss = log_val['val_loss']
            patience_counter = 0
            print(f"新的最佳验证loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"验证loss未下降，耐心计数器: {patience_counter}/{patience}")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"验证loss连续 {patience} 轮未下降，触发早停")
            break
    
    # 计算训练总时长
    train_end_time = time.time()
    end_datetime = datetime.datetime.now()
    total_training_time = train_end_time - train_start_time
    
    # 格式化时长显示
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print(f'\n训练完成！最佳验证准确率: {best_val_accuracy:.4f}')
    print(f'最佳验证loss: {best_val_loss:.4f}')
    print("="*50)
    print("训练时间统计:")
    print(f"训练开始时间：{start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练结束时间：{end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总训练时长：{hours}小时 {minutes}分钟 {seconds}秒")
    print(f"总训练时长：{total_training_time:.2f}秒")
    print("="*50)
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    return train_losses, train_accuracies, val_losses, val_accuracies

# 主训练函数
if __name__ == "__main__":
    # 开始训练
    train_losses, train_accuracies, val_losses, val_accuracies = train_model()
    
    # 绘制训练曲线
    print("\n" + "="*60)
    print("绘制训练曲线")
    print("="*60)
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # 在测试集上进行详细评估
    evaluation_results = detailed_test_evaluation()
    
    # 保存训练历史
    np.save(f'train_losses{suffix}.npy', train_losses)
    np.save(f'train_accuracies{suffix}.npy', train_accuracies)
    np.save(f'val_losses{suffix}.npy', val_losses)
    np.save(f'val_accuracies{suffix}.npy', val_accuracies)
    
    print("\n" + "="*60)
    print("训练完成！所有文件已保存")
    print("="*60)
    print("保存的文件:")
    print(f"- best_model{suffix}.pth (最佳模型)")
    print(f"- training_curves{suffix}.png (训练曲线组合图)")
    print(f"- loss_curves{suffix}.png (损失曲线)")
    print(f"- accuracy_curves{suffix}.png (准确率曲线)")
    print(f"- confusion_matrix{suffix}.png (混淆矩阵)")
    print(f"- confusion_matrix_normalized{suffix}.png (归一化混淆矩阵)")
    print(f"- detailed_test_results{suffix}.npy (详细测试结果)")
    print(f"- train_losses{suffix}.npy, train_accuracies{suffix}.npy, val_losses{suffix}.npy, val_accuracies{suffix}.npy (训练历史)")