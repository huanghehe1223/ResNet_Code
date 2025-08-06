import torchvision.transforms as T
import torchvision.transforms.v2 as TV2
import torch
import torch.nn as nn
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_data():
    """加载训练好的模型和测试数据"""
    
    # 测试集图像预处理
    test_transform = TV2.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
    
    # 数据路径
    test_path = "processed_dataset_first/test"
    
    # 加载测试数据集
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 加载类别映射
    idx_to_labels = np.load('idx_to_labels.npy', allow_pickle=True).item()
    labels_to_idx = np.load('labels_to_idx.npy', allow_pickle=True).item()
    
    num_classes = len(idx_to_labels)
    
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # 加载最佳模型权重
    if os.path.exists('best_model.pth'):
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型，验证准确率: {checkpoint['best_val_accuracy']:.4f}")
    else:
        print("警告：未找到 best_model.pth 文件，使用未训练的模型")
    
    model.eval()
    
    return model, test_loader, device, idx_to_labels, test_dataset

def evaluate_test_set_detailed(model, test_loader, device):
    """详细评估测试集"""
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    loss_list = []
    labels_list = []
    preds_list = []
    
    print("正在评估测试集...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            labels = labels.cpu().numpy()
            
            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
    
    return labels_list, preds_list, np.mean(loss_list)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, normalized=False):
    """绘制混淆矩阵"""
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    if normalized:
        cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix (Recall)'
        fmt = '.3f'
        cbar_label = 'Recall'
        filename_suffix = '_normalized'
    else:
        cm_plot = cm
        title = 'Confusion Matrix'
        fmt = 'd'
        cbar_label = 'Count'
        filename_suffix = ''
    
    # 设置图形大小
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图
    sns.heatmap(cm_plot, 
                annot=True, 
                fmt=fmt, 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': cbar_label})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    
    # 旋转标签以防重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        save_file = save_path + filename_suffix + '.png'
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存为 {save_file}")
    
    plt.show()
    
    return cm

def detailed_test_evaluation():
    """完整的详细测试评估"""
    
    print("="*70)
    print("详细测试集评估")
    print("="*70)
    
    # 加载模型和数据
    model, test_loader, device, idx_to_labels, test_dataset = load_model_and_data()
    
    # 获取类别名称
    class_names = [idx_to_labels[i] for i in range(len(idx_to_labels))]
    
    print(f"测试集信息:")
    print(f"- 样本总数: {len(test_dataset)}")
    print(f"- 类别数量: {len(class_names)}")
    print(f"- 类别名称: {class_names}")
    
    # 评估测试集
    y_true, y_pred, test_loss = evaluate_test_set_detailed(model, test_loader, device)
    
    # 计算总体指标
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    print("\n" + "="*70)
    print("总体评估指标")
    print("="*70)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision-macro): {precision_macro:.4f}")
    print(f"召回率 (Recall-macro): {recall_macro:.4f}")
    print(f"F1分数 (F1-score-macro): {f1_macro:.4f}")
    print(f"损失 (Loss): {test_loss:.4f}")
    
    # 逐类别详细报告
    print("\n" + "="*70)
    print("逐类别详细评估报告")
    print("="*70)
    
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=False
    )
    print(report)
    
    # 各类别样本数统计
    print("\n" + "-"*50)
    print("各类别样本数统计:")
    print("-"*50)
    unique, counts = np.unique(y_true, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"{class_names[idx]:<20}: {count:>4} 个样本")
    
    # 绘制混淆矩阵
    print("\n" + "-"*50)
    print("绘制混淆矩阵")
    print("-"*50)
    
    # 原始混淆矩阵
    cm = plot_confusion_matrix(y_true, y_pred, class_names, 
                               save_path='confusion_matrix', 
                               normalized=False)
    
    # 归一化混淆矩阵（召回率）
    cm_normalized = plot_confusion_matrix(y_true, y_pred, class_names, 
                                          save_path='confusion_matrix', 
                                          normalized=True)
    
    # 生成详细的分类报告字典
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    # 保存详细评估结果
    evaluation_results = {
        'overall_metrics': {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_score_macro': f1_macro,
            'loss': test_loss,
            'total_samples': len(y_true)
        },
        'class_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    # 保存结果
    np.save('detailed_test_results.npy', evaluation_results)
    print(f"\n详细评估结果已保存为: detailed_test_results.npy")
    
    # 显示性能最好和最差的类别
    print("\n" + "-"*50)
    print("类别性能分析")
    print("-"*50)
    
    class_f1_scores = []
    for class_name in class_names:
        if class_name in report_dict:
            f1 = report_dict[class_name]['f1-score']
            class_f1_scores.append((class_name, f1))
    
    # 按F1分数排序
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("各类别F1分数排名:")
    for i, (class_name, f1) in enumerate(class_f1_scores, 1):
        print(f"{i:2d}. {class_name:<20}: {f1:.4f}")
    
    print(f"\n性能最好的类别: {class_f1_scores[0][0]} (F1: {class_f1_scores[0][1]:.4f})")
    print(f"性能最差的类别: {class_f1_scores[-1][0]} (F1: {class_f1_scores[-1][1]:.4f})")
    
    return evaluation_results

if __name__ == "__main__":
    try:
        results = detailed_test_evaluation()
        print("\n" + "="*70)
        print("评估完成！")
        print("="*70)
        print("生成的文件:")
        print("- confusion_matrix.png (原始混淆矩阵)")
        print("- confusion_matrix_normalized.png (归一化混淆矩阵)")
        print("- detailed_test_results.npy (详细评估结果)")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        print("请确保:")
        print("1. 已完成模型训练并生成了 best_model.pth")
        print("2. 存在 idx_to_labels.npy 和 labels_to_idx.npy 文件")
        print("3. 测试数据集路径正确: processed_dataset_first/test")