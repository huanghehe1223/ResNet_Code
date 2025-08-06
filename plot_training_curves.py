import numpy as np
import matplotlib.pyplot as plt
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_training_history():
    """加载训练历史数据"""
    try:
        train_losses = np.load('train_losses.npy')
        train_accuracies = np.load('train_accuracies.npy')
        val_losses = np.load('val_losses.npy')
        val_accuracies = np.load('val_accuracies.npy')
        
        print(f"成功加载训练历史数据:")
        print(f"- 训练轮数: {len(train_losses)} epochs")
        print(f"- 训练数据形状: Loss({len(train_losses)}), Accuracy({len(train_accuracies)})")
        print(f"- 验证数据形状: Loss({len(val_losses)}), Accuracy({len(val_accuracies)})")
        
        return train_losses, train_accuracies, val_losses, val_accuracies
        
    except FileNotFoundError as e:
        print(f"错误: 无法找到训练历史文件")
        print("请确保以下文件存在:")
        print("- train_losses.npy")
        print("- train_accuracies.npy") 
        print("- val_losses.npy")
        print("- val_accuracies.npy")
        print("\n这些文件应该在训练完成后自动生成")
        return None, None, None, None

def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    """绘制训练过程中的loss和accuracy曲线"""
    
    if any(x is None for x in [train_losses, train_accuracies, val_losses, val_accuracies]):
        print("无法绘制曲线：训练历史数据缺失")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    print(f"\n开始绘制训练曲线...")
    print(f"训练轮数: {len(train_losses)} epochs")
    
    # 创建2x1的子图 - 组合图
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
    
    # 保存组合图
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 训练曲线组合图已保存: training_curves.png")
    plt.show()
    
    # 单独绘制Loss曲线（更大更详细的图）
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=5, alpha=0.8)
    plt.title('Training and Validation Loss Curves', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加最值标注
    min_train_loss_idx = np.argmin(train_losses)
    min_val_loss_idx = np.argmin(val_losses)
    plt.annotate(f'Min Train Loss: {train_losses[min_train_loss_idx]:.4f}', 
                xy=(min_train_loss_idx+1, train_losses[min_train_loss_idx]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(f'Min Val Loss: {val_losses[min_val_loss_idx]:.4f}', 
                xy=(min_val_loss_idx+1, val_losses[min_val_loss_idx]), 
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 损失曲线已保存: loss_curves.png")
    plt.show()
    
    # 单独绘制Accuracy曲线（更大更详细的图）
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2.5, marker='s', markersize=5, alpha=0.8)
    plt.title('Training and Validation Accuracy Curves', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加最值标注
    max_train_acc_idx = np.argmax(train_accuracies)
    max_val_acc_idx = np.argmax(val_accuracies)
    plt.annotate(f'Max Train Acc: {train_accuracies[max_train_acc_idx]:.4f}', 
                xy=(max_train_acc_idx+1, train_accuracies[max_train_acc_idx]), 
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.annotate(f'Max Val Acc: {val_accuracies[max_val_acc_idx]:.4f}', 
                xy=(max_val_acc_idx+1, val_accuracies[max_val_acc_idx]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('accuracy_curves.png', dpi=300, bbox_inches='tight')
    print("✓ 准确率曲线已保存: accuracy_curves.png")
    plt.show()

def print_training_statistics(train_losses, train_accuracies, val_losses, val_accuracies):
    """打印训练统计信息"""
    
    if any(x is None for x in [train_losses, train_accuracies, val_losses, val_accuracies]):
        return
    
    print("\n" + "="*60)
    print("训练统计信息")
    print("="*60)
    
    print(f"训练轮数: {len(train_losses)} epochs")
    print(f"\n最终指标:")
    print(f"  训练损失: {train_losses[-1]:.4f}")
    print(f"  训练准确率: {train_accuracies[-1]:.4f}")
    print(f"  验证损失: {val_losses[-1]:.4f}")
    print(f"  验证准确率: {val_accuracies[-1]:.4f}")
    
    print(f"\n最佳指标:")
    min_train_loss_epoch = np.argmin(train_losses) + 1
    max_train_acc_epoch = np.argmax(train_accuracies) + 1
    min_val_loss_epoch = np.argmin(val_losses) + 1
    max_val_acc_epoch = np.argmax(val_accuracies) + 1
    
    print(f"  最佳训练损失: {np.min(train_losses):.4f} (第 {min_train_loss_epoch} 轮)")
    print(f"  最佳训练准确率: {np.max(train_accuracies):.4f} (第 {max_train_acc_epoch} 轮)")
    print(f"  最佳验证损失: {np.min(val_losses):.4f} (第 {min_val_loss_epoch} 轮)")
    print(f"  最佳验证准确率: {np.max(val_accuracies):.4f} (第 {max_val_acc_epoch} 轮)")
    
    # 计算训练稳定性指标
    train_loss_std = np.std(train_losses)
    val_loss_std = np.std(val_losses)
    
    print(f"\n训练稳定性:")
    print(f"  训练损失标准差: {train_loss_std:.4f}")
    print(f"  验证损失标准差: {val_loss_std:.4f}")
    
    # 检查过拟合
    final_gap_loss = train_losses[-1] - val_losses[-1]
    final_gap_acc = val_accuracies[-1] - train_accuracies[-1]
    
    print(f"\n过拟合检查:")
    print(f"  最终损失差距 (val - train): {final_gap_loss:.4f}")
    print(f"  最终准确率差距 (val - train): {final_gap_acc:.4f}")
    
    if final_gap_loss > 0.5:
        print("  ⚠️  可能存在过拟合 (验证损失明显高于训练损失)")
    elif abs(final_gap_acc) < 0.05:
        print("  ✓ 训练良好 (训练和验证准确率接近)")
    else:
        print("  ℹ️  训练正常")

def main():
    """主函数"""
    print("="*60)
    print("训练曲线可视化工具")
    print("="*60)
    
    # 加载训练历史
    train_losses, train_accuracies, val_losses, val_accuracies = load_training_history()
    
    if train_losses is not None:
        # 打印统计信息
        print_training_statistics(train_losses, train_accuracies, val_losses, val_accuracies)
        
        # 绘制曲线
        plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
        
        print("\n" + "="*60)
        print("绘制完成！生成的图像文件:")
        print("="*60)
        print("📊 training_curves.png - 训练曲线组合图")
        print("📉 loss_curves.png - 损失曲线详细图")
        print("📈 accuracy_curves.png - 准确率曲线详细图")
        
    else:
        print("\n请先运行训练程序生成训练历史文件")

if __name__ == "__main__":
    main()