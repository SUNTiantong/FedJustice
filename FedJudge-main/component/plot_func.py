import matplotlib.pyplot as plt

def plot_loss(epochs, batch_loss, filename='loss_plot.png'):
    """
    绘制 Loss 曲线。

    参数:
    - epochs: 训练的轮次
    - batch_loss: 每个批次的 Loss
    - filename: 保存图形的文件路径
    """
    plt.figure(figsize=(8, 6))
    batch = list(range(1, len(batch_loss) + 1))
    plt.plot(batch, batch_loss, label='Epoch Loss', color='tab:red')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 保存图形到文件
    plt.savefig(filename)
    plt.close()

def plot_accuracy(epochs, now_batch_accuracy, filename='accuracy_plot.png'):
    """
    绘制 Accuracy 曲线。

    参数:
    - epochs: 训练的轮次
    - now_batch_accuracy: 每个批次的 Accuracy
    - filename: 保存图形的文件路径
    """
    plt.figure(figsize=(8, 6))
    batch = list(range(1, len(now_batch_accuracy) + 1))
    
    plt.plot(batch, now_batch_accuracy, label='Epoch Accuracy', color='tab:green')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Epoch Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    # 保存图形到文件
    plt.savefig(filename)
    plt.close()



# 你可以调用 train_and_plot 来训练并绘制图形
# train_and_plot(lora_model, tokenizer, gender_dataset, dataset, args, epochs=10)


# accuracy 0.024658323095823095     total_loss: 239.60148717463017
# DEO: 0.022432673431861838 