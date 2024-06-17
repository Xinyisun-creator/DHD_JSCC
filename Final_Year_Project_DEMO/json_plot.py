import json
import matplotlib.pyplot as plt

# 假设你的 JSON 文件名为 data.json
file_path = './LOG/JSCC_model_evaluation_metrics_20240617-112823.json'

# 从 JSON 文件读取数据
with open(file_path, 'r') as file:
    data = json.load(file)

# 提取每个变量的数据
epochs = [entry['epoch'] for entry in data]
valid_loss = [entry['valid_loss'] for entry in data]
psnr = [entry['PSNR'] for entry in data]
ssim = [entry['SSIM'] for entry in data]
hd_loss = [entry['HD_loss'] for entry in data]

# 创建一个包含所有变量的图表
plt.figure(figsize=(12, 8))

# 有效损失
plt.subplot(3, 1, 1)
plt.plot(epochs, valid_loss, label='Valid Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Valid Loss')
plt.title('Valid Loss over Epochs')
plt.legend()

# PSNR
plt.subplot(3, 1, 2)
plt.plot(epochs, psnr, label='PSNR', color='green')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR over Epochs')
plt.legend()

# SSIM 和 HD_loss
plt.subplot(3, 1, 3)
plt.plot(epochs, ssim, label='SSIM', color='red')
plt.plot(epochs, hd_loss, label='HD Loss', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('SSIM and HD Loss over Epochs')
plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.savefig("experiment_0617.png")
