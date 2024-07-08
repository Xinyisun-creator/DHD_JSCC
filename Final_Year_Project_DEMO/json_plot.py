import json
import matplotlib.pyplot as plt

file_path1 = './LOG/train_JSCC_model_with_nuswide_evaluation_metrics_20240617-174421.json'
file_path2 = './LOG/JSCC_model_evaluation_metrics_20240617-112823.json'
file_path3 = './LOG/TRAINboth_JSCC(tarined_withfixedDHD)_and_DHD_evaluation_metrics_20240625-132754.json'

# 从 JSON 文件读取数据
with open(file_path1, 'r') as file:
    data1 = json.load(file)
with open(file_path2, 'r') as file:
    data2 = json.load(file)
with open(file_path3, 'r') as file:
    data3 = json.load(file)

# 提取 JSON 文件的每个变量的数据
epochs1 = [entry['epoch'] for entry in data1]
valid_loss1 = [entry['valid_loss'] for entry in data1]
psnr1 = [entry['PSNR'] for entry in data1]
ssim1 = [entry['SSIM'] for entry in data1]

epochs2 = [entry['epoch'] for entry in data2]
valid_loss2 = [entry['valid_loss'] for entry in data2]
psnr2 = [entry['PSNR'] for entry in data2]
ssim2 = [entry['SSIM'] for entry in data2]
hd_loss2 = [entry['HD_loss'] for entry in data2]

epochs3 = [entry['epoch'] for entry in data3]
valid_loss3 = [entry['valid_loss'] for entry in data3]
psnr3 = [entry['PSNR'] for entry in data3]
ssim3 = [entry['SSIM'] for entry in data3]
hd_loss3 = [entry['HD_loss'] for entry in data3]
map3 = [entry['mAP'] for entry in data3]
dhd_maxmap3 = [entry['DHD_MAXmap'] for entry in data3]

# 创建一个包含所有变量的图表
plt.figure(figsize=(12, 10))

# 有效损失
plt.subplot(5, 1, 1)
plt.plot(epochs1, valid_loss1, label='Valid Loss (Model 1)', color='blue')
plt.plot(epochs2, valid_loss2, label='Valid Loss (Model 2)', color='cyan')
plt.plot(epochs3, valid_loss3, label='Valid Loss (Model 3)', color='magenta')
plt.xlabel('Epoch')
plt.ylabel('Valid Loss')
plt.title('Valid Loss over Epochs')
plt.legend()

# PSNR
plt.subplot(5, 1, 2)
plt.plot(epochs1, psnr1, label='PSNR (Model 1)', color='green')
plt.plot(epochs2, psnr2, label='PSNR (Model 2)', color='lightgreen')
plt.plot(epochs3, psnr3, label='PSNR (Model 3)', color='lime')
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR over Epochs')
plt.legend()

# SSIM
plt.subplot(5, 1, 3)
plt.plot(epochs1, ssim1, label='SSIM (Model 1)', color='red')
plt.plot(epochs2, ssim2, label='SSIM (Model 2)', color='orange')
plt.plot(epochs3, ssim3, label='SSIM (Model 3)', color='yellow')
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('SSIM over Epochs')
plt.legend()

# HD_loss
plt.subplot(5, 1, 4)
plt.plot(epochs2, hd_loss2, label='HD Loss (Model 2)', color='purple')
plt.plot(epochs3, hd_loss3, label='HD Loss (Model 3)', color='violet')
plt.xlabel('Epoch')
plt.ylabel('HD Loss')
plt.title('HD Loss over Epochs')
plt.legend()

# mAP and DHD_MAXmap
plt.subplot(5, 1, 5)
plt.plot(epochs3, map3, label='mAP (Model 3)', color='brown')
plt.axhline(y=dhd_maxmap3[-1], color='grey', linestyle='--', label=f'DHD_MAXmap (Model 3): {dhd_maxmap3[-1]}')
plt.xlabel('Epoch')
plt.ylabel('mAP / DHD_MAXmap')
plt.title('mAP and DHD_MAXmap over Epochs')
plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.savefig("./images/experiment_comparison_JSCC_nuswide_0626.png")
plt.show()