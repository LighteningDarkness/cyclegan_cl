import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json
from scipy import signal

idx2label = {
    1:'breast',
    4:'liver',
    3:'kidney',
    2:'carotid',
    5:'thyroid'
}

# 计算 LNCC (局部归一化互相关)
def lncc(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # 归一化
    img1 = (img1 - np.mean(img1)) / np.std(img1)
    img2 = (img2 - np.mean(img2)) / np.std(img2)
    
    # 局部归一化互相关
    return np.mean((img1 - np.mean(img1)) * (img2 - np.mean(img2))) / np.std(img1) / np.std(img2)
def calculate_lncc(img1, img2, window_size=9):
    """计算局部归一化互相关 (Local Normalized Cross-Correlation)"""
    # 使用更高效的卷积方法计算LNCC
    if window_size % 2 == 0:
        window_size += 1
    
    # 创建高斯窗口
    kernel_size = window_size
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    
    # 计算局部均值
    img1_mean = signal.convolve2d(img1, kernel, mode='same')
    img2_mean = signal.convolve2d(img2, kernel, mode='same')
    
    # 减去局部均值
    img1_local = img1 - img1_mean
    img2_local = img2 - img2_mean
    
    # 计算局部标准差
    img1_var = signal.convolve2d(img1_local**2, kernel, mode='same')
    img2_var = signal.convolve2d(img2_local**2, kernel, mode='same')
    img1_std = np.sqrt(np.maximum(img1_var, 1e-6))
    img2_std = np.sqrt(np.maximum(img2_var, 1e-6))
    
    # 计算局部协方差
    img_cov = signal.convolve2d(img1_local * img2_local, kernel, mode='same')
    
    # 计算LNCC
    lncc = img_cov / (img1_std * img2_std + 1e-6)
    
    return np.mean(lncc)

# 计算所有图像的指标
def compute_metrics(gt_folder, generated_folder, label_file):
    # 读取标签映射
    with open(label_file, 'r') as f:
        labels = json.load(f)

    # 用于存储每个类别的指标
    metrics = {category: {'SSIM': 0, 'PSNR': 0, 'LNCC': 0, 'count': 0} for category in set(labels.values())}

    # 遍历GT文件夹中的图像
    for filename in os.listdir(gt_folder):
        gt_image_path = os.path.join(gt_folder, filename)
        gen_image_path = os.path.join(generated_folder, filename)

        if not os.path.isfile(gt_image_path) or not os.path.isfile(gen_image_path):
            continue

        # 从文件名中提取索引
        image_id = os.path.splitext(filename)[0]

        # 获取该图像的类别
        if image_id not in labels:
            print(f"Warning: {filename} not found in labels, skipping!")
            continue
        category = labels[image_id]

        # 读取图像
        gt_img = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
        gen_img = cv2.imread(gen_image_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像大小一致
        if gt_img.shape != gen_img.shape:
            print(f"Warning: {filename} has different shapes, skipping!")
            continue

        # 计算 SSIM, PSNR, LNCC
        metrics[category]['SSIM'] += ssim(gt_img, gen_img)
        metrics[category]['PSNR'] += psnr(gt_img, gen_img)
        metrics[category]['LNCC'] += calculate_lncc(gt_img, gen_img)
        metrics[category]['count'] += 1

    # 计算每个类别的平均指标
    for category in metrics:
        if metrics[category]['count'] > 0:
            metrics[category]['SSIM'] /= metrics[category]['count']
            metrics[category]['PSNR'] /= metrics[category]['count']
            metrics[category]['LNCC'] /= metrics[category]['count']

    return metrics

# 结果输出
def print_metrics(metrics):
    avg_ssim = 0
    avg_psnr = 0
    avg_lncc = 0
    for category, values in metrics.items():
        print(f"Class: {idx2label[int(category)]}")
        print(f"  Average SSIM: {values['SSIM']}")
        print(f"  Average PSNR: {values['PSNR']}")
        print(f"  Average LNCC: {values['LNCC']}")
        print()
        avg_ssim += values['SSIM']
        avg_psnr += values['PSNR']
        avg_lncc += values['LNCC']
    print(f"Average")
    print(f"  Average SSIM: {avg_ssim/len(metrics)}")
    print(f"  Average PSNR: {avg_psnr/len(metrics)}")
    print(f"  Average LNCC: {avg_lncc/len(metrics)}")
    print()

# 主程序
gt_folder = "/home/user9/workspace/course/DCD-GAN/code/datasets/testB"  # 替换为GT文件夹路径
generated_folder = "/home/user9/workspace/course/DCD-GAN/code/results/dcd_mtl/dcd_mtl/test_latest/images/fake"  # 替换为生成结果文件夹路径
# generated_folder = "/home/user9/workspace/course/DCD-GAN/code/datasets/testA"  # 替换为生成结果文件夹路径
label_file = "/home/user9/workspace/course/DCD-GAN/code/datasets/label_inform.json"  # 替换为存储类别标签的JSON文件路径

# 计算指标
metrics = compute_metrics(gt_folder, generated_folder, label_file)

# 输出结果
print_metrics(metrics)
