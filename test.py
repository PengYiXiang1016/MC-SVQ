
    
# test_semantic_vqgan.py - 完全修复版本

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import lpips
from scipy import linalg
from torchvision.models import inception_v3, Inception_V3_Weights
import glob
import gc
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*antialias.*')

# 导入模型相关的类
from train import (
    FixedSemanticVQGAN, 
    COCOImageDataset,
    CaptionGenerator
)

# 尝试导入额外的评估库
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
    CLIP_AVAILABLE = False

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.fid import FrechetInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    print("torchmetrics not available. Some metrics will use alternative implementations.")
    TORCHMETRICS_AVAILABLE = False

# ================== 评估指标计算 ==================

class MetricsCalculator:
    """计算各种评估指标"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # LPIPS
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.lpips_fn.eval()
        
        # SSIM
        if TORCHMETRICS_AVAILABLE:
            self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        
        # Inception模型（用于FID和IS）
        self.inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
        self.inception_model.eval()
        
        # 修改Inception模型以提取特征
        self.inception_model.fc = nn.Identity()
        
        # CLIP模型（如果可用）
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/ViT-L-14.pt", device=device)
            self.clip_model.eval()
    
    def calculate_psnr(self, img1, img2):
        """计算PSNR - 返回CPU上的值"""
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
        return psnr.cpu().item()  # 转移到CPU并获取标量值
    
    def calculate_ssim(self, img1, img2):
        """计算SSIM - 返回CPU上的值"""
        if TORCHMETRICS_AVAILABLE:
            ssim_value = self.ssim_fn(img1, img2)
            return ssim_value.cpu().item()  # 转移到CPU并获取标量值
        else:
            # 简化版SSIM
            return self._simple_ssim(img1, img2)
    
    def _simple_ssim(self, img1, img2):
        """简化版SSIM计算"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.avg_pool2d(img1, 3, 1, 1)
        mu2 = F.avg_pool2d(img2, 3, 1, 1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, 3, 1, 1) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, 3, 1, 1) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean().cpu().item()  # 转移到CPU
    
    def calculate_lpips(self, img1, img2):
        """计算LPIPS - 返回CPU上的值"""
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1, img2).mean()
            return lpips_value.cpu().item()  # 转移到CPU
    
    def extract_inception_features(self, images, batch_size=32):
        """提取Inception特征（用于FID）"""
        features = []
        
        # 调整图像大小到299x299（添加antialias参数）
        resize_transform = transforms.Resize((299, 299), antialias=True)
        
        # Inception v3的标准预处理
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
                # 反归一化到[0,1]（从[-1,1]）
                batch = (batch + 1) / 2
                
                # 调整大小
                batch_resized = torch.stack([resize_transform(img) for img in batch])
                
                # 应用Inception的标准化
                batch_normalized = torch.stack([normalize(img) for img in batch_resized])
                
                # 提取特征
                feat = self.inception_model(batch_normalized)
                
                features.append(feat.cpu())  # 保存在CPU上
        
        return torch.cat(features, dim=0).numpy()
    
    def calculate_fid(self, real_features, fake_features):
        """计算FID"""
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        # 计算Frechet距离
        diff = mu1 - mu2
        
        # 计算协方差矩阵的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 数值稳定性检查
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)  # 确保返回Python float
    
    def calculate_inception_score(self, images, batch_size=32, splits=10):
        """计算Inception Score"""
        # 需要使用完整的Inception模型（带FC层）
        inception_full = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(self.device)
        inception_full.eval()
        
        # 获取预测
        preds = []
        
        # 调整图像大小（添加antialias参数）
        resize_transform = transforms.Resize((299, 299), antialias=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
                # 反归一化
                batch = (batch + 1) / 2
                
                # 调整大小和标准化
                batch_resized = torch.stack([resize_transform(img) for img in batch])
                batch_normalized = torch.stack([normalize(img) for img in batch_resized])
                
                # 获取logits
                logits = inception_full(batch_normalized)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # 转换为概率
                preds.append(F.softmax(logits, dim=1).cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        
        # 计算IS
        split_scores = []
        N = len(preds)
        for k in range(splits):
            start = k * (N // splits)
            end = (k + 1) * (N // splits) if k < splits - 1 else N
            part = preds[start:end]
            
            if len(part) == 0:
                continue
                
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(np.sum(pyx * (np.log(pyx + 1e-8) - np.log(py + 1e-8))))
            split_scores.append(np.exp(np.mean(scores)))
        
        return float(np.mean(split_scores)), float(np.std(split_scores))
    
    def calculate_clip_score(self, images, captions, batch_size=32):
        """计算CLIP Score"""
        if not CLIP_AVAILABLE:
            return None
        
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size].to(self.device)
                batch_captions = captions[i:i+batch_size]
                
                # 反归一化图像
                batch_images = (batch_images + 1) / 2
                
                # 调整图像大小以适应CLIP
                batch_images_resized = F.interpolate(batch_images, size=(224, 224), mode='bilinear', align_corners=False)
                
                # 编码图像和文本
                image_features = self.clip_model.encode_image(batch_images_resized)
                text_tokens = clip.tokenize(batch_captions, truncate=True).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                
                # 归一化特征
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 计算余弦相似度
                similarity = (image_features * text_features).sum(dim=-1)
                scores.extend(similarity.cpu().numpy())
        
        return float(np.mean(scores))

# ================== 测试函数 ==================

def test_model(checkpoint_path, config, test_data_dir, num_test_samples=5000):
    """测试模型并计算所有指标"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 如果config在checkpoint中，使用它
    if 'config' in checkpoint:
        config = checkpoint['config']
    
    # 创建模型
    model = FixedSemanticVQGAN(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # 创建Caption生成器
    caption_generator = CaptionGenerator()
    
    # 准备数据
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_dataset = COCOImageDataset(
        root_dir=test_data_dir,
        transform=transform,
        max_samples=num_test_samples
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTest dataset size: {len(test_dataset)}")
    
    # 初始化指标计算器
    metrics_calculator = MetricsCalculator(device)
    
    # 存储结果 - 确保所有值都是标量
    all_metrics = defaultdict(list)
    real_images_list = []
    fake_images_list = []
    captions_list = []
    generated_captions_list = []
    
    # 测试循环
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            
            # 生成重建图像
            reconstructed, info = model(images, return_tokens=True)
            
            # 收集图像用于FID和IS（保持在CPU上以节省GPU内存）
            real_images_list.append(images.cpu())
            fake_images_list.append(reconstructed.cpu())
            captions_list.extend(captions)
            
            # 生成captions（从tokens）
            if 'vq_info' in info:
                for i in range(len(images)):
                    tokens_dict = {}
                    for codebook_name in ['global', 'object', 'color', 'spatial', 'texture']:
                        if codebook_name in info['vq_info'] and info['vq_info'][codebook_name].get('tokens'):
                            tokens_dict[codebook_name] = [info['vq_info'][codebook_name]['tokens'][i]]
                    generated_caption = caption_generator.tokens_to_caption(tokens_dict)
                    generated_captions_list.append(generated_caption)
            
            # 计算批次指标 - 确保返回的是标量值
            batch_psnr = metrics_calculator.calculate_psnr(images, reconstructed)
            batch_ssim = metrics_calculator.calculate_ssim(images, reconstructed)
            batch_lpips = metrics_calculator.calculate_lpips(images, reconstructed)
            
            all_metrics['psnr'].append(batch_psnr)
            all_metrics['ssim'].append(batch_ssim)
            all_metrics['lpips'].append(batch_lpips)
            
            # 码本使用统计 - 确保是标量值
            for name, codebook in model.codebooks.items():
                usage = codebook._get_usage_ratio()
                if isinstance(usage, torch.Tensor):
                    usage = usage.cpu().item()
                all_metrics[f'{name}_usage'].append(float(usage))
            
            # 定期清理内存
            if batch_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    
    # 合并所有图像
    real_images = torch.cat(real_images_list, dim=0)
    fake_images = torch.cat(fake_images_list, dim=0)
    
    print("\nCalculating advanced metrics...")
    
    # 计算FID
    print("  Computing FID...")
    real_features = metrics_calculator.extract_inception_features(real_images)
    fake_features = metrics_calculator.extract_inception_features(fake_images)
    fid_score = metrics_calculator.calculate_fid(real_features, fake_features)
    
    # 计算Inception Score
    print("  Computing Inception Score...")
    is_mean, is_std = metrics_calculator.calculate_inception_score(fake_images)
    
    # 计算CLIP Score
    clip_score_real = None
    clip_score_generated = None
    if CLIP_AVAILABLE:
        print("  Computing CLIP Scores...")
        # CLIP Score with real captions
        clip_score_real = metrics_calculator.calculate_clip_score(fake_images, captions_list)
        # CLIP Score with generated captions
        if generated_captions_list:
            clip_score_generated = metrics_calculator.calculate_clip_score(fake_images, generated_captions_list)
    
    # 汇总结果 - 确保所有值都是Python原生类型
    results = {
        'PSNR': float(np.mean(all_metrics['psnr'])),
        'SSIM': float(np.mean(all_metrics['ssim'])),
        'LPIPS': float(np.mean(all_metrics['lpips'])),
        'FID': float(fid_score),
        'IS_mean': float(is_mean),
        'IS_std': float(is_std),
        'CLIP_Score_Real': float(clip_score_real) if clip_score_real is not None else None,
        'CLIP_Score_Generated': float(clip_score_generated) if clip_score_generated is not None else None
    }
    
    # 码本使用率
    for name in ['global', 'object', 'color', 'spatial', 'texture']:
        if f'{name}_usage' in all_metrics:
            results[f'{name}_codebook_usage'] = float(np.mean(all_metrics[f'{name}_usage']))
    
    return results

# ================== 结果展示和保存 ==================

def save_results(results, save_path):
    """保存测试结果"""
    # 创建DataFrame
    df = pd.DataFrame([results])
    
    # 保存为CSV
    csv_path = save_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    
    # 保存为JSON
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # 打印结果表格
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    
    # 图像质量指标
    print("\n🖼️  Image Quality Metrics:")
    print(f"  PSNR:  {results['PSNR']:.2f} dB")
    print(f"  SSIM:  {results['SSIM']:.4f}")
    print(f"  LPIPS: {results['LPIPS']:.4f} (lower is better)")
    print(f"  FID:   {results['FID']:.2f} (lower is better)")
    
    # Inception Score
    print(f"\n🎯 Inception Score:")
    print(f"  Mean: {results['IS_mean']:.2f}")
    print(f"  Std:  {results['IS_std']:.2f}")
    
    # CLIP Scores
    if results['CLIP_Score_Real'] is not None:
        print(f"\n📝 CLIP Scores:")
        print(f"  With Real Captions:      {results['CLIP_Score_Real']:.4f}")
        if results['CLIP_Score_Generated'] is not None:
            print(f"  With Generated Captions: {results['CLIP_Score_Generated']:.4f}")
    
    # 码本使用率
    print(f"\n📚 Codebook Usage:")
    for name in ['global', 'object', 'color', 'spatial', 'texture']:
        key = f'{name}_codebook_usage'
        if key in results:
            print(f"  {name}: {results[key]:.2%}")
    
    print("\n" + "="*80)
    print(f"\n✅ Results saved to: {save_path}")

# ================== 主程序 ==================

if __name__ == "__main__":
    # 配置
    checkpoint_path = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoint_epoch_45.pt"
    test_data_dir = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data"
    
    # 模型配置
    config = {
        'model': {
            'encoder': {
                'base_channels': 64
            },
            'decoder': {
                'base_channels': 64
            },
            'discriminator_channels': 32,
            'codebooks': {
                'global': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/global',
                    'embedding_dim': 768,
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'object': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/object',
                    'embedding_dim': 512,
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'color': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/color',
                    'embedding_dim': 256,
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'spatial': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/spatial',
                    'embedding_dim': 384,
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'texture': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/texture',
                    'embedding_dim': 256,
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                }
            }
        },
        'data': {
            'image_size': 128,
            'num_workers': 4
        },
        'training': {
            'use_gan': False
        }
    }
    
    # 测试模型
    results = test_model(
        checkpoint_path=checkpoint_path,
        config=config,
        test_data_dir=test_data_dir,
        num_test_samples=1000
    )
    
    # 保存结果
    # output_dir = os.path.dirname(checkpoint_path)
    output_dir = '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs'
    save_path = os.path.join(output_dir, "test_results.json")
    save_results(results, save_path)
    
    # 生成可视化样本
    print("\n🎨 Generating visualization samples...")
    from train import save_samples
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'config' in checkpoint:
        config = checkpoint['config']
    
    model = FixedSemanticVQGAN(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    vis_dataset = COCOImageDataset(
        root_dir=test_data_dir,
        transform=transform,
        max_samples=16
    )
    
    vis_loader = DataLoader(vis_dataset, batch_size=8, shuffle=True)
    
    with torch.no_grad():
        for images, captions in vis_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            save_samples(model, images, reconstructed, 
                       os.path.join(output_dir, 'test_visualization.png'))
            break
    
    print("✅ Test completed!")