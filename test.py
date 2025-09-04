
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


from train import (
    FixedSemanticVQGAN, 
    COCOImageDataset,
    CaptionGenerator
)


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



class MetricsCalculator:

    
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
  
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
        return psnr.cpu().item() 
    
    def calculate_ssim(self, img1, img2):

        if TORCHMETRICS_AVAILABLE:
            ssim_value = self.ssim_fn(img1, img2)
            return ssim_value.cpu().item()
        else:
        
            return self._simple_ssim(img1, img2)
    
    def _simple_ssim(self, img1, img2):
     
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
        
        return ssim_map.mean().cpu().item() 
    
    def calculate_lpips(self, img1, img2):

        with torch.no_grad():
            lpips_value = self.lpips_fn(img1, img2).mean()
            return lpips_value.cpu().item() 
    
    def extract_inception_features(self, images, batch_size=32):
       
        features = []
        
      
        resize_transform = transforms.Resize((299, 299), antialias=True)
        
  
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
              
                batch = (batch + 1) / 2
                
          
                batch_resized = torch.stack([resize_transform(img) for img in batch])
                
            
                batch_normalized = torch.stack([normalize(img) for img in batch_resized])
                
            
                feat = self.inception_model(batch_normalized)
                
                features.append(feat.cpu())
        
        return torch.cat(features, dim=0).numpy()
    
    def calculate_fid(self, real_features, fake_features):

        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
  
        diff = mu1 - mu2
        
     
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
  
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)  
    
    def calculate_inception_score(self, images, batch_size=32, splits=10):
        """计算Inception Score"""
      
        inception_full = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(self.device)
        inception_full.eval()
   
        preds = []
        
      
        resize_transform = transforms.Resize((299, 299), antialias=True)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(self.device)
                
         
                batch = (batch + 1) / 2
                
          
                batch_resized = torch.stack([resize_transform(img) for img in batch])
                batch_normalized = torch.stack([normalize(img) for img in batch_resized])
                
           
                logits = inception_full(batch_normalized)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
         
                preds.append(F.softmax(logits, dim=1).cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        

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
   
        if not CLIP_AVAILABLE:
            return None
        
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size].to(self.device)
                batch_captions = captions[i:i+batch_size]
                
            
                batch_images = (batch_images + 1) / 2
                
               
                batch_images_resized = F.interpolate(batch_images, size=(224, 224), mode='bilinear', align_corners=False)
                
               
                image_features = self.clip_model.encode_image(batch_images_resized)
                text_tokens = clip.tokenize(batch_captions, truncate=True).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                
           
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            
                similarity = (image_features * text_features).sum(dim=-1)
                scores.extend(similarity.cpu().numpy())
        
        return float(np.mean(scores))



def test_model(checkpoint_path, config, test_data_dir, num_test_samples=5000):
  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
   
    if 'config' in checkpoint:
        config = checkpoint['config']
    
  
    model = FixedSemanticVQGAN(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(" Model loaded successfully")
    

    caption_generator = CaptionGenerator()
    
   
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

    metrics_calculator = MetricsCalculator(device)
    

    all_metrics = defaultdict(list)
    real_images_list = []
    fake_images_list = []
    captions_list = []
    generated_captions_list = []
    

    print("\nEvaluating model...")
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            

            reconstructed, info = model(images, return_tokens=True)

            real_images_list.append(images.cpu())
            fake_images_list.append(reconstructed.cpu())
            captions_list.extend(captions)
            
    
            if 'vq_info' in info:
                for i in range(len(images)):
                    tokens_dict = {}
                    for codebook_name in ['global', 'object', 'color', 'spatial', 'texture']:
                        if codebook_name in info['vq_info'] and info['vq_info'][codebook_name].get('tokens'):
                            tokens_dict[codebook_name] = [info['vq_info'][codebook_name]['tokens'][i]]
                    generated_caption = caption_generator.tokens_to_caption(tokens_dict)
                    generated_captions_list.append(generated_caption)
         
            batch_psnr = metrics_calculator.calculate_psnr(images, reconstructed)
            batch_ssim = metrics_calculator.calculate_ssim(images, reconstructed)
            batch_lpips = metrics_calculator.calculate_lpips(images, reconstructed)
            
            all_metrics['psnr'].append(batch_psnr)
            all_metrics['ssim'].append(batch_ssim)
            all_metrics['lpips'].append(batch_lpips)
            
          
            for name, codebook in model.codebooks.items():
                usage = codebook._get_usage_ratio()
                if isinstance(usage, torch.Tensor):
                    usage = usage.cpu().item()
                all_metrics[f'{name}_usage'].append(float(usage))
            
        
            if batch_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    

    real_images = torch.cat(real_images_list, dim=0)
    fake_images = torch.cat(fake_images_list, dim=0)
    
    print("\nCalculating advanced metrics...")
    
   
    print("  Computing FID...")
    real_features = metrics_calculator.extract_inception_features(real_images)
    fake_features = metrics_calculator.extract_inception_features(fake_images)
    fid_score = metrics_calculator.calculate_fid(real_features, fake_features)
    
   
    print("  Computing Inception Score...")
    is_mean, is_std = metrics_calculator.calculate_inception_score(fake_images)
    
  
    clip_score_real = None
    clip_score_generated = None
    if CLIP_AVAILABLE:
        print("  Computing CLIP Scores...")
        # CLIP Score with real captions
        clip_score_real = metrics_calculator.calculate_clip_score(fake_images, captions_list)
        # CLIP Score with generated captions
        if generated_captions_list:
            clip_score_generated = metrics_calculator.calculate_clip_score(fake_images, generated_captions_list)
    

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
    

    for name in ['global', 'object', 'color', 'spatial', 'texture']:
        if f'{name}_usage' in all_metrics:
            results[f'{name}_codebook_usage'] = float(np.mean(all_metrics[f'{name}_usage']))
    
    return results



def save_results(results, save_path):
 

    df = pd.DataFrame([results])
    

    csv_path = save_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    

    print("\n" + "="*80)
    print(" TEST RESULTS")
    print("="*80)
    

    print("\n  Image Quality Metrics:")
    print(f"  PSNR:  {results['PSNR']:.2f} dB")
    print(f"  SSIM:  {results['SSIM']:.4f}")
    print(f"  LPIPS: {results['LPIPS']:.4f} (lower is better)")
    print(f"  FID:   {results['FID']:.2f} (lower is better)")
    

    print(f"\n Inception Score:")
    print(f"  Mean: {results['IS_mean']:.2f}")
    print(f"  Std:  {results['IS_std']:.2f}")
    

    if results['CLIP_Score_Real'] is not None:
        print(f"\n CLIP Scores:")
        print(f"  With Real Captions:      {results['CLIP_Score_Real']:.4f}")
        if results['CLIP_Score_Generated'] is not None:
            print(f"  With Generated Captions: {results['CLIP_Score_Generated']:.4f}")
    

    print(f"\n Codebook Usage:")
    for name in ['global', 'object', 'color', 'spatial', 'texture']:
        key = f'{name}_codebook_usage'
        if key in results:
            print(f"  {name}: {results[key]:.2%}")
    
    print("\n" + "="*80)
    print(f"\n Results saved to: {save_path}")


if __name__ == "__main__":

    checkpoint_path = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoint_epoch_45.pt"
    test_data_dir = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data"
    
 
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
    

    results = test_model(
        checkpoint_path=checkpoint_path,
        config=config,
        test_data_dir=test_data_dir,
        num_test_samples=1000
    )
    


    output_dir = '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs'
    save_path = os.path.join(output_dir, "test_results.json")
    save_results(results, save_path)
    

    print("\n Generating visualization samples...")
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
    
    print("Test completed!")
