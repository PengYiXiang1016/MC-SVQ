
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
import wandb
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import lpips
from torchvision.models import vgg16
import glob
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from collections import Counter
import gc


from train import (
    FixedSemanticCodebook,
    ImprovedSemanticEncoder,
    ImprovedSemanticDecoder,
    PatchDiscriminator,
    FixedSemanticVQGAN,
    COCOImageDataset,
    CaptionGenerator,
    save_samples,
    calculate_psnr,
    calculate_ssim
)


def load_fixed_test_images(image_paths: List[str], transform, device):

    test_images = []
    valid_paths = []
    
    for path in image_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                test_images.append(img_tensor)
                valid_paths.append(path)
                print(f" Loaded test image: {os.path.basename(path)}")
            except Exception as e:
                print(f" Failed to load {path}: {e}")
        else:
            print(f" Test image not found: {path}")
    
    if test_images:
 
        test_images = torch.cat(test_images, dim=0)
        print(f"\n📸 Loaded {len(valid_paths)} fixed test images for monitoring")
    else:
        test_images = None
        print("\n⚠  No valid test images loaded")
    
    return test_images, valid_paths


def train_fixed_semantic_vqgan_resume(config: dict, resume_from: Optional[str] = None):
 
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    os.makedirs(config['output_dir'], exist_ok=True)
    
  
    caption_generator = CaptionGenerator()
    

    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    

    fixed_test_images = None
    fixed_test_paths = []
    if 'fixed_test_images' in config['data'] and config['data']['fixed_test_images']:
        fixed_test_images, fixed_test_paths = load_fixed_test_images(
            config['data']['fixed_test_images'], 
            transform, 
            device
        )
    
    dataset = COCOImageDataset(
        root_dir=config['data']['train_dir'],
        transform=transform,
        max_samples=config['data'].get('max_samples')
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    

    model = FixedSemanticVQGAN(config).to(device)
    

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print("\nCodebook dimensions:")
    for name, cb_config in config['model']['codebooks'].items():
        print(f"  {name}: {cb_config.get('embedding_dim', 256)}")
    

    g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.codebooks.parameters())
    g_optimizer = torch.optim.AdamW(g_params, lr=config['training']['learning_rate'], betas=(0.5, 0.999))
    
    d_optimizer = None
    if model.use_gan:
        d_optimizer = torch.optim.AdamW(
            model.discriminator.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.5, 0.999)
        )
    

    g_scheduler = CosineAnnealingWarmRestarts(g_optimizer, T_0=len(dataloader), T_mult=2)
    

    start_epoch = 0
    global_step = 0
    best_loss = float('inf')
    

    if resume_from and os.path.exists(resume_from):
        print(f"\n Resuming training from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        

        model_state = checkpoint['model_state_dict']
        current_state = model.state_dict()
        
        matched_keys = []
        missing_keys = []
        
        for key in current_state.keys():
            if key in model_state and current_state[key].shape == model_state[key].shape:
                matched_keys.append(key)
            else:
                missing_keys.append(key)
        
        filtered_state = {k: v for k, v in model_state.items() if k in matched_keys}
        model.load_state_dict(filtered_state, strict=False)
        
        print(f" Loaded {len(matched_keys)} matching keys")
        
        if missing_keys:
            print(f"⚠  Missing keys: {len(missing_keys)}")
            if any('discriminator' in key for key in missing_keys):
                print("📌 Note: Discriminator weights missing - will initialize randomly")
        
   
        if 'optimizer_state_dict' in checkpoint:
            try:
                g_optimizer.load_state_dict(checkpoint['optimizer_state_dict']['g_optimizer'])
                print(" Generator optimizer state loaded")
            except:
                print("  Using fresh generator optimizer")
        
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            global_step = checkpoint.get('global_step', 0)
        if 'best_loss' in checkpoint:
            best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"\n Resuming from epoch {start_epoch}, global step {global_step}")
        
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    

    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss_fn.eval()
    for param in perceptual_loss_fn.parameters():
        param.requires_grad = False
    

    caption_print_interval = config['training'].get('caption_print_interval', 200)
    
    print(f"\n Starting training from epoch {start_epoch}")
    print(f"   GAN training: {'Enabled' if model.use_gan else 'Disabled'}")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        epoch_losses = []
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}") as pbar:
            for batch_idx, (images, captions) in enumerate(pbar):
                images = images.to(device)
                
            
                return_tokens = (global_step % caption_print_interval == 0)
                
    
                train_gan = model.use_gan and d_optimizer and global_step > config['training'].get('gan_start_step', 5000)
                
     
                if train_gan and global_step % config['training']['d_steps_per_g'] == 0:
             
                    with torch.no_grad():
                        fake_images, _ = model(images, return_tokens=False)
                        fake_images = fake_images.detach()
                    
              
                    d_optimizer.zero_grad()
                    
                    real_pred = model.discriminator(images)
                    fake_pred = model.discriminator(fake_images)
                    
                    d_loss = F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()
                    d_loss.backward()
                    d_optimizer.step()
                
          
       
                reconstructed, info = model(images, return_tokens=return_tokens)
                
    
                recon_loss = F.mse_loss(reconstructed, images) * config['training']['loss_weights']['reconstruction']
                
           
                with torch.no_grad():
                    perceptual_loss_value = perceptual_loss_fn(reconstructed, images).mean().item()
                perceptual_loss = perceptual_loss_value * config['training']['loss_weights']['perceptual']
                
          
                vq_loss = sum(info['vq_losses'].values()) * config['training']['loss_weights']['vq']
                
      
                gan_loss_value = 0.0
                if train_gan:
                   
                    fake_pred_g = model.discriminator(reconstructed)
                    gan_loss = -fake_pred_g.mean() * config['training']['loss_weights']['gan']
                    gan_loss_value = gan_loss.item()
                    
                  
                    g_loss = recon_loss + vq_loss + perceptual_loss + gan_loss
                else:
                
                    g_loss = recon_loss + vq_loss + perceptual_loss
                
         
                g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                
        
                torch.nn.utils.clip_grad_norm_(g_params, config['training']['gradient_clip'])
                
                g_optimizer.step()
                g_scheduler.step()
                
            
                total_loss = g_loss.item()
                epoch_losses.append(total_loss)
                
    
                with torch.no_grad():
                    pbar.set_postfix({
                        'loss': f"{total_loss:.2f}",
                        'recon': f"{F.mse_loss(reconstructed, images).item():.3f}",
                        'gan': f"{gan_loss_value:.3f}",
                        'lr': f"{g_optimizer.param_groups[0]['lr']:.2e}"
                    })
                
      
                if return_tokens and info['vq_info']['global'].get('tokens'):
                    with torch.no_grad():
                        print(f"\n{'='*80}")
                        print(f"[Step {global_step}] Caption Comparison:")
                        print(f"{'='*80}")
                        
                        idx = np.random.randint(0, len(captions))
                        
                        print(f"\n Real Caption:")
                        print(f"   {captions[idx]}")
                        
                        tokens_dict = {}
                        for codebook_name in ['global', 'object', 'color', 'spatial', 'texture']:
                            if codebook_name in info['vq_info'] and info['vq_info'][codebook_name].get('tokens'):
                                tokens_dict[codebook_name] = [info['vq_info'][codebook_name]['tokens'][idx]]
                        
                        generated_caption = caption_generator.tokens_to_caption(tokens_dict)
                        print(f"\n Generated Caption from Tokens:")
                        print(f"   {generated_caption}")
                        print(f"{'='*80}\n")
                
      
                if global_step % config['training']['sample_interval'] == 0:
                    with torch.no_grad():
                  
                        print(f"\n Step {global_step} - Codebook Usage:")
                        for name, codebook in model.codebooks.items():
                            usage = codebook._get_usage_ratio()
                            perplexity = info['vq_info'][name]['perplexity'].item()
                            print(f"  {name}: {usage:.2%} (perplexity: {perplexity:.1f})")
                   
                        if fixed_test_images is not None:
                     
                            model.eval()
                            test_reconstructed, _ = model(fixed_test_images, return_tokens=False)
                            model.train()
                            
                        
                            save_samples(model, fixed_test_images, test_reconstructed, 
                                       os.path.join(config['output_dir'], f'fixed_samples_step_{global_step}.png'))
                            
                      
                            psnr = calculate_psnr(fixed_test_images, test_reconstructed)
                            ssim = calculate_ssim(fixed_test_images, test_reconstructed)
                            print(f"   Fixed Test Images - PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")
                        else:
                       
                            save_samples(model, images[:8], reconstructed[:8], 
                                       os.path.join(config['output_dir'], f'samples_step_{global_step}.png'))
                    
                            psnr = calculate_psnr(images, reconstructed)
                            ssim = calculate_ssim(images, reconstructed)
                            print(f"   PSNR: {psnr:.2f}, SSIM: {ssim:.3f}\n")
                
                global_step += 1
                
         
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        

        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\n Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f" New best loss: {best_loss:.4f}")
        

        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
         
            codebook_stats = {}
            for name, codebook in model.codebooks.items():
                codebook_stats[name] = {
                    'usage_count': codebook.usage_count.cpu(),
                    'total_count': codebook.total_count.cpu()
                }
            
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'best_loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {
                    'g_optimizer': g_optimizer.state_dict(),
                },
                'scheduler_state_dict': g_scheduler.state_dict(),
                'codebook_stats': codebook_stats,
                'config': config
            }
            
            if model.use_gan and d_optimizer:
                checkpoint['optimizer_state_dict']['d_optimizer'] = d_optimizer.state_dict()
            
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
    
            if avg_epoch_loss == best_loss:
                best_path = os.path.join(config['output_dir'], 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f" Saved best model to {best_path}")
            
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f" Saved checkpoint for epoch {epoch+1}")


if __name__ == "__main__":

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
            'train_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data',
            'image_size': 128,
            'num_workers': 4,
            'max_samples': None,
           
            'fixed_test_images': [
                '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data/000000000072.jpg',
                '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data/000000581118.jpg',
                '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data/000000576981.jpg',
                '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data/000000495079.jpg',
               
            ]
        },
        'training': {
            'batch_size': 16,
            'num_epochs': 200,
            'learning_rate': 0.0001,
            'use_gan': True,
            'gan_loss_type': 'standard',
            'gan_start_step': 5000, 
            'd_steps_per_g': 1,
            'gradient_clip': 1.0,
            'loss_weights': {
                'reconstruction': 1.0,
                'perceptual': 0.1,
                'vq': 1.0,
                'gan': 0.1
            },
            'sample_interval': 500,
            'checkpoint_interval': 5,
            'caption_print_interval': 200
        },
        'output_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoints_custom'
    }
    
  
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
 
    # resume_checkpoint = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoint_epoch_45.pt"
    resume_checkpoint = None
    

    train_fixed_semantic_vqgan_resume(config, resume_from=resume_checkpoint)
