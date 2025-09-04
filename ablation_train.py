
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
    PatchDiscriminator,
    COCOImageDataset,
    CaptionGenerator,
    save_samples,
    calculate_psnr,
    calculate_ssim
)


class FlexibleSemanticDecoder(nn.Module):
  

    def __init__(self, in_channels: int, start_spatial_size: int,
                 base_channels: int = 64, target_size: int = 128):
        super().__init__()

        layers = []
        current_size = start_spatial_size
        current_channels = in_channels
        channels = base_channels

        while current_size < target_size:
            layers.append(
                nn.ConvTranspose2d(current_channels, channels, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(True))

            current_size *= 2
            current_channels = channels
            channels //= 2
            if channels < 8:
                channels = 8


        layers.append(nn.Conv2d(current_channels, 3, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)
        print(f"[Decoder] Upsample from {start_spatial_size} -> {target_size} in {len(layers)//3} steps")

    def forward(self, x):
        return self.layers(x)


class CustomizableSemanticVQGAN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        
        self.enabled_codebooks = config['model'].get('enabled_codebooks', 
                                                     ['global', 'object', 'color', 'spatial', 'texture'])
        print(f"\n Enabled codebooks: {self.enabled_codebooks}")
        
        self.total_codebook_dim = sum(
            config['model']['codebooks'][name]['embedding_dim'] 
            for name in self.enabled_codebooks
        )
        
        # 创建编码器
        self.encoder = ImprovedSemanticEncoder(
            in_channels=3,
            base_channels=config['model']['encoder']['base_channels']
        )
        
        # 获取编码器输出维度和空间尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config['data']['image_size'], config['data']['image_size'])
            encoder_output = self.encoder(dummy_input)
            if isinstance(encoder_output, dict):
                if 'features' in encoder_output:
                    h = encoder_output['features']
                elif 'h' in encoder_output:
                    h = encoder_output['h']
                elif 'output' in encoder_output:
                    h = encoder_output['output']
                else:
                    for value in encoder_output.values():
                        if isinstance(value, torch.Tensor) and value.dim() == 4:
                            h = value
                            break
            else:
                h = encoder_output
            
            self.encoder_out_dim = h.shape[1]
            self.encoder_spatial_size = h.shape[2]
            print(f"Detected encoder output: {self.encoder_out_dim} channels, "
                  f"{self.encoder_spatial_size}x{self.encoder_spatial_size} spatial")
        
    
        if self.encoder_out_dim != self.total_codebook_dim:
            self.encoder_proj = nn.Conv2d(self.encoder_out_dim, self.total_codebook_dim, kernel_size=1)
            print(f"Added encoder projection: {self.encoder_out_dim} -> {self.total_codebook_dim}")
        else:
            self.encoder_proj = nn.Identity()
        
        self.decoder = FlexibleSemanticDecoder(
            in_channels=self.total_codebook_dim,
            start_spatial_size=self.encoder_spatial_size,
            base_channels=config['model']['decoder']['base_channels'],
            target_size=config['data']['image_size']
        )
        print(f"Created decoder with input channels: {self.total_codebook_dim}")
 
        

        self.codebooks = nn.ModuleDict()
        for name in self.enabled_codebooks:
            cb_config = config['model']['codebooks'][name]
            self.codebooks[name] = FixedSemanticCodebook(
                codebook_path=cb_config['path'],
                embedding_dim=cb_config['embedding_dim'],
                commitment_cost=cb_config.get('commitment_cost', 0.25),
                decay=cb_config.get('decay', 0.99),
                epsilon=cb_config.get('epsilon', 1e-5)
            )
        
 
        self.use_gan = config['training'].get('use_gan', False)
        if self.use_gan:
            self.discriminator = PatchDiscriminator(
                in_channels=3,
                base_channels=config['model'].get('discriminator_channels', 64),
                n_layers=3
            )
        

        self.codebook_dims = {
            name: config['model']['codebooks'][name]['embedding_dim']
            for name in self.enabled_codebooks
        }

        self.split_points = []
        current_dim = 0
        for name in self.enabled_codebooks:
            self.split_points.append(current_dim)
            current_dim += self.codebook_dims[name]
        self.split_points.append(current_dim)
    
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:

        encoder_output = self.encoder(x)
        
        # 处理字典或张量输出
        if isinstance(encoder_output, dict):
        
            if 'features' in encoder_output:
                h = encoder_output['features']
            elif 'h' in encoder_output:
                h = encoder_output['h']
            elif 'output' in encoder_output:
                h = encoder_output['output']
            else:
        
                for value in encoder_output.values():
                    if isinstance(value, torch.Tensor) and value.dim() == 4:
                        h = value
                        break
        else:
            h = encoder_output
        

        h = self.encoder_proj(h)
        
        B, C, H, W = h.shape
 
        features = {}
        for i, name in enumerate(self.enabled_codebooks):
            start_dim = self.split_points[i]
            end_dim = self.split_points[i + 1]
            features[name] = h[:, start_dim:end_dim, :, :]
        
        return features
    
    def decode(self, quantized_features: Dict[str, torch.Tensor]) -> torch.Tensor:

        features_list = []
        for name in self.enabled_codebooks:
            features_list.append(quantized_features[name])
        

        combined_features = torch.cat(features_list, dim=1)
        

        reconstructed = self.decoder(combined_features)
        return reconstructed
    
    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> Tuple[torch.Tensor, dict]:
 
        encoded_features = self.encode(x)
        

        quantized_features = {}
        vq_losses = {}
        vq_info = {}
        
        for name in self.enabled_codebooks:
     
            result = self.codebooks[name](encoded_features[name])
            
 
            if isinstance(result, tuple) and len(result) == 2:
                quantized, loss_dict = result
                quantized_features[name] = quantized
                vq_losses[name] = loss_dict['loss']
                vq_info[name] = {
                    'perplexity': loss_dict.get('perplexity', torch.tensor(0.0)),
                    'usage': loss_dict.get('usage', 0.0),
                    'encoding_indices': loss_dict.get('encoding_indices', None)
                }
                if return_tokens and 'encoding_indices' in loss_dict:
                    vq_info[name]['tokens'] = loss_dict['encoding_indices'].cpu().numpy().tolist()
            else:
                
                raise ValueError(f"Unexpected return format from codebook {name}: {type(result)}")
        
   
        reconstructed = self.decode(quantized_features)
        
      
        info = {
            'vq_losses': vq_losses,
            'vq_info': vq_info
        }
        
        return reconstructed, info


class FlexibleCaptionGenerator:

    
    def __init__(self, enabled_codebooks: List[str]):
        self.enabled_codebooks = enabled_codebooks
        self.templates = {
            'global': [
                "This image shows {}",
                "A photograph of {}",
                "An image depicting {}",
                "This is {}"
            ],
            'object': [
                "featuring {} prominently",
                "with {} visible",
                "containing {}",
                "showing {}"
            ],
            'color': [
                "in {} tones",
                "with {} colors",
                "displaying {} hues",
                "colored in {}"
            ],
            'spatial': [
                "arranged {}",
                "positioned {}",
                "with {} composition",
                "laid out {}"
            ],
            'texture': [
                "with {} texture",
                "having {} surface",
                "showing {} patterns",
                "with {} finish"
            ]
        }
    
    def tokens_to_caption(self, tokens_dict: Dict[str, List[int]]) -> str:
       
        caption_parts = []
        

        for codebook_name in self.enabled_codebooks:
            if codebook_name in tokens_dict and tokens_dict[codebook_name]:
                tokens = tokens_dict[codebook_name][0] if isinstance(tokens_dict[codebook_name][0], list) else [tokens_dict[codebook_name][0]]
                
                # 基于token生成描述
                if codebook_name == 'global':
                    descriptions = ["scene " + str(t % 10) for t in tokens[:3]]
                elif codebook_name == 'object':
                    object_types = ["person", "car", "animal", "building", "tree", "furniture", "food", "device"]
                    descriptions = [object_types[t % len(object_types)] for t in tokens[:2]]
                elif codebook_name == 'color':
                    color_types = ["warm", "cool", "vibrant", "muted", "monochrome", "colorful"]
                    descriptions = [color_types[t % len(color_types)] for t in tokens[:1]]
                elif codebook_name == 'spatial':
                    spatial_types = ["centrally", "symmetrically", "diagonally", "scattered", "grouped"]
                    descriptions = [spatial_types[t % len(spatial_types)] for t in tokens[:1]]
                elif codebook_name == 'texture':
                    texture_types = ["smooth", "rough", "detailed", "simple", "complex"]
                    descriptions = [texture_types[t % len(texture_types)] for t in tokens[:1]]
    
                if codebook_name in self.templates and descriptions:
                    template = np.random.choice(self.templates[codebook_name])
                    description = " and ".join(descriptions)
                    caption_parts.append(template.format(description))

        if caption_parts:
       
            if 'global' in self.enabled_codebooks and any('This image' in part or 'A photograph' in part for part in caption_parts):
                main_parts = [p for p in caption_parts if 'This image' in p or 'A photograph' in p]
                other_parts = [p for p in caption_parts if p not in main_parts]
                caption = main_parts[0] + ", " + ", ".join(other_parts) + "."
            else:
                caption = ", ".join(caption_parts) + "."
            
            return caption
        else:
            return "An image."


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
        print(f"\n Loaded {len(valid_paths)} fixed test images for monitoring")
    else:
        test_images = None
        print("\n  No valid test images loaded")
    
    return test_images, valid_paths

def train_customizable_semantic_vqgan(config: dict, resume_from: Optional[str] = None):

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    os.makedirs(config['output_dir'], exist_ok=True)
    

    enabled_codebooks = config['model'].get('enabled_codebooks', 
                                           ['global', 'object', 'color', 'spatial', 'texture'])
    

    caption_generator = FlexibleCaptionGenerator(enabled_codebooks)
    
 
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
    

    model = CustomizableSemanticVQGAN(config).to(device)
    

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print("\nEnabled codebook dimensions:")
    for name in enabled_codebooks:
        dim = config['model']['codebooks'][name].get('embedding_dim', 256)
        print(f"  {name}: {dim}")
    print(f"Total codebook dimension: {model.total_codebook_dim}")
    print(f"Encoder output dimension: {model.encoder_out_dim}")
    

    g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + \
               list(model.codebooks.parameters())
    

    if not isinstance(model.encoder_proj, nn.Identity):
        g_params.extend(list(model.encoder_proj.parameters()))
    
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
        

        if 'config' in checkpoint:
            saved_codebooks = checkpoint['config']['model'].get('enabled_codebooks', 
                                                               ['global', 'object', 'color', 'spatial', 'texture'])
            if set(saved_codebooks) != set(enabled_codebooks):
                print(f"  WARNING: Codebook configuration mismatch!")
                print(f"   Saved: {saved_codebooks}")
                print(f"   Current: {enabled_codebooks}")
                print("   This may cause loading issues or unexpected behavior.")

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
            print(f"  Missing keys: {len(missing_keys)}")

            missing_codebooks = set()
            for key in missing_keys:
                if 'codebooks.' in key:
                    codebook_name = key.split('.')[1]
                    missing_codebooks.add(codebook_name)
            if missing_codebooks:
                print(f"   Missing codebooks: {missing_codebooks}")
        

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
    print(f"   Active codebooks: {enabled_codebooks}")
    
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
                
    
                if return_tokens and any(info['vq_info'][cb].get('tokens') for cb in enabled_codebooks if cb in info['vq_info']):
                    with torch.no_grad():
                        print(f"\n{'='*80}")
                        print(f"[Step {global_step}] Caption Comparison (Active codebooks: {enabled_codebooks}):")
                        print(f"{'='*80}")
                        
                        idx = np.random.randint(0, len(captions))
                        
                        print(f"\n Real Caption:")
                        print(f"   {captions[idx]}")
                        
                        tokens_dict = {}
                        for codebook_name in enabled_codebooks:
                            if codebook_name in info['vq_info'] and info['vq_info'][codebook_name].get('tokens'):
                                tokens_dict[codebook_name] = [info['vq_info'][codebook_name]['tokens'][idx]]
                        
                        generated_caption = caption_generator.tokens_to_caption(tokens_dict)
                        print(f"\n Generated Caption from Tokens:")
                        print(f"   {generated_caption}")
                        print(f"{'='*80}\n")

                if global_step % config['training']['sample_interval'] == 0:
                    with torch.no_grad():
                    
                        print(f"\n Step {global_step} - Codebook Usage:")
                        for name in enabled_codebooks:
                            if name in model.codebooks:
                                usage = model.codebooks[name]._get_usage_ratio()
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
            for name in enabled_codebooks:
                if name in model.codebooks:
                    codebook_stats[name] = {
                        'usage_count': model.codebooks[name].usage_count.cpu(),
                        'total_count': model.codebooks[name].total_count.cpu()
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
                'config': config,
                'enabled_codebooks': enabled_codebooks  # 保存启用的码本信息
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
      
            # 'enabled_codebooks': ['global'],  
            # 'enabled_codebooks': ['global', 'object'],  
            # 'enabled_codebooks': ['global', 'object', 'color'],  
            'enabled_codebooks': ['global', 'object', 'color', 'texture'],  
            
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
            'num_epochs': 1000,
            'learning_rate': 0.00005,
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
            'sample_interval': 5000,
            'checkpoint_interval': 100,
            'caption_print_interval': 2000
        },
        'output_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/no_spatial'
    }
    

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
  
    resume_checkpoint = None  
    # resume_checkpoint = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoint_epoch_60.pt"
    
   
    train_customizable_semantic_vqgan(config, resume_from=resume_checkpoint)
    
    
