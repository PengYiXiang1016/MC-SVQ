
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import os
from typing import Dict, List, Tuple, Optional
from torchvision import transforms
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


from train import (
    FixedSemanticCodebook,
    ImprovedSemanticEncoder,
    ImprovedSemanticDecoder,
    PatchDiscriminator,
    FixedSemanticVQGAN,
    CaptionGenerator
)

class CodebookAttentionVisualizer:

    
    def __init__(self, model: FixedSemanticVQGAN, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
     
        self.codebook_colors = {
            'global': 'Blues',
            'object': 'Reds', 
            'color': 'Greens',
            'spatial': 'Purples',
            'texture': 'Oranges'
        }
        
    def extract_codebook_features(self, image: torch.Tensor) -> Dict[str, Dict]:
  
        with torch.no_grad():
      
            reconstructed, info = self.model(image, return_tokens=True)
            
          
            encoded = self.model.encoder(image)
            
       
            codebook_info = {}
            
    
            if isinstance(encoded, dict):
              
                for name, codebook in self.model.codebooks.items():
                    if name in encoded:
                        features = encoded[name]
                    else:
                 
                        features = list(encoded.values())[0] if encoded else None
                        
                    if features is None:
                        print(f"Warning: No features found for {name} codebook")
                        continue
                        
           
                    codebook_info[name] = self._process_codebook_features(
                        features, codebook, name, info
                    )
            else:
              
                features = encoded
                
                for name, codebook in self.model.codebooks.items():
                    codebook_info[name] = self._process_codebook_features(
                        features, codebook, name, info
                    )
                
        return codebook_info
    
    def _process_codebook_features(self, features: torch.Tensor, 
                                 codebook: FixedSemanticCodebook,
                                 codebook_name: str,
                                 forward_info: Dict) -> Dict:

        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
     
        embeddings = codebook.embeddings
        
     
        if C != embeddings.shape[1]:
    
            if not hasattr(self, f'projection_{codebook_name}'):
                projection = nn.Linear(C, embeddings.shape[1]).to(self.device)
     
                nn.init.xavier_uniform_(projection.weight)
                setattr(self, f'projection_{codebook_name}', projection)
            else:
                projection = getattr(self, f'projection_{codebook_name}')
            
            with torch.no_grad():
                features_flat = projection(features_flat)
        

        distances = torch.cdist(features_flat, embeddings)
        
   
        encoding_indices = torch.argmin(distances, dim=1)
        
    
        attention_weights = F.softmax(-distances / 0.1, dim=1)  # 添加温度参数
        
    
        max_attention = torch.max(attention_weights, dim=1)[0]
        max_attention = max_attention.view(B, H, W)
        

        encoding_indices = encoding_indices.view(B, H, W)
        
  
        if 'vq_info' in forward_info and codebook_name in forward_info['vq_info']:
            vq_info = forward_info['vq_info'][codebook_name]
            if 'encodings' in vq_info:
          
                actual_encodings = vq_info['encodings']
                if actual_encodings.shape[-2:] == (H, W):
                    encoding_indices = actual_encodings
                    
        return {
            'attention_map': max_attention,
            'encoding_indices': encoding_indices,
            'distances': distances.view(B, H, W, -1),
            'features_shape': (B, C, H, W)
        }
    
    def create_attention_heatmap(self, 
                               image: torch.Tensor,
                               codebook_info: Dict[str, Dict],
                               save_path: Optional[str] = None) -> plt.Figure:
     
        image_np = image.squeeze(0).cpu().numpy()
        image_np = (image_np * 0.5 + 0.5).transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.2)
        
 
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(image_np)
        ax_orig.set_title('Original Image', fontsize=14, fontweight='bold')
        ax_orig.axis('off')
        
   
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for idx, (name, info) in enumerate(codebook_info.items()):
            if idx < len(positions):
                row, col = positions[idx]
                ax = fig.add_subplot(gs[row, col])
                
               
                attention_map = info['attention_map'].squeeze(0).cpu().numpy()
                
        
                _, _, feat_h, feat_w = info['features_shape']
                
              
                if attention_map.shape != (image_np.shape[0], image_np.shape[1]):
                    attention_map_resized = F.interpolate(
                        torch.tensor(attention_map).unsqueeze(0).unsqueeze(0),
                        size=(image_np.shape[0], image_np.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().numpy()
                else:
                    attention_map_resized = attention_map
                
          
                im = ax.imshow(attention_map_resized, 
                             cmap=self.codebook_colors[name],
                             alpha=0.8)
                
                
                ax.imshow(image_np, alpha=0.3)
                
                ax.set_title(f'{name.capitalize()} Codebook Attention\n(Feature size: {feat_h}×{feat_w})', 
                           fontsize=12, fontweight='bold')
                ax.axis('off')
                
             
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
        

        ax_combined = fig.add_subplot(gs[2, :])
        
       
        combined_map = np.zeros((image_np.shape[0], image_np.shape[1], 3))
        
        color_channels = {
            'global': [0, 0, 1],    
            'object': [1, 0, 0],    
            'color': [0, 1, 0],     
            'spatial': [1, 0, 1],   
            'texture': [1, 0.5, 0]  
        }
        
        for name, info in codebook_info.items():
            attention_map = info['attention_map'].squeeze(0).cpu().numpy()
            
       
            if attention_map.shape != (image_np.shape[0], image_np.shape[1]):
                attention_map_resized = F.interpolate(
                    torch.tensor(attention_map).unsqueeze(0).unsqueeze(0),
                    size=(image_np.shape[0], image_np.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
            else:
                attention_map_resized = attention_map
            
      
            if attention_map_resized.max() > 0:
                attention_map_resized = attention_map_resized / attention_map_resized.max()
            
            
            color = color_channels.get(name, [1, 1, 1])
            for i in range(3):
                combined_map[:, :, i] += attention_map_resized * color[i] * 0.3
        

        combined_map = np.clip(combined_map, 0, 1)
        
  
        ax_combined.imshow(image_np)
        ax_combined.imshow(combined_map, alpha=0.6)
        ax_combined.set_title('Combined Codebook Attention Map', 
                            fontsize=14, fontweight='bold')
        ax_combined.axis('off')
        
   
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_channels[name], 
                               label=name.capitalize(), 
                               alpha=0.6) 
                         for name in codebook_info.keys()]
        ax_combined.legend(handles=legend_elements, 
                          loc='center left', 
                          bbox_to_anchor=(1, 0.5),
                          fontsize=10)
        
        plt.suptitle('Multi-Codebook Semantic Attention Analysis', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Saved attention heatmap to: {save_path}")
        
        return fig
    
    def analyze_codebook_usage(self, 
                             codebook_info: Dict[str, Dict]) -> Dict[str, Dict]:
 
        usage_stats = {}
        
        for name, info in codebook_info.items():
            indices = info['encoding_indices'].cpu().numpy().flatten()
            unique_indices, counts = np.unique(indices, return_counts=True)
            
            usage_stats[name] = {
                'num_unique_codes': len(unique_indices),
                'most_used_codes': unique_indices[np.argsort(counts)[-5:]][::-1],
                'usage_distribution': counts / counts.sum()
            }
            
        return usage_stats


def load_model_from_checkpoint(checkpoint_path: str, config: dict, device: torch.device):
 
    model = FixedSemanticVQGAN(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f" Model loaded from: {checkpoint_path}")
    return model


def test_single_image(image_path: str, 
                     model: FixedSemanticVQGAN,
                     output_dir: str,
                     device: torch.device):

    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    

    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"\n  Processing image: {os.path.basename(image_path)}")
    

    visualizer = CodebookAttentionVisualizer(model, device)

    print(" Extracting codebook features...")
    codebook_info = visualizer.extract_codebook_features(image_tensor)
    

    print(" Creating attention heatmap...")
    output_path = os.path.join(output_dir, 
                              f'attention_heatmap_{os.path.basename(image_path)}')
    fig = visualizer.create_attention_heatmap(image_tensor, 
                                            codebook_info, 
                                            save_path=output_path)
    plt.show()
    

    print("\n Codebook usage statistics:")
    usage_stats = visualizer.analyze_codebook_usage(codebook_info)
    for name, stats in usage_stats.items():
        print(f"\n{name.capitalize()} codebook:")
        print(f"  - Unique codes used: {stats['num_unique_codes']}")
        print(f"  - Most used codes: {stats['most_used_codes']}")
    

    print("\n Generating reconstruction...")
    with torch.no_grad():
        reconstructed, info = model(image_tensor, return_tokens=True)
    

    fig_recon = plt.figure(figsize=(10, 5))
    

    ax1 = fig_recon.add_subplot(1, 2, 1)
    image_np = image_tensor.squeeze(0).cpu().numpy()
    image_np = (image_np * 0.5 + 0.5).transpose(1, 2, 0)
    ax1.imshow(np.clip(image_np, 0, 1))
    ax1.set_title('Original', fontsize=12)
    ax1.axis('off')
    

    ax2 = fig_recon.add_subplot(1, 2, 2)
    recon_np = reconstructed.squeeze(0).cpu().numpy()
    recon_np = (recon_np * 0.5 + 0.5).transpose(1, 2, 0)
    ax2.imshow(np.clip(recon_np, 0, 1))
    ax2.set_title('Reconstruction', fontsize=12)
    ax2.axis('off')
    
    plt.suptitle('Original vs Reconstruction', fontsize=14)
    recon_path = os.path.join(output_dir, 
                             f'reconstruction_{os.path.basename(image_path)}')
    plt.savefig(recon_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f" Reconstruction saved to: {recon_path}")
    

    if 'vq_info' in info:
        caption_generator = CaptionGenerator()
        tokens_dict = {}
        
        for codebook_name in ['global', 'object', 'color', 'spatial', 'texture']:
            if codebook_name in info['vq_info'] and info['vq_info'][codebook_name].get('tokens'):
                tokens_dict[codebook_name] = info['vq_info'][codebook_name]['tokens']
        
        if tokens_dict:
            generated_caption = caption_generator.tokens_to_caption(tokens_dict)
            print(f"\n Generated Caption from Tokens:")
            print(f"   {generated_caption}")


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
            'image_size': 128
        },
        'training': {
            'use_gan': True  
        }
    }
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
 
    checkpoint_path = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoint_epoch_60.pt"
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    

    test_images = [
        '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data/000000581881.jpg',

    ]
    

    output_dir = '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/attention'
    

    for image_path in test_images:
        if os.path.exists(image_path):
            test_single_image(image_path, model, output_dir, device)
        else:
            print(f" Image not found: {image_path}")
