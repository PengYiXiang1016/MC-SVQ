
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


class FixedSemanticCodebook(nn.Module):

    
    def __init__(self,
                 codebook_path: str,
                 embedding_dim: int = 256,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 epsilon: float = 1e-5,
                 device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        
   
        if codebook_path.endswith('.npy'):
            embeddings_path = codebook_path
            vocab_path = codebook_path.replace('embeddings.npy', 'vocab.json')
        else:
            embeddings_path = os.path.join(codebook_path, 'embeddings.npy')
            vocab_path = os.path.join(codebook_path, 'vocab.json')
        

        clip_embeddings = np.load(embeddings_path).astype(np.float32)
        
  
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.num_embeddings = len(self.vocab)
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        print(f"Loaded codebook from {codebook_path}")
        print(f"  Original shape: {clip_embeddings.shape}")
        print(f"  Vocab size: {self.num_embeddings}")
        print(f"  Target embedding dim: {self.embedding_dim}")
        

        clip_dim = clip_embeddings.shape[1]
        self.projection = nn.Linear(clip_dim, embedding_dim)
        

        self.embeddings = nn.Parameter(torch.randn(self.num_embeddings, embedding_dim))
        

        self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings))
        self.register_buffer('ema_embeddings', torch.randn(self.num_embeddings, embedding_dim))
        self.decay = decay
        self.epsilon = epsilon
        
      
        self.register_buffer('usage_count', torch.zeros(self.num_embeddings))
        self.register_buffer('total_count', torch.tensor(0))
        
    
        self.register_buffer('_clip_embeddings_cache', torch.from_numpy(clip_embeddings))
        self._initialized = False
    
    def _lazy_init(self):
    
        if not self._initialized:
            with torch.no_grad():
            
                projected = self.projection(self._clip_embeddings_cache)
                projected = F.normalize(projected, dim=1)
                self.embeddings.data.copy_(projected)
                self.ema_embeddings.data.copy_(projected)
             
                noise = torch.randn_like(self.embeddings.data) * 0.02
                self.embeddings.data = F.normalize(self.embeddings.data + noise, dim=1)
                
            self._initialized = True
    
    def forward(self, inputs: torch.Tensor, return_tokens: bool = False) -> Tuple[torch.Tensor, Dict]:
     
        self._lazy_init()
        
     
        input_shape = inputs.shape
        
    
        if len(input_shape) == 4:
            B, C, H, W = input_shape
            flat_input = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            B = input_shape[0]
            flat_input = inputs
        
   
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.t())
        )
        
   
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
     
        quantized = torch.matmul(encodings, self.embeddings)
   
        if self.training:
            with torch.no_grad():
      
                self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings.sum(0)
                
                n = torch.sum(self.ema_cluster_size)
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon) 
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                
                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_embeddings = self.decay * self.ema_embeddings + (1 - self.decay) * dw
                
                self.embeddings.data = self.ema_embeddings / (self.ema_cluster_size.unsqueeze(1) + self.epsilon)
                
               
                self._update_usage(encoding_indices)
        
      
        commitment_loss = F.mse_loss(quantized.detach(), flat_input) * self.commitment_cost
        embedding_loss = F.mse_loss(quantized, flat_input.detach())
        
     
        quantized = flat_input + (quantized - flat_input).detach()
        

        if len(input_shape) == 4:
            quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2)
            encoding_indices = encoding_indices.view(B, H, W)
        else:
            quantized = quantized.view(input_shape)
            encoding_indices = encoding_indices.view(B)
        
     
        with torch.no_grad():
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        

        tokens = None
        if return_tokens:
            tokens = self._indices_to_tokens(encoding_indices.detach())
        
        info = {
            'indices': encoding_indices.detach(),  
            'loss': commitment_loss + embedding_loss,
            'perplexity': perplexity.detach(),
            'usage_ratio': self._get_usage_ratio(),
            'encodings': encodings.detach(), 
            'distances': distances.detach(),  
            'tokens': tokens
        }
        
        return quantized, info
    
    def _update_usage(self, indices: torch.Tensor):
       
        self.total_count += indices.numel()
        indices_list = indices.view(-1).tolist()
        for idx in set(indices_list):
            self.usage_count[idx] += indices_list.count(idx)
    
    def _get_usage_ratio(self):
       
        if self.total_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()
    
    def _indices_to_tokens(self, indices: torch.Tensor) -> List:
       
        indices_np = indices.cpu().numpy()
        if len(indices.shape) == 3:  # [B, H, W]
            B, H, W = indices.shape
            tokens = []
            for b in range(B):
                batch_tokens = []
                for h in range(H):
                    row_tokens = []
                    for w in range(W):
                        idx = indices_np[b, h, w]
                        row_tokens.append(self.vocab[idx])
                    batch_tokens.append(row_tokens)
                tokens.append(batch_tokens)
        else:  
            tokens = [self.vocab[idx] for idx in indices_np]
        return tokens
class ImprovedSemanticEncoder(nn.Module):

    
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 feature_dims: Dict[str, int] = None):
        super().__init__()
        
  
        if feature_dims is None:
            feature_dims = {
                'global': 256,
                'object': 256,
                'color': 256,
                'spatial': 256,
                'texture': 256
            }
        self.feature_dims = feature_dims
        
  
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        

        self.down1 = self._make_layer(base_channels, base_channels * 2, 2)      # 128->64
        self.down2 = self._make_layer(base_channels * 2, base_channels * 4, 2)  # 64->32
        self.down3 = self._make_layer(base_channels * 4, base_channels * 8, 2)  # 32->16


        self.global_head = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1),  # 16->8
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1),  # 8->4
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, feature_dims['global'], 1),
            nn.BatchNorm2d(feature_dims['global']),
            nn.Tanh()
        )
        
        self.object_head = nn.Sequential(
            nn.Conv2d(base_channels * 4, feature_dims['object'], 1),
            nn.BatchNorm2d(feature_dims['object']),
            nn.Tanh()
        )
        
        self.color_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, feature_dims['color'], 1),
            nn.BatchNorm2d(feature_dims['color']),
            nn.Tanh()
        )
        
        self.spatial_head = nn.Sequential(
            nn.Conv2d(base_channels * 4, feature_dims['spatial'], 1),
            nn.BatchNorm2d(feature_dims['spatial']),
            nn.Tanh()
        )
        
        self.texture_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, feature_dims['texture'], 1),
            nn.BatchNorm2d(feature_dims['texture']),
            nn.Tanh()
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        x = self.initial(x)   
        
        f1 = self.down1(x)    
        f2 = self.down2(f1)    
        f3 = self.down3(f2)    
        
        features = {
            'global': self.global_head(f3),    # [B, feature_dims['global'], 4, 4]
            'object': self.object_head(f2),    # [B, feature_dims['object'], 32, 32]
            'color': self.color_head(f1),      # [B, feature_dims['color'], 64, 64]
            'spatial': self.spatial_head(f2),  # [B, feature_dims['spatial'], 32, 32]
            'texture': self.texture_head(f1)   # [B, feature_dims['texture'], 64, 64]
        }
        
        return features


class ImprovedSemanticDecoder(nn.Module):

    
    def __init__(self, out_channels: int = 3, base_channels: int = 64,
                 feature_dims: Dict[str, int] = None):
        super().__init__()
        

        if feature_dims is None:
            feature_dims = {
                'global': 256,
                'object': 256,
                'color': 256,
                'spatial': 256,
                'texture': 256
            }
        self.feature_dims = feature_dims
   
        self.global_proj = nn.Conv2d(feature_dims['global'], base_channels * 8, 1)
        

        total_channels = (base_channels * 8 + feature_dims['object'] + 
                         feature_dims['color'] + feature_dims['spatial'] + 
                         feature_dims['texture'])
        
        self.fusion_conv = nn.Conv2d(total_channels, base_channels * 8, 1)
        
    
        self.up1 = self._make_layer(base_channels * 8, base_channels * 4, 2)   
        self.up2 = self._make_layer(base_channels * 4, base_channels * 2, 2)   
        self.up3 = self._make_layer(base_channels * 2, base_channels, 1)    
        
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 7, padding=3),
            nn.Tanh()
        )
    
    def _make_layer(self, in_channels, out_channels, scale_factor):
        layers = []
        if scale_factor > 1:
            layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)
    
    def forward(self, features):
      
        global_feat = self.global_proj(features['global'])  
        
      
        H, W = features['object'].shape[2:4] 
        
      
        combined = torch.cat([
            F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False),
            features['object'],
            F.interpolate(features['color'], size=(H, W), mode='bilinear', align_corners=False),
            features['spatial'],
            F.interpolate(features['texture'], size=(H, W), mode='bilinear', align_corners=False)
        ], dim=1)
        
        x = self.fusion_conv(combined) 
        
       
        x = self.up1(x)  # [B, 256, 64, 64]
        x = self.up2(x)  # [B, 128, 128, 128]
        x = self.up3(x)  # [B, 64, 128, 128]
        x = self.output(x)  # [B, 3, 128, 128]
        
        return x


class PatchDiscriminator(nn.Module):
 
    
    def __init__(self, in_channels=3, n_channels=64):
        super().__init__()
        
        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, n_channels, normalize=False),
            *discriminator_block(n_channels, n_channels * 2),
            *discriminator_block(n_channels * 2, n_channels * 4),
            *discriminator_block(n_channels * 4, n_channels * 8, stride=1),
            nn.Conv2d(n_channels * 8, 1, 4, 1, 1)
        )
    
    def forward(self, img):
        return self.model(img)

class FixedSemanticVQGAN(nn.Module):
   
    
    def __init__(self, config: dict):
        super().__init__()
        
       
        feature_dims = {}
        for name, cb_config in config['model']['codebooks'].items():
            feature_dims[name] = cb_config.get('embedding_dim', 256)
        
        self.encoder = ImprovedSemanticEncoder(
            in_channels=3,
            base_channels=config['model']['encoder']['base_channels'],
            feature_dims=feature_dims
        )
        
        self.decoder = ImprovedSemanticDecoder(
            out_channels=3,
            base_channels=config['model']['decoder']['base_channels'],
            feature_dims=feature_dims
        )
        
       
        self.codebooks = nn.ModuleDict()
        for name, cb_config in config['model']['codebooks'].items():
            self.codebooks[name] = FixedSemanticCodebook(
                codebook_path=cb_config['path'],
                embedding_dim=cb_config.get('embedding_dim', 256),
                commitment_cost=cb_config['commitment_cost'],
                decay=cb_config.get('decay', 0.99),
                epsilon=cb_config.get('epsilon', 1e-5)
            )
        
     
        self.use_gan = config['training']['use_gan']
        if self.use_gan:
            self.discriminator = PatchDiscriminator(
                in_channels=3,
                n_channels=config['model']['discriminator_channels']
            )
    
    def forward(self, x, return_tokens=False):
        
        encoded_features = self.encoder(x)
        
       
        quantized_features = {}
        vq_losses = {}
        vq_info = {}
        
        for name, feature in encoded_features.items():
           
            quantized, info = self.codebooks[name](feature, return_tokens=return_tokens)
            quantized_features[name] = quantized
            vq_losses[name] = info['loss']
            vq_info[name] = info
        
       
        reconstructed = self.decoder(quantized_features)
        
        return reconstructed, {
            'vq_losses': vq_losses,
            'vq_info': vq_info,
            'encoded_features': encoded_features,
            'quantized_features': quantized_features
        }

# ================== 6. 数据集 ==================
class COCOImageDataset(Dataset):
   
    
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
      
        self.captions = {}
     
        ann_file = '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/raw/captions_train2017.json'
        if os.path.exists(ann_file):
            import json
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
     
            img_to_caps = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in img_to_caps:
                    img_to_caps[img_id] = []
                img_to_caps[img_id].append(ann['caption'])
 
            for img_info in coco_data['images']:
                if img_info['id'] in img_to_caps:
                    self.captions[img_info['file_name']] = img_to_caps[img_info['id']]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
    
        filename = os.path.basename(img_path)
        caption = self.captions.get(filename, ["No caption available"])[0]
        
        return image, caption

class CaptionGenerator:
  
    
    def __init__(self):
        self.special_tokens = ['<pad>', '<unk>', '<mask>', '<object>', '<global>', 
                              '<color>', '<spatial>', '<texture>', '<item>', '<scene>']
    
    def tokens_to_caption(self, tokens_dict: Dict[str, List]) -> str:
  
        captions = []
        

        if 'global' in tokens_dict and tokens_dict['global']:
            global_tokens = self._extract_main_global(tokens_dict['global'][0])
            if global_tokens:
                captions.append(f"Scene: {', '.join(global_tokens[:3])}")
        
）
        if 'object' in tokens_dict and tokens_dict['object']:
            objects = self._extract_main_objects(tokens_dict['object'][0])
            if objects:
                captions.append(f"Objects: {', '.join(objects[:5])}")
        
      
        if 'color' in tokens_dict and tokens_dict['color']:
            colors = self._extract_main_colors(tokens_dict['color'][0])
            if colors:
                captions.append(f"Colors: {', '.join(colors[:3])}")
        

        if 'spatial' in tokens_dict and tokens_dict['spatial']:
            spatial = self._extract_spatial_info(tokens_dict['spatial'][0])
            if spatial:
                captions.append(f"Layout: {spatial}")
        
   
        if 'texture' in tokens_dict and tokens_dict['texture']:
            textures = self._extract_textures(tokens_dict['texture'][0])
            if textures:
                captions.append(f"Textures: {', '.join(textures[:3])}")
        
        return " | ".join(captions) if captions else "No meaningful tokens extracted"
    
    def _is_special_token(self, token: str) -> bool:
      
        return any(token.startswith(st) for st in self.special_tokens) or (token.startswith('<') and token.endswith('>'))
    
    def _extract_main_global(self, global_tokens_2d: List[List[str]]) -> List[str]:

        flat_tokens = [token for row in global_tokens_2d for token in row]
        
    
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
   
        counter = Counter(valid_tokens)
        return [token for token, _ in counter.most_common(3)]
    
    def _extract_main_objects(self, object_tokens_2d: List[List[str]]) -> List[str]:
 
       
        flat_tokens = [token for row in object_tokens_2d for token in row]
        
    
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
        
        counter = Counter(valid_tokens)
        
      
        return [obj for obj, _ in counter.most_common(5)]
    
    def _extract_main_colors(self, color_tokens_2d: List[List[str]]) -> List[str]:
   
        flat_tokens = [token for row in color_tokens_2d for token in row]
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
  
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 
                      'orange', 'pink', 'purple', 'brown']
        colors = [t for t in valid_tokens if any(c in t.lower() for c in color_words)]
        
        counter = Counter(colors)
        return [color for color, _ in counter.most_common(3)]
    
    def _extract_spatial_info(self, spatial_tokens_2d: List[List[str]]) -> str:
      
        flat_tokens = [token for row in spatial_tokens_2d for token in row]
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
   
        spatial_words = ['center', 'left', 'right', 'top', 'bottom', 'middle', 
                        'foreground', 'background', 'aligned', 'scattered']
        spatial_tokens = [t for t in valid_tokens if any(s in t.lower() for s in spatial_words)]
        
        if spatial_tokens:
            counter = Counter(spatial_tokens)
            main_spatial = counter.most_common(1)[0][0]
            return main_spatial
        return ""
    
    def _extract_textures(self, texture_tokens_2d: List[List[str]]) -> List[str]:
 
        flat_tokens = [token for row in texture_tokens_2d for token in row]
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
    
        texture_words = ['smooth', 'rough', 'shiny', 'matte', 'metal', 'wood', 
                        'glass', 'fabric', 'stone', 'plastic']
        textures = [t for t in valid_tokens if any(tx in t.lower() for tx in texture_words)]
        
        counter = Counter(textures)
        return [texture for texture, _ in counter.most_common(3)]


def save_samples(model, real_images, reconstructed, save_path):
 
    import torchvision.utils as vutils
    
 
    real_images = real_images * 0.5 + 0.5
    reconstructed = reconstructed * 0.5 + 0.5
    

    comparison = torch.cat([real_images, reconstructed], dim=0)
    
  
    vutils.save_image(comparison, save_path, nrow=real_images.size(0), normalize=True)

def calculate_psnr(img1, img2):
  
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(img1.device)
        return ssim(img1, img2).item()
    except:
       
        return 0.0


def train_fixed_semantic_vqgan(config: dict):
  
    
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
   
    os.makedirs(config['output_dir'], exist_ok=True)
    
  
    caption_generator = CaptionGenerator()
    

    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
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
    
    if model.use_gan:
        d_optimizer = torch.optim.AdamW(
            model.discriminator.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.5, 0.999)
        )
    

    g_scheduler = CosineAnnealingWarmRestarts(g_optimizer, T_0=len(dataloader), T_mult=2)
    

    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss_fn.eval()
    for param in perceptual_loss_fn.parameters():
        param.requires_grad = False
    

    global_step = 0
    caption_print_interval = config['training'].get('caption_print_interval', 200)
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}") as pbar:
            for batch_idx, (images, captions) in enumerate(pbar):
                images = images.to(device)
                
                
                return_tokens = (global_step % caption_print_interval == 0)
                
            
                reconstructed, info = model(images, return_tokens=return_tokens)
                
     
                recon_loss = F.mse_loss(reconstructed, images) * config['training']['loss_weights']['reconstruction']
                
          
                with torch.no_grad():
                    perceptual_loss_value = perceptual_loss_fn(reconstructed, images).mean().item()
                perceptual_loss = perceptual_loss_value * config['training']['loss_weights']['perceptual']
                
         
                vq_loss = sum(info['vq_losses'].values()) * config['training']['loss_weights']['vq']
                
            
                g_loss = recon_loss + vq_loss + perceptual_loss
    
                g_optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更彻底清理梯度
                g_loss.backward()
                
      
                torch.nn.utils.clip_grad_norm_(g_params, config['training']['gradient_clip'])
                
                g_optimizer.step()
                g_scheduler.step()
                
        
                gan_loss_value = 0.0
                if model.use_gan and global_step > 5000:
          
                    if global_step % config['training']['d_steps_per_g'] == 0:
                        d_optimizer.zero_grad(set_to_none=True)
                        
                       
                        with torch.no_grad():
                            fake_detached = reconstructed.detach()
                        
                        real_pred = model.discriminator(images)
                        fake_pred = model.discriminator(fake_detached)
                        
                        d_loss = F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()
                        
                        d_loss.backward()
                        d_optimizer.step()
                    
           
                    g_optimizer.zero_grad(set_to_none=True)
                    
                    fake_pred = model.discriminator(reconstructed)
                    gan_loss = -fake_pred.mean() * config['training']['loss_weights']['gan']
                    gan_loss_value = gan_loss.item()
                    
                    gan_loss.backward()
                    g_optimizer.step()
                
         
                with torch.no_grad():
                    pbar.set_postfix({
                        'loss': f"{g_loss.item():.2f}",
                        'recon': f"{F.mse_loss(reconstructed, images).item():.3f}",
                        'gan': f"{gan_loss_value:.3f}"
                    })
                
        
                if return_tokens and info['vq_info']['global'].get('tokens'):
                    with torch.no_grad():  # 确保不在计算图中
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
              
                        print(f"\n Token Details:")
                        for name, tokens in tokens_dict.items():
                            if name == 'global':
                              
                                token_2d = tokens[0]
                                global_tokens = []
                                for row in token_2d:
                                    for token in row:
                                        if not caption_generator._is_special_token(token):
                                            global_tokens.append(token)
                                if global_tokens:
                                    print(f"   {name} (4x4): {', '.join(global_tokens[:8])}...")
                            else:
                           
                                token_2d = tokens[0]
                                h, w = len(token_2d), len(token_2d[0])
                                center_tokens = []
                                for i in range(max(0, h//2-1), min(h, h//2+2)):
                                    for j in range(max(0, w//2-1), min(w, w//2+2)):
                                        if not caption_generator._is_special_token(token_2d[i][j]):
                                            center_tokens.append(token_2d[i][j])
                                if center_tokens:
                                    print(f"   {name} (center): {', '.join(center_tokens[:5])}...")
                        
                        print(f"{'='*80}\n")

                if global_step % config['training']['sample_interval'] == 0:
                    with torch.no_grad():  
                    
                        print(f"\n Step {global_step} - Codebook Usage:")
                        for name, codebook in model.codebooks.items():
                            usage = codebook._get_usage_ratio()
                            perplexity = info['vq_info'][name]['perplexity'].item()
                            print(f"  {name}: {usage:.2%} (perplexity: {perplexity:.1f})")
                        
                 
                        save_samples(model, images[:8], reconstructed[:8], 
                                   os.path.join(config['output_dir'], f'samples_step_{global_step}.png'))
                        
                        psnr = calculate_psnr(images, reconstructed)
                        ssim = calculate_ssim(images, reconstructed)
                        print(f"   PSNR: {psnr:.2f}, SSIM: {ssim:.3f}\n")
                
                global_step += 1
                
              
                if global_step % 50 == 0:
                 
                    del reconstructed, info, recon_loss, vq_loss, g_loss
                    if model.use_gan and global_step > 5000:
                        del fake_pred
                    torch.cuda.empty_cache()
                    gc.collect()
        
  
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            
            torch.cuda.empty_cache()
            gc.collect()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {
                    'g_optimizer': g_optimizer.state_dict(),
                },
                'config': config
            }
            
  
            if model.use_gan:
                checkpoint['optimizer_state_dict']['d_optimizer'] = d_optimizer.state_dict()
            
           
            torch.save(checkpoint, os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt'))
            
          
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
            'max_samples': None
        },
        'training': {
            'batch_size': 16,  
            'num_epochs': 100,
            'learning_rate': 0.0001,
            'use_gan': True,
            'gan_loss_type': 'standard',
            'd_steps_per_g': 1,
            'gradient_clip': 1.0,
            'loss_weights': {
                'reconstruction': 1.0,
                'perceptual': 0.1,
                'vq': 1.0,
                'gan': 0.0
            },
            'sample_interval': 500,
            'checkpoint_interval': 5,
            'caption_print_interval': 200  
        },
        'output_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs'
    }
    

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    

    train_fixed_semantic_vqgan(config)
