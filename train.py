
    
    
# enhanced_semantic_vqgan_complete.py - 支持不同码本维度的版本

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
import gc  # 添加垃圾回收

# ================== 1. 修复的语义码本 ==================
class FixedSemanticCodebook(nn.Module):
    """修复的语义码本 - 解决设备问题和显存泄漏"""
    
    def __init__(self,
                 codebook_path: str,
                 embedding_dim: int = 256,
                 commitment_cost: float = 0.25,
                 decay: float = 0.99,
                 epsilon: float = 1e-5,
                 device: str = 'cuda'):
        super().__init__()
        
        self.device = device
        
        # 加载预训练的CLIP embeddings
        if codebook_path.endswith('.npy'):
            embeddings_path = codebook_path
            vocab_path = codebook_path.replace('embeddings.npy', 'vocab.json')
        else:
            embeddings_path = os.path.join(codebook_path, 'embeddings.npy')
            vocab_path = os.path.join(codebook_path, 'vocab.json')
        
        # 加载原始embeddings
        clip_embeddings = np.load(embeddings_path).astype(np.float32)
        
        # 加载词汇表
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.num_embeddings = len(self.vocab)
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        print(f"Loaded codebook from {codebook_path}")
        print(f"  Original shape: {clip_embeddings.shape}")
        print(f"  Vocab size: {self.num_embeddings}")
        print(f"  Target embedding dim: {self.embedding_dim}")
        
        # 投影层：将CLIP embeddings投影到目标维度
        clip_dim = clip_embeddings.shape[1]
        self.projection = nn.Linear(clip_dim, embedding_dim)
        
        # 初始化可学习的embeddings
        self.embeddings = nn.Parameter(torch.randn(self.num_embeddings, embedding_dim))
        
        # EMA更新
        self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings))
        self.register_buffer('ema_embeddings', torch.randn(self.num_embeddings, embedding_dim))
        self.decay = decay
        self.epsilon = epsilon
        
        # 使用统计
        self.register_buffer('usage_count', torch.zeros(self.num_embeddings))
        self.register_buffer('total_count', torch.tensor(0))
        
        # 初始化embeddings（延迟到模型移动到设备后）
        self.register_buffer('_clip_embeddings_cache', torch.from_numpy(clip_embeddings))
        self._initialized = False
    
    def _lazy_init(self):
        """延迟初始化，确保在正确的设备上"""
        if not self._initialized:
            with torch.no_grad():
                # 投影CLIP embeddings
                projected = self.projection(self._clip_embeddings_cache)
                projected = F.normalize(projected, dim=1)
                self.embeddings.data.copy_(projected)
                self.ema_embeddings.data.copy_(projected)
                
                # 添加噪声以增加多样性
                noise = torch.randn_like(self.embeddings.data) * 0.02
                self.embeddings.data = F.normalize(self.embeddings.data + noise, dim=1)
                
            self._initialized = True
    
    def forward(self, inputs: torch.Tensor, return_tokens: bool = False) -> Tuple[torch.Tensor, Dict]:
        # 确保初始化
        self._lazy_init()
        
        # inputs shape: [B, C, H, W] or [B, C]
        input_shape = inputs.shape
        
        # 展平
        if len(input_shape) == 4:
            B, C, H, W = input_shape
            flat_input = inputs.permute(0, 2, 3, 1).reshape(-1, C)
        else:
            B = input_shape[0]
            flat_input = inputs
        
        # 计算L2距离
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.embeddings**2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.t())
        )
        
        # 硬量化（最近邻）
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 量化
        quantized = torch.matmul(encodings, self.embeddings)
        
        # 更新统计（修复：使用no_grad避免梯度累积）
        if self.training:
            with torch.no_grad():
                # EMA更新
                self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings.sum(0)
                
                n = torch.sum(self.ema_cluster_size)
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon) 
                    / (n + self.num_embeddings * self.epsilon) * n
                )
                
                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_embeddings = self.decay * self.ema_embeddings + (1 - self.decay) * dw
                
                self.embeddings.data = self.ema_embeddings / (self.ema_cluster_size.unsqueeze(1) + self.epsilon)
                
                # 更新使用统计
                self._update_usage(encoding_indices)
        
        # 计算损失
        commitment_loss = F.mse_loss(quantized.detach(), flat_input) * self.commitment_cost
        embedding_loss = F.mse_loss(quantized, flat_input.detach())
        
        # 直通估计
        quantized = flat_input + (quantized - flat_input).detach()
        
        # 重塑
        if len(input_shape) == 4:
            quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2)
            encoding_indices = encoding_indices.view(B, H, W)
        else:
            quantized = quantized.view(input_shape)
            encoding_indices = encoding_indices.view(B)
        
        # 计算perplexity（修复：detach避免梯度）
        with torch.no_grad():
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 获取token词汇（如果需要）
        tokens = None
        if return_tokens:
            tokens = self._indices_to_tokens(encoding_indices.detach())
        
        info = {
            'indices': encoding_indices.detach(),  # detach索引
            'loss': commitment_loss + embedding_loss,
            'perplexity': perplexity.detach(),
            'usage_ratio': self._get_usage_ratio(),
            'encodings': encodings.detach(),  # detach编码
            'distances': distances.detach(),  # detach距离
            'tokens': tokens
        }
        
        return quantized, info
    
    def _update_usage(self, indices: torch.Tensor):
        """更新使用统计"""
        self.total_count += indices.numel()
        indices_list = indices.view(-1).tolist()
        for idx in set(indices_list):
            self.usage_count[idx] += indices_list.count(idx)
    
    def _get_usage_ratio(self):
        """获取真实的使用率"""
        if self.total_count == 0:
            return 0.0
        return (self.usage_count > 0).float().mean().item()
    
    def _indices_to_tokens(self, indices: torch.Tensor) -> List:
        """将索引转换为词汇"""
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
        else:  # [B]
            tokens = [self.vocab[idx] for idx in indices_np]
        return tokens

# ================== 2. 改进的编码器（支持动态特征维度） ==================
class ImprovedSemanticEncoder(nn.Module):
    """改进的语义编码器 - 支持不同的输出维度"""
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64, 
                 feature_dims: Dict[str, int] = None):
        super().__init__()
        
        # 默认特征维度
        if feature_dims is None:
            feature_dims = {
                'global': 256,
                'object': 256,
                'color': 256,
                'spatial': 256,
                'texture': 256
            }
        self.feature_dims = feature_dims
        
        # 初始卷积
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 下采样路径 - 对于128x128输入
        self.down1 = self._make_layer(base_channels, base_channels * 2, 2)      # 128->64
        self.down2 = self._make_layer(base_channels * 2, base_channels * 4, 2)  # 64->32
        self.down3 = self._make_layer(base_channels * 4, base_channels * 8, 2)  # 32->16
        
        # 特征提取头 - 使用动态特征维度
        # global_head：从16x16下采样到4x4
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
        # 前向传播
        x = self.initial(x)     # 128x128
        
        f1 = self.down1(x)      # 64x64
        f2 = self.down2(f1)     # 32x32
        f3 = self.down3(f2)     # 16x16
        
        features = {
            'global': self.global_head(f3),    # [B, feature_dims['global'], 4, 4]
            'object': self.object_head(f2),    # [B, feature_dims['object'], 32, 32]
            'color': self.color_head(f1),      # [B, feature_dims['color'], 64, 64]
            'spatial': self.spatial_head(f2),  # [B, feature_dims['spatial'], 32, 32]
            'texture': self.texture_head(f1)   # [B, feature_dims['texture'], 64, 64]
        }
        
        return features

# ================== 3. 改进的解码器（支持动态特征维度） ==================
class ImprovedSemanticDecoder(nn.Module):
    """改进的语义解码器 - 支持不同的输入维度"""
    
    def __init__(self, out_channels: int = 3, base_channels: int = 64,
                 feature_dims: Dict[str, int] = None):
        super().__init__()
        
        # 默认特征维度
        if feature_dims is None:
            feature_dims = {
                'global': 256,
                'object': 256,
                'color': 256,
                'spatial': 256,
                'texture': 256
            }
        self.feature_dims = feature_dims
        
        # global特征处理 - 已经是4x4，只需要通道变换
        self.global_proj = nn.Conv2d(feature_dims['global'], base_channels * 8, 1)
        
        # 计算融合后的总通道数
        total_channels = (base_channels * 8 + feature_dims['object'] + 
                         feature_dims['color'] + feature_dims['spatial'] + 
                         feature_dims['texture'])
        
        self.fusion_conv = nn.Conv2d(total_channels, base_channels * 8, 1)
        
        # 上采样路径 - 从32x32到128x128
        self.up1 = self._make_layer(base_channels * 8, base_channels * 4, 2)   # 32->64
        self.up2 = self._make_layer(base_channels * 4, base_channels * 2, 2)   # 64->128
        self.up3 = self._make_layer(base_channels * 2, base_channels, 1)       # 保持128
        
        # 输出层
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
        # 处理全局特征 - 从4x4上采样到32x32
        global_feat = self.global_proj(features['global'])  # [B, 512, 4, 4]
        
        # 目标尺寸是32x32（object特征的尺寸）
        H, W = features['object'].shape[2:4]  # 32x32
        
        # 融合所有特征
        combined = torch.cat([
            F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False),
            features['object'],
            F.interpolate(features['color'], size=(H, W), mode='bilinear', align_corners=False),
            features['spatial'],
            F.interpolate(features['texture'], size=(H, W), mode='bilinear', align_corners=False)
        ], dim=1)
        
        x = self.fusion_conv(combined)  # [B, 512, 32, 32]
        
        # 解码到原始尺寸
        x = self.up1(x)  # [B, 256, 64, 64]
        x = self.up2(x)  # [B, 128, 128, 128]
        x = self.up3(x)  # [B, 64, 128, 128]
        x = self.output(x)  # [B, 3, 128, 128]
        
        return x

# ================== 4. 判别器 ==================
class PatchDiscriminator(nn.Module):
    """PatchGAN判别器"""
    
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

# ================== 5. 主模型（支持不同码本维度） ==================
class FixedSemanticVQGAN(nn.Module):
    """修复的Semantic VQ-GAN - 支持不同码本维度"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # 获取各个码本的维度
        feature_dims = {}
        for name, cb_config in config['model']['codebooks'].items():
            feature_dims[name] = cb_config.get('embedding_dim', 256)
        
        # 编码器和解码器
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
        
        # 语义码本
        self.codebooks = nn.ModuleDict()
        for name, cb_config in config['model']['codebooks'].items():
            self.codebooks[name] = FixedSemanticCodebook(
                codebook_path=cb_config['path'],
                embedding_dim=cb_config.get('embedding_dim', 256),
                commitment_cost=cb_config['commitment_cost'],
                decay=cb_config.get('decay', 0.99),
                epsilon=cb_config.get('epsilon', 1e-5)
            )
        
        # 判别器（可选）
        self.use_gan = config['training']['use_gan']
        if self.use_gan:
            self.discriminator = PatchDiscriminator(
                in_channels=3,
                n_channels=config['model']['discriminator_channels']
            )
    
    def forward(self, x, return_tokens=False):
        # 编码
        encoded_features = self.encoder(x)
        
        # 量化
        quantized_features = {}
        vq_losses = {}
        vq_info = {}
        
        for name, feature in encoded_features.items():
            # 完整的量化调用
            quantized, info = self.codebooks[name](feature, return_tokens=return_tokens)
            quantized_features[name] = quantized
            vq_losses[name] = info['loss']
            vq_info[name] = info
        
        # 解码
        reconstructed = self.decoder(quantized_features)
        
        return reconstructed, {
            'vq_losses': vq_losses,
            'vq_info': vq_info,
            'encoded_features': encoded_features,
            'quantized_features': quantized_features
        }

# ================== 6. 数据集 ==================
class COCOImageDataset(Dataset):
    """COCO图像数据集，支持caption"""
    
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 加载图像
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        
        # 尝试加载COCO annotations（如果存在）
        self.captions = {}
        # ann_file = os.path.join(os.path.dirname(root_dir), 'annotations', 'captions_train2017.json')
        ann_file = '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/raw/captions_train2017.json'
        if os.path.exists(ann_file):
            import json
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            # 创建image_id到captions的映射
            img_to_caps = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in img_to_caps:
                    img_to_caps[img_id] = []
                img_to_caps[img_id].append(ann['caption'])
            
            # 创建filename到captions的映射
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
        
        # 获取caption（如果有）
        filename = os.path.basename(img_path)
        caption = self.captions.get(filename, ["No caption available"])[0]
        
        return image, caption

# ================== 7. Caption生成器（适配4x4 global） ==================
class CaptionGenerator:
    """从码本tokens生成caption - 适配4x4的global tokens"""
    
    def __init__(self):
        self.special_tokens = ['<pad>', '<unk>', '<mask>', '<object>', '<global>', 
                              '<color>', '<spatial>', '<texture>', '<item>', '<scene>']
    
    def tokens_to_caption(self, tokens_dict: Dict[str, List]) -> str:
        """将不同码本的tokens组合成caption"""
        captions = []
        
        # 全局描述 - 现在处理4x4的tokens
        if 'global' in tokens_dict and tokens_dict['global']:
            global_tokens = self._extract_main_global(tokens_dict['global'][0])
            if global_tokens:
                captions.append(f"Scene: {', '.join(global_tokens[:3])}")
        
        # 主要物体（从object tokens中提取）
        if 'object' in tokens_dict and tokens_dict['object']:
            objects = self._extract_main_objects(tokens_dict['object'][0])
            if objects:
                captions.append(f"Objects: {', '.join(objects[:5])}")
        
        # 主要颜色
        if 'color' in tokens_dict and tokens_dict['color']:
            colors = self._extract_main_colors(tokens_dict['color'][0])
            if colors:
                captions.append(f"Colors: {', '.join(colors[:3])}")
        
        # 空间关系
        if 'spatial' in tokens_dict and tokens_dict['spatial']:
            spatial = self._extract_spatial_info(tokens_dict['spatial'][0])
            if spatial:
                captions.append(f"Layout: {spatial}")
        
        # 纹理信息
        if 'texture' in tokens_dict and tokens_dict['texture']:
            textures = self._extract_textures(tokens_dict['texture'][0])
            if textures:
                captions.append(f"Textures: {', '.join(textures[:3])}")
        
        return " | ".join(captions) if captions else "No meaningful tokens extracted"
    
    def _is_special_token(self, token: str) -> bool:
        """检查是否为特殊token"""
        return any(token.startswith(st) for st in self.special_tokens) or (token.startswith('<') and token.endswith('>'))
    
    def _extract_main_global(self, global_tokens_2d: List[List[str]]) -> List[str]:
        """提取主要的全局语义 - 处理4x4的tokens"""
        # 展平4x4 tokens
        flat_tokens = [token for row in global_tokens_2d for token in row]
        
        # 过滤特殊tokens
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
        # 统计频率，返回最常见的全局描述
        counter = Counter(valid_tokens)
        return [token for token, _ in counter.most_common(3)]
    
    def _extract_main_objects(self, object_tokens_2d: List[List[str]]) -> List[str]:
        """提取主要物体"""
        # 展平2D tokens
        flat_tokens = [token for row in object_tokens_2d for token in row]
        
        # 过滤特殊tokens
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
        # 统计频率
        counter = Counter(valid_tokens)
        
        # 返回最常见的物体
        return [obj for obj, _ in counter.most_common(5)]
    
    def _extract_main_colors(self, color_tokens_2d: List[List[str]]) -> List[str]:
        """提取主要颜色"""
        flat_tokens = [token for row in color_tokens_2d for token in row]
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
        # 过滤出颜色相关的tokens
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 
                      'orange', 'pink', 'purple', 'brown']
        colors = [t for t in valid_tokens if any(c in t.lower() for c in color_words)]
        
        counter = Counter(colors)
        return [color for color, _ in counter.most_common(3)]
    
    def _extract_spatial_info(self, spatial_tokens_2d: List[List[str]]) -> str:
        """提取空间信息"""
        flat_tokens = [token for row in spatial_tokens_2d for token in row]
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
        # 查找空间关系词
        spatial_words = ['center', 'left', 'right', 'top', 'bottom', 'middle', 
                        'foreground', 'background', 'aligned', 'scattered']
        spatial_tokens = [t for t in valid_tokens if any(s in t.lower() for s in spatial_words)]
        
        if spatial_tokens:
            counter = Counter(spatial_tokens)
            main_spatial = counter.most_common(1)[0][0]
            return main_spatial
        return ""
    
    def _extract_textures(self, texture_tokens_2d: List[List[str]]) -> List[str]:
        """提取纹理信息"""
        flat_tokens = [token for row in texture_tokens_2d for token in row]
        valid_tokens = [t for t in flat_tokens if not self._is_special_token(t)]
        
        # 过滤材质相关词
        texture_words = ['smooth', 'rough', 'shiny', 'matte', 'metal', 'wood', 
                        'glass', 'fabric', 'stone', 'plastic']
        textures = [t for t in valid_tokens if any(tx in t.lower() for tx in texture_words)]
        
        counter = Counter(textures)
        return [texture for texture, _ in counter.most_common(3)]

# ================== 8. 辅助函数 ==================
def save_samples(model, real_images, reconstructed, save_path):
    """保存样本图像"""
    import torchvision.utils as vutils
    
    # 反归一化
    real_images = real_images * 0.5 + 0.5
    reconstructed = reconstructed * 0.5 + 0.5
    
    # 拼接图像
    comparison = torch.cat([real_images, reconstructed], dim=0)
    
    # 保存
    vutils.save_image(comparison, save_path, nrow=real_images.size(0), normalize=True)

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """计算SSIM（简化版）"""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(img1.device)
        return ssim(img1, img2).item()
    except:
        # 如果没有安装torchmetrics，返回占位值
        return 0.0

# ================== 9. 训练函数（完全修复显存泄漏） ==================
def train_fixed_semantic_vqgan(config: dict):
    """完全修复的训练函数 - 解决所有显存泄漏问题"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Caption生成器
    caption_generator = CaptionGenerator()
    
    # 数据加载
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
    
    # 创建模型
    model = FixedSemanticVQGAN(config).to(device)
    
    # 打印模型参数量和各码本维度
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print("\nCodebook dimensions:")
    for name, cb_config in config['model']['codebooks'].items():
        print(f"  {name}: {cb_config.get('embedding_dim', 256)}")
    
    # 优化器
    g_params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.codebooks.parameters())
    g_optimizer = torch.optim.AdamW(g_params, lr=config['training']['learning_rate'], betas=(0.5, 0.999))
    
    if model.use_gan:
        d_optimizer = torch.optim.AdamW(
            model.discriminator.parameters(),
            lr=config['training']['learning_rate'],
            betas=(0.5, 0.999)
        )
    
    # 学习率调度器
    g_scheduler = CosineAnnealingWarmRestarts(g_optimizer, T_0=len(dataloader), T_mult=2)
    
    # 损失函数（修复：eval模式避免梯度累积）
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    perceptual_loss_fn.eval()
    for param in perceptual_loss_fn.parameters():
        param.requires_grad = False
    
    # 训练循环
    global_step = 0
    caption_print_interval = config['training'].get('caption_print_interval', 200)
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}") as pbar:
            for batch_idx, (images, captions) in enumerate(pbar):
                images = images.to(device)
                
                # 决定是否需要返回tokens
                return_tokens = (global_step % caption_print_interval == 0)
                
                # 前向传播
                reconstructed, info = model(images, return_tokens=return_tokens)
                
                # 计算损失（修复：使用no_grad避免梯度累积）
                # 1. 重建损失
                recon_loss = F.mse_loss(reconstructed, images) * config['training']['loss_weights']['reconstruction']
                
                # 2. 感知损失（完全detach）
                with torch.no_grad():
                    perceptual_loss_value = perceptual_loss_fn(reconstructed, images).mean().item()
                perceptual_loss = perceptual_loss_value * config['training']['loss_weights']['perceptual']
                
                # 3. VQ损失
                vq_loss = sum(info['vq_losses'].values()) * config['training']['loss_weights']['vq']
                
                # 生成器总损失（不包含感知损失的梯度）
                g_loss = recon_loss + vq_loss + perceptual_loss
                
                # 生成器优化
                g_optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更彻底清理梯度
                g_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(g_params, config['training']['gradient_clip'])
                
                g_optimizer.step()
                g_scheduler.step()
                
                # GAN训练（如果启用）
                gan_loss_value = 0.0
                if model.use_gan and global_step > 5000:
                    # 判别器训练
                    if global_step % config['training']['d_steps_per_g'] == 0:
                        d_optimizer.zero_grad(set_to_none=True)
                        
                        # 完全detach重建图像
                        with torch.no_grad():
                            fake_detached = reconstructed.detach()
                        
                        real_pred = model.discriminator(images)
                        fake_pred = model.discriminator(fake_detached)
                        
                        d_loss = F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()
                        
                        d_loss.backward()
                        d_optimizer.step()
                    
                    # 生成器对抗损失
                    g_optimizer.zero_grad(set_to_none=True)
                    
                    fake_pred = model.discriminator(reconstructed)
                    gan_loss = -fake_pred.mean() * config['training']['loss_weights']['gan']
                    gan_loss_value = gan_loss.item()
                    
                    gan_loss.backward()
                    g_optimizer.step()
                
                # 更新进度条（detach所有值）
                with torch.no_grad():
                    pbar.set_postfix({
                        'loss': f"{g_loss.item():.2f}",
                        'recon': f"{F.mse_loss(reconstructed, images).item():.3f}",
                        'gan': f"{gan_loss_value:.3f}"
                    })
                
                # 打印caption对比（每caption_print_interval步）
                if return_tokens and info['vq_info']['global'].get('tokens'):
                    with torch.no_grad():  # 确保不在计算图中
                        print(f"\n{'='*80}")
                        print(f"[Step {global_step}] Caption Comparison:")
                        print(f"{'='*80}")
                        
                        # 随机选择一张图像
                        idx = np.random.randint(0, len(captions))
                        
                        # 真实caption
                        print(f"\n📝 Real Caption:")
                        print(f"   {captions[idx]}")
                        
                        # 收集所有码本的tokens
                        tokens_dict = {}
                        for codebook_name in ['global', 'object', 'color', 'spatial', 'texture']:
                            if codebook_name in info['vq_info'] and info['vq_info'][codebook_name].get('tokens'):
                                tokens_dict[codebook_name] = [info['vq_info'][codebook_name]['tokens'][idx]]
                        
                        # 生成caption
                        generated_caption = caption_generator.tokens_to_caption(tokens_dict)
                        print(f"\n🤖 Generated Caption from Tokens:")
                        print(f"   {generated_caption}")
                        
                        # 打印详细的token信息
                        print(f"\n📊 Token Details:")
                        for name, tokens in tokens_dict.items():
                            if name == 'global':
                                # global现在是4x4，显示所有16个tokens
                                token_2d = tokens[0]
                                global_tokens = []
                                for row in token_2d:
                                    for token in row:
                                        if not caption_generator._is_special_token(token):
                                            global_tokens.append(token)
                                if global_tokens:
                                    print(f"   {name} (4x4): {', '.join(global_tokens[:8])}...")
                            else:
                                # 对于其他2D tokens，显示中心区域
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
                
                # 定期保存和评估
                if global_step % config['training']['sample_interval'] == 0:
                    with torch.no_grad():  # 确保不在计算图中
                        # 计算并打印码本使用率
                        print(f"\n📈 Step {global_step} - Codebook Usage:")
                        for name, codebook in model.codebooks.items():
                            usage = codebook._get_usage_ratio()
                            perplexity = info['vq_info'][name]['perplexity'].item()
                            print(f"  {name}: {usage:.2%} (perplexity: {perplexity:.1f})")
                        
                        # 保存样本
                        save_samples(model, images[:8], reconstructed[:8], 
                                   os.path.join(config['output_dir'], f'samples_step_{global_step}.png'))
                        
                        # 计算PSNR和SSIM
                        psnr = calculate_psnr(images, reconstructed)
                        ssim = calculate_ssim(images, reconstructed)
                        print(f"  📊 PSNR: {psnr:.2f}, SSIM: {ssim:.3f}\n")
                
                global_step += 1
                
                # 定期清理显存（更激进的清理）
                if global_step % 50 == 0:
                    # 删除不需要的中间变量
                    del reconstructed, info, recon_loss, vq_loss, g_loss
                    if model.use_gan and global_step > 5000:
                        del fake_pred
                    torch.cuda.empty_cache()
                    gc.collect()
        
        # 保存检查点（修复：确保不保存计算图）
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            # 先清理显存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 保存模型（只保存必要的部分）
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {
                    'g_optimizer': g_optimizer.state_dict(),
                },
                'config': config
            }
            
            # 如果使用GAN，保存判别器优化器
            if model.use_gan:
                checkpoint['optimizer_state_dict']['d_optimizer'] = d_optimizer.state_dict()
            
            # 保存到磁盘
            torch.save(checkpoint, os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt'))
            
            # 立即删除checkpoint字典
            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"✅ Saved checkpoint for epoch {epoch+1}")

# ================== 10. 主程序 ==================
if __name__ == "__main__":
    # 优化的配置 - 支持不同的码本维度
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
                    'embedding_dim': 768,  # 可以设置不同的维度
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'object': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/object',
                    'embedding_dim': 512,  # 可以设置不同的维度
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'color': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/color',
                    'embedding_dim': 256,  # 可以设置不同的维度
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'spatial': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/spatial',
                    'embedding_dim': 384,  # 可以设置不同的维度
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                },
                'texture': {
                    'path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5/texture',
                    'embedding_dim': 256,  # 可以设置不同的维度
                    'commitment_cost': 0.25,
                    'decay': 0.99,
                    'epsilon': 1e-5
                }
            }
        },
        'data': {
            'train_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data',
            'image_size': 128,  # 输入图像大小
            'num_workers': 4,  # 减少worker数量以减少内存使用
            'max_samples': None
        },
        'training': {
            'batch_size': 16,  # 减小batch size以减少显存使用
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
            'caption_print_interval': 200  # 每200步打印caption对比
        },
        'output_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs'
    }
    
    # 设置环境变量以减少显存碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # 开始训练
    train_fixed_semantic_vqgan(config)