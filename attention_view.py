
# test_codebook_attention_heatmap.py - 修复版本2
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

# 导入模型相关的类
from train import (
    FixedSemanticCodebook,
    ImprovedSemanticEncoder,
    ImprovedSemanticDecoder,
    PatchDiscriminator,
    FixedSemanticVQGAN,
    CaptionGenerator
)

class CodebookAttentionVisualizer:
    """码本注意力可视化器"""
    
    def __init__(self, model: FixedSemanticVQGAN, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 为每个码本定义颜色方案
        self.codebook_colors = {
            'global': 'Blues',
            'object': 'Reds', 
            'color': 'Greens',
            'spatial': 'Purples',
            'texture': 'Oranges'
        }
        
    def extract_codebook_features(self, image: torch.Tensor) -> Dict[str, Dict]:
        """提取每个码本的特征和注意力信息"""
        with torch.no_grad():
            # 通过完整的forward pass获取信息
            reconstructed, info = self.model(image, return_tokens=True)
            
            # 通过编码器获取特征
            encoded = self.model.encoder(image)
            
            # 存储每个码本的信息
            codebook_info = {}
            
            # 检查encoded的类型
            if isinstance(encoded, dict):
                # 如果编码器返回字典，使用对应的特征
                for name, codebook in self.model.codebooks.items():
                    if name in encoded:
                        features = encoded[name]
                    else:
                        # 如果没有对应的特征，使用第一个可用的特征
                        features = list(encoded.values())[0] if encoded else None
                        
                    if features is None:
                        print(f"Warning: No features found for {name} codebook")
                        continue
                        
                    # 处理特征
                    codebook_info[name] = self._process_codebook_features(
                        features, codebook, name, info
                    )
            else:
                # 如果编码器返回单个张量，为所有码本使用相同的特征
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
        """处理单个码本的特征"""
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # 获取码本的embeddings - 修复：直接使用codebook.embeddings
        embeddings = codebook.embeddings
        
        # 如果特征维度和码本维度不匹配，需要投影
        if C != embeddings.shape[1]:
            # 创建投影层
            if not hasattr(self, f'projection_{codebook_name}'):
                projection = nn.Linear(C, embeddings.shape[1]).to(self.device)
                # 初始化投影层
                nn.init.xavier_uniform_(projection.weight)
                setattr(self, f'projection_{codebook_name}', projection)
            else:
                projection = getattr(self, f'projection_{codebook_name}')
            
            with torch.no_grad():
                features_flat = projection(features_flat)
        
        # 计算距离
        distances = torch.cdist(features_flat, embeddings)
        
        # 获取最近的码本索引
        encoding_indices = torch.argmin(distances, dim=1)
        
        # 计算软注意力权重（使用负距离的softmax）
        attention_weights = F.softmax(-distances / 0.1, dim=1)  # 添加温度参数
        
        # 获取最大注意力权重作为该位置的激活强度
        max_attention = torch.max(attention_weights, dim=1)[0]
        max_attention = max_attention.view(B, H, W)
        
        # 获取使用的码本索引
        encoding_indices = encoding_indices.view(B, H, W)
        
        # 如果forward_info中有对应的信息，使用它
        if 'vq_info' in forward_info and codebook_name in forward_info['vq_info']:
            vq_info = forward_info['vq_info'][codebook_name]
            if 'encodings' in vq_info:
                # 使用实际的编码信息
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
        """创建注意力热力图可视化"""
        
        # 将图像转换为numpy并反归一化
        image_np = image.squeeze(0).cpu().numpy()
        image_np = (image_np * 0.5 + 0.5).transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        
        # 创建图形布局
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.2)
        
        # 1. 显示原始图像
        ax_orig = fig.add_subplot(gs[0, 0])
        ax_orig.imshow(image_np)
        ax_orig.set_title('Original Image', fontsize=14, fontweight='bold')
        ax_orig.axis('off')
        
        # 2. 为每个码本创建热力图
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for idx, (name, info) in enumerate(codebook_info.items()):
            if idx < len(positions):
                row, col = positions[idx]
                ax = fig.add_subplot(gs[row, col])
                
                # 获取注意力图
                attention_map = info['attention_map'].squeeze(0).cpu().numpy()
                
                # 获取特征图的尺寸
                _, _, feat_h, feat_w = info['features_shape']
                
                # 如果注意力图尺寸与图像不同，进行上采样
                if attention_map.shape != (image_np.shape[0], image_np.shape[1]):
                    attention_map_resized = F.interpolate(
                        torch.tensor(attention_map).unsqueeze(0).unsqueeze(0),
                        size=(image_np.shape[0], image_np.shape[1]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze().numpy()
                else:
                    attention_map_resized = attention_map
                
                # 创建热力图
                im = ax.imshow(attention_map_resized, 
                             cmap=self.codebook_colors[name],
                             alpha=0.8)
                
                # 叠加原始图像
                ax.imshow(image_np, alpha=0.3)
                
                ax.set_title(f'{name.capitalize()} Codebook Attention\n(Feature size: {feat_h}×{feat_w})', 
                           fontsize=12, fontweight='bold')
                ax.axis('off')
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
        
        # 3. 创建组合热力图
        ax_combined = fig.add_subplot(gs[2, :])
        
        # 为每个码本分配不同的颜色通道
        combined_map = np.zeros((image_np.shape[0], image_np.shape[1], 3))
        
        color_channels = {
            'global': [0, 0, 1],    # 蓝色
            'object': [1, 0, 0],    # 红色
            'color': [0, 1, 0],     # 绿色
            'spatial': [1, 0, 1],   # 紫色
            'texture': [1, 0.5, 0]  # 橙色
        }
        
        for name, info in codebook_info.items():
            attention_map = info['attention_map'].squeeze(0).cpu().numpy()
            
            # 上采样到图像大小
            if attention_map.shape != (image_np.shape[0], image_np.shape[1]):
                attention_map_resized = F.interpolate(
                    torch.tensor(attention_map).unsqueeze(0).unsqueeze(0),
                    size=(image_np.shape[0], image_np.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().numpy()
            else:
                attention_map_resized = attention_map
            
            # 归一化
            if attention_map_resized.max() > 0:
                attention_map_resized = attention_map_resized / attention_map_resized.max()
            
            # 添加到组合图
            color = color_channels.get(name, [1, 1, 1])
            for i in range(3):
                combined_map[:, :, i] += attention_map_resized * color[i] * 0.3
        
        # 归一化组合图
        combined_map = np.clip(combined_map, 0, 1)
        
        # 显示组合图
        ax_combined.imshow(image_np)
        ax_combined.imshow(combined_map, alpha=0.6)
        ax_combined.set_title('Combined Codebook Attention Map', 
                            fontsize=14, fontweight='bold')
        ax_combined.axis('off')
        
        # 添加图例
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
            print(f"✅ Saved attention heatmap to: {save_path}")
        
        return fig
    
    def analyze_codebook_usage(self, 
                             codebook_info: Dict[str, Dict]) -> Dict[str, Dict]:
        """分析码本使用情况"""
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
    """从checkpoint加载模型"""
    model = FixedSemanticVQGAN(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    return model


def test_single_image(image_path: str, 
                     model: FixedSemanticVQGAN,
                     output_dir: str,
                     device: torch.device):
    """测试单张图片"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 使用训练时的图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    print(f"\n🖼️  Processing image: {os.path.basename(image_path)}")
    
    # 创建可视化器
    visualizer = CodebookAttentionVisualizer(model, device)
    
    # 提取码本特征
    print("📊 Extracting codebook features...")
    codebook_info = visualizer.extract_codebook_features(image_tensor)
    
    # 创建热力图
    print("🎨 Creating attention heatmap...")
    output_path = os.path.join(output_dir, 
                              f'attention_heatmap_{os.path.basename(image_path)}')
    fig = visualizer.create_attention_heatmap(image_tensor, 
                                            codebook_info, 
                                            save_path=output_path)
    plt.show()
    
    # 分析码本使用情况
    print("\n📈 Codebook usage statistics:")
    usage_stats = visualizer.analyze_codebook_usage(codebook_info)
    for name, stats in usage_stats.items():
        print(f"\n{name.capitalize()} codebook:")
        print(f"  - Unique codes used: {stats['num_unique_codes']}")
        print(f"  - Most used codes: {stats['most_used_codes']}")
    
    # 生成重建图像进行对比
    print("\n🔄 Generating reconstruction...")
    with torch.no_grad():
        reconstructed, info = model(image_tensor, return_tokens=True)
    
    # 保存重建对比图
    fig_recon = plt.figure(figsize=(10, 5))
    
    # 原始图像
    ax1 = fig_recon.add_subplot(1, 2, 1)
    image_np = image_tensor.squeeze(0).cpu().numpy()
    image_np = (image_np * 0.5 + 0.5).transpose(1, 2, 0)
    ax1.imshow(np.clip(image_np, 0, 1))
    ax1.set_title('Original', fontsize=12)
    ax1.axis('off')
    
    # 重建图像
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
    
    print(f"✅ Reconstruction saved to: {recon_path}")
    
    # 如果有token信息，生成caption
    if 'vq_info' in info:
        caption_generator = CaptionGenerator()
        tokens_dict = {}
        
        for codebook_name in ['global', 'object', 'color', 'spatial', 'texture']:
            if codebook_name in info['vq_info'] and info['vq_info'][codebook_name].get('tokens'):
                tokens_dict[codebook_name] = info['vq_info'][codebook_name]['tokens']
        
        if tokens_dict:
            generated_caption = caption_generator.tokens_to_caption(tokens_dict)
            print(f"\n🤖 Generated Caption from Tokens:")
            print(f"   {generated_caption}")


if __name__ == "__main__":
    # 配置（与训练时相同）
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
            'use_gan': False  # 与训练时保持一致
        }
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    checkpoint_path = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoint_epoch_60.pt"
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    
    # 测试图像路径
    test_images = [
        '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data/000000581881.jpg',
        # 可以添加更多测试图像
    ]
    
    # 输出目录
    output_dir = '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/attention'
    
    # 测试每张图像
    for image_path in test_images:
        if os.path.exists(image_path):
            test_single_image(image_path, model, output_dir, device)
        else:
            print(f"❌ Image not found: {image_path}")