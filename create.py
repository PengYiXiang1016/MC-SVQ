
    
    
# build_image_based_codebooks_stable.py - 稳定版本

import torch
import torch.nn as nn
import numpy as np
import json
import clip
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
import torch.nn.functional as F

class ImageFeatureExtractor:
    """从真实图像中提取特征用于码本初始化"""
    
    def __init__(self, 
                 clip_model_path: str = None,
                 feature_type: str = 'clip'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_type = feature_type
        
        if feature_type == 'clip':
            if clip_model_path and os.path.exists(clip_model_path):
                self.model, self.preprocess = clip.load(clip_model_path, device=self.device)
            else:
                self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
            
            # 检查模型精度
            self.model_dtype = next(self.model.parameters()).dtype
            print(f"CLIP model loaded with dtype: {self.model_dtype}")
            
            # 设置为评估模式
            self.model.eval()
            
            # 特征维度
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device).to(self.model_dtype)
                dummy_features = self.model.encode_image(dummy_input)
                self.feature_dim = dummy_features.shape[1]
                print(f"Feature dimension: {self.feature_dim}")

class ImageDataset(Dataset):
    """COCO数据集"""
    def __init__(self, image_dir: str, transform=None, max_images: int = 10000):
        self.image_dir = image_dir
        self.transform = transform
        
        # 收集图像路径
        self.image_paths = []
        
        # COCO数据集结构
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(image_dir, filename))
        
        # 限制数量
        if len(self.image_paths) > max_images:
            self.image_paths = random.sample(self.image_paths, max_images)
        
        print(f"Found {len(self.image_paths)} images")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个随机彩色图像作为fallback
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
        
        return image

class ImageBasedCodebookBuilder:
    """基于真实图像特征的码本构建器"""
    
    def __init__(self,
                 image_dir: str,
                 clip_model_path: str = None,
                 embedding_dims: Dict[str, int] = None):
        
        self.image_dir = image_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征提取器
        self.feature_extractor = ImageFeatureExtractor(clip_model_path, 'clip')
        
        # Embedding维度
        self.embedding_dims = embedding_dims or {
            'global': 512,
            'object': 768,
            'color': 256,
            'spatial': 256,
            'texture': 512
        }
        
        # 加载CLIP模型
        self.clip_model = self.feature_extractor.model
        self.preprocess = self.feature_extractor.preprocess
    
    def extract_image_features_simple(self, 
                                    num_features: int = 50000,
                                    batch_size: int = 32) -> np.ndarray:
        """简单稳定的特征提取方法"""
        
        print(f"Extracting features from images...")
        
        # 创建数据集
        # 每张图像生成多个crop来增加特征多样性
        crops_per_image = 5
        needed_images = num_features // crops_per_image
        
        dataset = ImageDataset(
            self.image_dir, 
            transform=self.preprocess,
            max_images=needed_images
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        all_features = []
        
        # 定义数据增强
        augmentation_transforms = [
            self.preprocess,  # 原始
            transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                self.preprocess
            ]),
            transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 0.8)),
                self.preprocess
            ]),
            transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                self.preprocess
            ]),
            transforms.Compose([
                transforms.RandomRotation(15),
                transforms.CenterCrop(224),
                self.preprocess
            ])
        ]
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
                try:
                    # 对每个批次的图像应用不同的增强
                    batch_features = []
                    
                    # 方法1: 直接编码原始图像
                    batch = batch.to(self.device).to(self.clip_model.dtype)
                    features = self.clip_model.encode_image(batch)
                    features = features.cpu().numpy()
                    batch_features.append(features)
                    
                    # 方法2: 通过添加噪声创建变体
                    for noise_level in [0.05, 0.1]:
                        noisy_batch = batch + torch.randn_like(batch) * noise_level
                        noisy_features = self.clip_model.encode_image(noisy_batch)
                        noisy_features = noisy_features.cpu().numpy()
                        batch_features.append(noisy_features)
                    
                    # 合并批次特征
                    batch_features = np.vstack(batch_features)
                    all_features.append(batch_features)
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
                
                # 检查是否已经收集足够的特征
                current_features = sum(f.shape[0] for f in all_features)
                if current_features >= num_features:
                    break
        
        if not all_features:
            raise ValueError("No features extracted!")
        
        # 合并所有特征
        all_features = np.vstack(all_features)
        
        # 打乱并截取到目标数量
        indices = np.random.permutation(all_features.shape[0])[:num_features]
        all_features = all_features[indices]
        
        print(f"Extracted {all_features.shape[0]} features with dimension {all_features.shape[1]}")
        
        # 归一化
        all_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
        
        return all_features.astype(np.float32)
    
    def create_diverse_features(self, base_features: np.ndarray, target_count: int) -> np.ndarray:
        """从基础特征创建更多样化的特征集"""
        
        n_base = base_features.shape[0]
        feature_dim = base_features.shape[1]
        
        all_features = [base_features]
        
        # 1. 特征插值
        n_interpolate = min(target_count // 4, n_base)
        for _ in range(n_interpolate):
            idx1, idx2 = np.random.choice(n_base, 2, replace=False)
            alpha = np.random.beta(2, 2)
            interpolated = alpha * base_features[idx1] + (1 - alpha) * base_features[idx2]
            interpolated = interpolated / np.linalg.norm(interpolated)
            all_features.append(interpolated.reshape(1, -1))
        
        # 2. 添加结构化噪声
        n_noise = min(target_count // 4, n_base)
        noise_features = base_features[:n_noise].copy()
        
        # 使用不同频率的正弦噪声
        for i in range(n_noise):
            freq = np.random.uniform(0.5, 5.0)
            phase = np.random.uniform(0, 2 * np.pi)
            noise = 0.1 * np.sin(freq * np.arange(feature_dim) + phase)
            noise_features[i] += noise
            noise_features[i] = noise_features[i] / np.linalg.norm(noise_features[i])
        
        all_features.append(noise_features)
        
        # 3. 特征组合
        n_combine = min(target_count // 4, n_base // 3)
        for _ in range(n_combine):
            indices = np.random.choice(n_base, 3, replace=False)
            weights = np.random.dirichlet([1, 1, 1])
            combined = sum(w * base_features[idx] for w, idx in zip(weights, indices))
            combined = combined / np.linalg.norm(combined)
            all_features.append(combined.reshape(1, -1))
        
        # 4. PCA变换
        if n_base > 100:
            pca = PCA(n_components=min(64, feature_dim))
            pca_features = pca.fit_transform(base_features)
            
            # 在PCA空间中添加噪声
            pca_noise = pca_features + np.random.randn(*pca_features.shape) * 0.2
            
            # 反变换回原空间
            reconstructed = pca.inverse_transform(pca_noise)
            reconstructed = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)
            
            n_pca = min(target_count // 4, reconstructed.shape[0])
            all_features.append(reconstructed[:n_pca])
        
        # 合并所有特征
        all_features = np.vstack(all_features)
        
        # 随机采样到目标数量
        if all_features.shape[0] > target_count:
            indices = np.random.choice(all_features.shape[0], target_count, replace=False)
            all_features = all_features[indices]
        
        return all_features
    
    def cluster_features_multiresolution(self,
                                       features: np.ndarray,
                                       codebook_configs: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
        """多分辨率聚类"""
        
        print("Performing multi-resolution clustering...")
        
        codebooks = {}
        
        for codebook_name, (n_codes, target_dim) in codebook_configs.items():
            print(f"\nBuilding {codebook_name} codebook ({n_codes} codes, {target_dim} dims)...")
            
            # 为每个码本类型创建专门的特征
            if codebook_name == 'global':
                # 全局码本：使用原始特征
                specialized_features = features
                
            elif codebook_name == 'object':
                # 物体码本：增加局部特征的多样性
                specialized_features = self.create_diverse_features(features, n_codes * 2)
                
            elif codebook_name == 'color':
                # 颜色码本：使用颜色增强的特征
                # 简单的颜色空间变换模拟
                color_matrix = np.random.randn(features.shape[1], features.shape[1])
                color_matrix = color_matrix / np.linalg.norm(color_matrix, axis=0)
                specialized_features = features @ color_matrix
                specialized_features = specialized_features / np.linalg.norm(specialized_features, axis=1, keepdims=True)
                
            elif codebook_name == 'spatial':
                # 空间码本：添加位置编码
                n_feat = min(features.shape[0], n_codes * 2)
                spatial_features = features[:n_feat].copy()
                
                # 添加2D位置编码
                pos_enc = np.zeros((n_feat, 16))
                for i in range(8):
                    pos_enc[:, 2*i] = np.sin(np.arange(n_feat) / (10000 ** (2*i/16)))
                    pos_enc[:, 2*i+1] = np.cos(np.arange(n_feat) / (10000 ** (2*i/16)))
                
                spatial_features = np.hstack([spatial_features, pos_enc])
                specialized_features = spatial_features
                
            elif codebook_name == 'texture':
                # 纹理码本：使用高频特征
                # DCT变换模拟
                dct_matrix = np.random.randn(features.shape[1], features.shape[1])
                Q, R = np.linalg.qr(dct_matrix)
                texture_features = features @ Q
                # 强调高频成分
                texture_features[:, :features.shape[1]//2] *= 0.5
                texture_features = texture_features / np.linalg.norm(texture_features, axis=1, keepdims=True)
                specialized_features = texture_features
            
            else:
                specialized_features = features
            
            # 聚类
            print(f"  Clustering {specialized_features.shape[0]} features into {n_codes} codes...")
            
            # 使用mini-batch K-means
            kmeans = MiniBatchKMeans(
                n_clusters=n_codes,
                init='k-means++',
                max_iter=100,
                batch_size=min(256, specialized_features.shape[0]),
                reassignment_ratio=0.01,
                random_state=42,
                n_init=3
            )
            
            kmeans.fit(specialized_features)
            cluster_centers = kmeans.cluster_centers_
            
            # 调整到目标维度
            current_dim = cluster_centers.shape[1]
            
            if current_dim > target_dim:
                # PCA降维
                print(f"  Reducing dimension from {current_dim} to {target_dim}")
                pca = PCA(n_components=target_dim, random_state=42)
                cluster_centers = pca.fit_transform(cluster_centers)
                
            elif current_dim < target_dim:
                # 填充维度
                print(f"  Expanding dimension from {current_dim} to {target_dim}")
                padding = np.random.randn(n_codes, target_dim - current_dim) * 0.01
                cluster_centers = np.hstack([cluster_centers, padding])
            
            # 归一化
            cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
            
            codebooks[codebook_name] = cluster_centers.astype(np.float32)
            
            # 打印聚类统计
            print(f"  Codebook shape: {cluster_centers.shape}")
        
        return codebooks
    
    def optimize_codebook_diversity(self, 
                                  codebook: np.ndarray,
                                  name: str,
                                  target_rank_ratio: float = 0.6) -> np.ndarray:
        """优化码本多样性"""
        
        n_codes, dim = codebook.shape
        
        # 计算当前统计
        U, S, Vt = np.linalg.svd(codebook, full_matrices=False)
        
        # 有效秩
        S_normalized = S / (S.sum() + 1e-10)
        entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-10))
        current_rank = np.exp(entropy)
        current_rank_ratio = current_rank / min(n_codes, dim)
        
        print(f"  Current effective rank: {current_rank:.1f} ({current_rank_ratio*100:.1f}%)")
        
        if current_rank_ratio >= target_rank_ratio:
            return codebook
        
        # 优化奇异值分布
        print(f"  Optimizing diversity (target: {target_rank_ratio*100:.1f}%)...")
        
        # 计算目标奇异值分布
        # 使用更平滑的幂律分布
        alpha = 0.5
        target_S = np.zeros_like(S)
        for i in range(len(S)):
            target_S[i] = 1.0 / (i + 1) ** alpha
        
        # 归一化以保持总能量
        target_S = target_S * (S.sum() / target_S.sum())
        
        # 平滑插值
        lambda_factor = min(0.5, (target_rank_ratio - current_rank_ratio) * 2)
        new_S = (1 - lambda_factor) * S + lambda_factor * target_S
        
        # 重构码本
        codebook_optimized = U @ np.diag(new_S) @ Vt
        
        # 添加正交化的随机向量以增加多样性
        if current_rank_ratio < 0.3:
            print("  Adding orthogonal components...")
            
            # 生成与现有码本正交的向量
            null_space_dim = dim - len(S)
            if null_space_dim > 0:
                # 扩展Vt到完整的正交基
                _, _, Vt_full = np.linalg.svd(np.random.randn(dim, dim))
                orthogonal_components = Vt_full[len(S):]
                
                # 添加一些正交成分
                n_add = min(n_codes // 10, null_space_dim)
                random_coeffs = np.random.randn(n_codes, n_add) * 0.1
                orthogonal_contribution = random_coeffs @ orthogonal_components[:n_add]
                
                codebook_optimized += orthogonal_contribution
        
        # 最终归一化
        codebook_optimized = codebook_optimized / np.linalg.norm(codebook_optimized, axis=1, keepdims=True)
        
        # 验证新的有效秩
        _, S_new, _ = np.linalg.svd(codebook_optimized, full_matrices=False)
        S_new_normalized = S_new / (S_new.sum() + 1e-10)
        entropy_new = -np.sum(S_new_normalized * np.log(S_new_normalized + 1e-10))
        new_rank = np.exp(entropy_new)
        new_rank_ratio = new_rank / min(n_codes, dim)
        
        print(f"  New effective rank: {new_rank:.1f} ({new_rank_ratio*100:.1f}%)")
        
        return codebook_optimized.astype(np.float32)
    
    def build_all_codebooks(self,
                          output_dir: str,
                          vocab_dir: Optional[str] = None,
                          num_features: int = 50000):
        """构建所有码本"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 码本配置
        codebook_configs = {
            'global': (2048, self.embedding_dims['global']),
            'object': (8192, self.embedding_dims['object']),  
            'color': (1024, self.embedding_dims['color']),
            'spatial': (512, self.embedding_dims['spatial']),
            'texture': (2048, self.embedding_dims['texture'])
        }
        
        # 目标有效秩
        target_rank_ratios = {
            'global': 0.6,
            'object': 0.5,  # 对于大码本，降低目标以确保可达到
            'color': 0.6,
            'spatial': 0.6,
            'texture': 0.5
        }
        
        # 提取特征
        print("\n" + "="*60)
        print("STEP 1: Extracting image features")
        print("="*60)
        
        base_features = self.extract_image_features_simple(num_features)
        
        # 多分辨率聚类
        print("\n" + "="*60)
        print("STEP 2: Multi-resolution clustering")
        print("="*60)
        
        codebooks = self.cluster_features_multiresolution(base_features, codebook_configs)
        
        # 优化并保存
        print("\n" + "="*60)
        print("STEP 3: Optimizing and saving codebooks")
        print("="*60)
        
        for name, codebook in codebooks.items():
            print(f"\n{'='*50}")
            print(f"Finalizing {name} codebook")
            print(f"{'='*50}")
            
            # 优化多样性
            codebook = self.optimize_codebook_diversity(
                codebook, 
                name,
                target_rank_ratios[name]
            )
            
            # 创建输出目录
            codebook_dir = os.path.join(output_dir, name)
            os.makedirs(codebook_dir, exist_ok=True)
            
            # 处理词汇表
            if vocab_dir and os.path.exists(os.path.join(vocab_dir, name, 'vocab.json')):
                with open(os.path.join(vocab_dir, name, 'vocab.json'), 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                
                # 调整大小
                if len(vocab) != codebook.shape[0]:
                    if len(vocab) > codebook.shape[0]:
                        vocab = vocab[:codebook.shape[0]]
                    else:
                        for i in range(len(vocab), codebook.shape[0]):
                            vocab.append(f"<{name}_{i}>")
            else:
                vocab = [f"<{name}_{i}>" for i in range(codebook.shape[0])]
            
            # 保存
            np.save(os.path.join(codebook_dir, 'embeddings.npy'), codebook)
            
            with open(os.path.join(codebook_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
                json.dump(vocab, f, indent=2, ensure_ascii=False)
            
            # 计算并保存统计
            stats = self.compute_codebook_stats(codebook, name)
            with open(os.path.join(codebook_dir, 'stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"\nFinal statistics for {name}:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean similarity: {stats['mean_similarity']:.3f} ± {stats['std_similarity']:.3f}")
            print(f"  Effective rank: {stats['effective_rank']:.1f} ({stats['effective_rank_ratio']*100:.1f}%)")
            print(f"  Max similarity: {stats['max_similarity']:.3f}")
        
        print("\n" + "="*60)
        print("✅ All codebooks built successfully!")
        print("="*60)
    
    def compute_codebook_stats(self, codebook: np.ndarray, name: str) -> Dict:
        """计算码本统计信息"""
        n_codes, dim = codebook.shape
        
        # 采样计算相似度（对于大码本）
        n_sample = min(500, n_codes)
        indices = np.random.choice(n_codes, n_sample, replace=False)
        sampled = codebook[indices]
        
        # 相似度矩阵
        sim_matrix = sampled @ sampled.T
        
        # 去除对角线
        mask = ~np.eye(n_sample, dtype=bool)
        similarities = sim_matrix[mask]
        
        # 有效秩
        _, S, _ = np.linalg.svd(codebook, full_matrices=False)
        S_normalized = S / (S.sum() + 1e-10)
        entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-10))
        effective_rank = np.exp(entropy)
        
        return {
            'name': name,
            'shape': [int(n_codes), int(dim)],
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities)),
            'effective_rank': float(effective_rank),
            'effective_rank_ratio': float(effective_rank / min(n_codes, dim)),
            'percentile_90': float(np.percentile(similarities, 90)),
            'percentile_95': float(np.percentile(similarities, 95))
        }


if __name__ == "__main__":
    # 配置
    image_dir = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data"
    output_dir = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook5"
    vocab_dir = "/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/codebook4"
    
    # 自定义embedding维度
    embedding_dims = {
        'global': 512,
        'object': 768,
        'color': 256,
        'spatial': 256,
        'texture': 512
    }
    
    # 构建
    builder = ImageBasedCodebookBuilder(
        image_dir=image_dir,
        clip_model_path="/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/ViT-L-14.pt",
        embedding_dims=embedding_dims
    )
    
    builder.build_all_codebooks(
        output_dir=output_dir,
        vocab_dir=vocab_dir,
        num_features=30000  # 使用3万个特征
    )