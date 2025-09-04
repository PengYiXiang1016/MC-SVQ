
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import gc
from collections import defaultdict
import math
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from torch.cuda.amp import autocast, GradScaler
from peft import LoraConfig, get_peft_model, TaskType


try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    METRICS_AVAILABLE = True
except:
    print("Warning: COCO evaluation metrics not available")
    METRICS_AVAILABLE = False


import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VQ_GAN5.train import (
    FixedSemanticVQGAN,
    CaptionGenerator
)

def clean_caption_text(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)
    
   
    text = re.sub(r'<[^>]*>', '', text)
    

    text = re.sub(r'\s+', ' ', text).strip()
    

    text = re.sub(r'[^\x00-\x7F]+', '', text)
    

    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    

    words = text.split()
    if len(words) > 30:  
        text = ' '.join(words[:30]) + '.'
    
    return text.strip()

def safe_decode_text(text: Any) -> str:
   
    if isinstance(text, bytes):
        try:
            text = text.decode('utf-8', errors='ignore')
        except:
            text = str(text)
    elif not isinstance(text, str):
        text = str(text)
    
    return clean_caption_text(text)


class MemoryEfficientMultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
   
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key_value.size(1)
        
     
        residual = query
        
     
        Q = self.qkv_proj(query)[:, :, :self.d_model]
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        

        KV = self.qkv_proj(key_value)[:, :, self.d_model:]
        K, V = KV.chunk(2, dim=-1)
        K = K.view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        
        with autocast(enabled=False):  
            scores = torch.matmul(Q.float(), K.transpose(-2, -1).float()) / math.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            context = torch.matmul(attn_weights, V.float()).to(query.dtype)
        

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.o_proj(context)
        
        
        output = self.layer_norm(output + residual)
        
        return output

class MemoryEfficientQFormerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
     
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
    
        self.cross_attn = MemoryEfficientMultiHeadCrossAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
     
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Linear(d_ff // 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    
        self.use_checkpoint = True
        
    def _forward_impl(self, x: torch.Tensor, visual_features: torch.Tensor, 
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
  
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        
        
        cross_out = self.cross_attn(x, visual_features)
        x = self.norm2(x + cross_out)
        
   
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x
    
    def forward(self, x: torch.Tensor, visual_features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, visual_features, mask, use_reentrant=False)
        else:
            return self._forward_impl(x, visual_features, mask)

class MemoryEfficientQFormer(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.n_layers = config['n_layers']
        self.n_queries = config['n_queries']
        

        self.queries = nn.Parameter(torch.randn(1, self.n_queries, self.d_model) * 0.02)
        
   
        self.shared_projection = nn.Sequential(
            nn.Linear(max(config['codebook_dims'].values()), self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )
        

        self.codebook_adapters = nn.ModuleDict()
        for name, dim in config['codebook_dims'].items():
            self.codebook_adapters[name] = nn.Linear(dim, max(config['codebook_dims'].values()))
        
  
        self.blocks = nn.ModuleList([
            MemoryEfficientQFormerBlock(self.d_model, self.n_heads, self.d_model * 2)
            for _ in range(self.n_layers)
        ])
        
      
        self.output_projection = nn.Linear(self.d_model, config['llama_dim'])
        
    def forward(self, codebook_features: Dict[str, torch.Tensor]) -> torch.Tensor:
       
        if not codebook_features:
            raise ValueError("No codebook features provided")
            
        batch_size = next(iter(codebook_features.values())).size(0)
        device = next(iter(codebook_features.values())).device

        visual_features_list = []
        
        for name, features in codebook_features.items():
            if features is None:
                continue
                
   
            adapted = self.codebook_adapters[name](features)
            
           
            if len(adapted.shape) == 2:
                adapted = adapted.unsqueeze(1)
            
          
            projected = self.shared_projection(adapted)
            visual_features_list.append(projected)
        

        if visual_features_list:
            visual_features = torch.cat(visual_features_list, dim=1)
        else:
          
            visual_features = torch.zeros(batch_size, 1, self.d_model, device=device)
        
       
        del visual_features_list
        torch.cuda.empty_cache()
        
   
        queries = self.queries.expand(batch_size, -1, -1)
        
   
        x = queries
        for block in self.blocks:
            x = block(x, visual_features)
        

        visual_embeds = self.output_projection(x)
        
        return visual_embeds


class VQGANLLaMA3CaptionModelWithLoRA(nn.Module):
 
    def __init__(self, vqgan_config: dict, qformer_config: dict, llama_model_path: str):
        super().__init__()
        
    
        full_vqgan_config = {
            'model': vqgan_config['model'],
            'training': {
                'use_gan': True,
                'gan_loss_type': 'standard',
                'gan_start_step': 5000,
                'd_steps_per_g': 1,
                'loss_weights': {
                    'reconstruction': 1.0,
                    'perceptual': 0.1,
                    'vq': 1.0,
                    'gan': 0.1
                }
            }
        }
        

        self.vqgan = FixedSemanticVQGAN(full_vqgan_config)
        self.vqgan.eval()
        for param in self.vqgan.parameters():
            param.requires_grad = False
        
  
        self.qformer = MemoryEfficientQFormer(qformer_config)
        
   
        print("Loading LLaMA3 model with 4-bit quantization...")
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        

        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
 
        self.llama = get_peft_model(self.llama, lora_config)
        self.llama.print_trainable_parameters()
        
    
        special_tokens = {
            'additional_special_tokens': ['<visual>', '<brief>', '</brief>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.visual_token_id = self.tokenizer.convert_tokens_to_ids("<visual>")
        self.brief_start_token_id = self.tokenizer.convert_tokens_to_ids("<brief>")
        self.brief_end_token_id = self.tokenizer.convert_tokens_to_ids("</brief>")
        self.llama.resize_token_embeddings(len(self.tokenizer))
        
        # 视觉投影层（连接QFormer和LLaMA）
        self.visual_projection = nn.Sequential(
            nn.Linear(qformer_config['n_queries'] * qformer_config['llama_dim'], 
                     qformer_config['llama_dim']),
            nn.LayerNorm(qformer_config['llama_dim']),
            nn.Dropout(0.1)
        )
        
    def extract_visual_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
 
        with torch.no_grad():
            batch_size = images.size(0)
            
   
            if batch_size > 4:
                all_features = defaultdict(list)
                
                for i in range(0, batch_size, 4):
                    batch_images = images[i:i+4]
                    _, info = self.vqgan(batch_images, return_tokens=False)
                    
          
                    if 'quantized_features' in info:
                        for name, features in info['quantized_features'].items():
                            if features is not None:
                                all_features[name].append(features)
                
         
                codebook_features = {}
                for name, features_list in all_features.items():
                    if features_list:
                        codebook_features[name] = torch.cat(features_list, dim=0)
                
    
                del all_features
                torch.cuda.empty_cache()
                
            else:
                _, info = self.vqgan(images, return_tokens=False)
                
               
                if 'quantized_features' in info:
                    codebook_features = {}
                    
                    for name, features in info['quantized_features'].items():
                        if features is not None:
                   
                            if len(features.shape) == 4:  # [B, C, H, W]
                                if name == 'global':
                               
                                    features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
                                else:
                                
                                    b, c, h, w = features.shape
                                    features = features.permute(0, 2, 3, 1).reshape(b, h * w, c)
                            codebook_features[name] = features
                else:
                   
                    codebook_features = {}
                    
                    if 'vq_info' in info:
                        for name in ['global', 'object', 'color', 'spatial', 'texture']:
                            if name in info['vq_info']:
                                vq_info = info['vq_info'][name]
                                
                 
                                quantized = None
                                if isinstance(vq_info, dict):
                                    for key in ['z_q', 'quantized', 'features', 'embeddings']:
                                        if key in vq_info:
                                            quantized = vq_info[key]
                                            break
                                
                                if quantized is not None:
                  
                                    if len(quantized.shape) == 4:  # [B, C, H, W]
                                        if name == 'global':
                                            quantized = F.adaptive_avg_pool2d(quantized, 1).squeeze(-1).squeeze(-1)
                                        else:
                                            b, c, h, w = quantized.shape
                                            quantized = quantized.permute(0, 2, 3, 1).reshape(b, h * w, c)
                                    elif len(quantized.shape) == 2 and name != 'global':  # [B, C]
                                        quantized = quantized.unsqueeze(1) 
                                    
                                    codebook_features[name] = quantized
        
 
        if not codebook_features:
       
            device = images.device
            batch_size = images.size(0)
            for name, dim in self.qformer.config['codebook_dims'].items():
                if name == 'global':
                    codebook_features[name] = torch.randn(batch_size, dim, device=device)
                else:
                    codebook_features[name] = torch.randn(batch_size, 16, dim, device=device)  # 16是序列长度
        
        return codebook_features
    
    def forward(self, images: torch.Tensor, captions: List[str], 
                max_length: int = 32) -> Dict[str, torch.Tensor]:
      
        batch_size = images.size(0)
        device = images.device
        
        # 使用混合精度
        with autocast():
       
            codebook_features = self.extract_visual_features(images)
            
   
            visual_embeds = self.qformer(codebook_features)  
     
            del codebook_features
            torch.cuda.empty_cache()
            
    
            prompts = []
            for _ in range(batch_size):
             
                prompt = "<visual> <brief>Describe the image in one short sentence:"
                prompts.append(prompt)
            
   
            prompt_tokens = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=20  
            ).to(device)
            
            caption_tokens = self.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length, 
                add_special_tokens=False  
            ).to(device)
            
        
            brief_end_tokens = torch.full(
                (batch_size, 1), 
                self.brief_end_token_id, 
                dtype=torch.long, 
                device=device
            )
            
         
            visual_embeds_flat = visual_embeds.reshape(batch_size, -1)  
            visual_embed_projected = self.visual_projection(visual_embeds_flat) 
            
  
            prompt_embeds = self.llama.get_input_embeddings()(prompt_tokens.input_ids)
            

            visual_token_mask = prompt_tokens.input_ids == self.visual_token_id
            for i in range(batch_size):
                visual_pos = visual_token_mask[i].nonzero(as_tuple=True)[0]
                if len(visual_pos) > 0:
                    prompt_embeds[i, visual_pos[0]] = visual_embed_projected[i]
            
 
            full_embeds = torch.cat([
                prompt_embeds,
                self.llama.get_input_embeddings()(caption_tokens.input_ids),
                self.llama.get_input_embeddings()(brief_end_tokens)
            ], dim=1)
            
            full_attention_mask = torch.cat([
                prompt_tokens.attention_mask,
                caption_tokens.attention_mask,
                torch.ones_like(brief_end_tokens)
            ], dim=1)
            

            labels = torch.cat([
                torch.full_like(prompt_tokens.input_ids, -100),
                caption_tokens.input_ids,
                brief_end_tokens
            ], dim=1)
        
      
        outputs = self.llama(
            inputs_embeds=full_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            return_dict=True
        )
        
   
        caption_lengths = (caption_tokens.attention_mask.sum(dim=1).float() - 1).clamp(min=1)
        length_penalty = torch.where(
            caption_lengths > 15, 
            (caption_lengths - 15) * 0.01,  
            torch.zeros_like(caption_lengths)
        )
        
        total_loss = outputs.loss + length_penalty.mean()
        
        return {
            'loss': total_loss,
            'logits': outputs.logits,
            'base_loss': outputs.loss,
            'length_penalty': length_penalty.mean()
        }
    
    @torch.no_grad()
    def generate_caption(self, images: torch.Tensor, max_length: int = 25,  # 减少最大长度
                        temperature: float = 0.7, top_p: float = 0.9) -> List[str]:

        batch_size = images.size(0)
        device = images.device
        

        codebook_features = self.extract_visual_features(images)
        visual_embeds = self.qformer(codebook_features)
        
       
        del codebook_features
        torch.cuda.empty_cache()
        
       
        prompts = []
        for _ in range(batch_size):
          
            prompt = "<visual> <brief>Describe the image in one short sentence:"
            prompts.append(prompt)
            
        prompt_tokens = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=20
        ).to(device)
        

        visual_embeds_flat = visual_embeds.reshape(batch_size, -1)
        visual_embed_projected = self.visual_projection(visual_embeds_flat)
        
        prompt_embeds = self.llama.get_input_embeddings()(prompt_tokens.input_ids)
        visual_token_mask = prompt_tokens.input_ids == self.visual_token_id
        
        for i in range(batch_size):
            visual_pos = visual_token_mask[i].nonzero(as_tuple=True)[0]
            if len(visual_pos) > 0:
                prompt_embeds[i, visual_pos[0]] = visual_embed_projected[i]
        
   
        with autocast():
            outputs = self.llama.generate(
                inputs_embeds=prompt_embeds,
                max_new_tokens=max_length,
                min_new_tokens=5, 
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self.tokenizer.eos_token_id, self.brief_end_token_id],  
                repetition_penalty=1.2, 
            )
        
        
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        

        cleaned_captions = []
        for caption in captions:
      
            caption = caption.replace("<visual>", "").replace("<brief>", "")
            caption = caption.replace("Describe the image in one short sentence:", "").strip()
            
          
            caption = clean_caption_text(caption)
            
            cleaned_captions.append(caption)
        
        return cleaned_captions


class COCOCaptionDataset(Dataset):
   
    def __init__(self, root_dir: str, ann_file: str, transform=None, max_samples: Optional[int] = None):
        self.root_dir = root_dir
        self.transform = transform
        

        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
     
        self.img_id_to_anns = defaultdict(list)
        for ann in self.coco['annotations']:
     
            clean_caption = clean_caption_text(ann['caption'])
            if clean_caption:  
                self.img_id_to_anns[ann['image_id']].append(clean_caption)
        
        
        self.img_id_to_filename = {}
        for img in self.coco['images']:
            self.img_id_to_filename[img['id']] = img['file_name']
        
       
        self.img_ids = [img_id for img_id, anns in self.img_id_to_anns.items() if anns]
        
       
        if max_samples is not None:
            self.img_ids = self.img_ids[:max_samples]
        
        print(f"Loaded {len(self.img_ids)} images with clean captions")
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
   
        img_filename = self.img_id_to_filename[img_id]
        img_path = os.path.join(self.root_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
       
        captions = self.img_id_to_anns[img_id]
        
       
        caption = np.random.choice(captions)
        
        return image, caption, img_id


def custom_collate_fn(batch):
    
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    img_ids = [item[2] for item in batch]
    
    return images, captions, img_ids


def compute_metrics_safe(generated_captions: List[str], reference_captions: List[List[str]]) -> Dict[str, float]:
  
    if not METRICS_AVAILABLE:
        return {'avg_length': np.mean([len(cap.split()) for cap in generated_captions])}
    
   
    cleaned_generated = []
    cleaned_references = []
    
    for gen_cap, ref_caps in zip(generated_captions, reference_captions):
  
        gen_cap_clean = safe_decode_text(gen_cap)
        if not gen_cap_clean or len(gen_cap_clean.strip()) == 0:
            gen_cap_clean = "a photo." 
        
   
        ref_caps_clean = []
        for ref_cap in ref_caps:
            ref_cap_clean = safe_decode_text(ref_cap)
            if ref_cap_clean and len(ref_cap_clean.strip()) > 0:
                ref_caps_clean.append(ref_cap_clean)
        
   
        if not ref_caps_clean:
            ref_caps_clean = ["a photo."]
        
        cleaned_generated.append(gen_cap_clean)
        cleaned_references.append(ref_caps_clean)

    gts = {}
    res = {}
    
    for i, (gen_cap, ref_caps) in enumerate(zip(cleaned_generated, cleaned_references)):
        gts[i] = ref_caps
        res[i] = [gen_cap]
    
    metrics = {}
    

    gen_lengths = [len(cap.split()) for cap in cleaned_generated]
    metrics['avg_gen_length'] = np.mean(gen_lengths)
    metrics['std_gen_length'] = np.std(gen_lengths)
    

    try:
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts, res)
        for i, score in enumerate(bleu_scores):
            metrics[f'BLEU-{i+1}'] = float(score)
    except Exception as e:
        print(f"BLEU computation failed: {e}")
      
        metrics['BLEU-1'] = 0.0
        metrics['BLEU-4'] = 0.0
    

    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(gts, res)
        metrics['CIDEr'] = float(cider_score)
    except Exception as e:
        print(f"CIDEr computation failed: {e}")
        metrics['CIDEr'] = 0.0
    
   
    try:
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts, res)
        metrics['ROUGE-L'] = float(rouge_score)
    except Exception as e:
        print(f"ROUGE computation failed: {e}")
        metrics['ROUGE-L'] = 0.0
    

    print("Skipping METEOR computation due to encoding issues")
    
    return metrics


def train_qformer_llama3_memory_optimized(config: dict):

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    torch.backends.cudnn.benchmark = True
    

    os.makedirs(config['output_dir'], exist_ok=True)
    
 
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    

    train_dataset = COCOCaptionDataset(
        root_dir=config['data']['train_dir'],
        ann_file=config['data']['train_ann_file'],
        transform=transform,
        max_samples=config['data'].get('max_samples')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  
        collate_fn=custom_collate_fn  
    )
    
   
    if config['data'].get('val_ann_file'):
        val_dataset = COCOCaptionDataset(
            root_dir=config['data']['val_dir'],
            ann_file=config['data']['val_ann_file'],
            transform=transform,
            max_samples=50  
        )
    
 
    print("Creating model...")
    model = VQGANLLaMA3CaptionModelWithLoRA(
        vqgan_config=config['vqgan'],
        qformer_config=config['qformer'],
        llama_model_path=config['llama_model_path']
    )
    

    if config['vqgan_checkpoint']:
        print(f"Loading VQGAN checkpoint from {config['vqgan_checkpoint']}")
        vqgan_state = torch.load(config['vqgan_checkpoint'], map_location='cpu')
        model.vqgan.load_state_dict(vqgan_state['model_state_dict'], strict=False)
        del vqgan_state
        gc.collect()
    
    model = model.to(device)
    

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    

    optimizer_grouped_parameters = [
        {
            'params': model.qformer.parameters(),
            'lr': config['training']['learning_rate']
        },
        {
            'params': model.visual_projection.parameters(),
            'lr': config['training']['learning_rate']
        },
        {
            'params': model.llama.parameters(),
            'lr': config['training']['learning_rate'] * 0.1  # LoRA参数使用较小学习率
        }
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        weight_decay=config['training']['weight_decay']
    )
    

    scaler = GradScaler()
    
    
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    num_warmup_steps = int(0.1 * num_training_steps)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
  
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 4)
    
 
    print("\n Starting training...")
    global_step = 0
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_losses = []
        epoch_length_penalties = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}") as pbar:
            for batch_idx, (images, captions, img_ids) in enumerate(pbar):
                images = images.to(device)
                
          
                with autocast():
                    outputs = model(images, captions)
                    loss = outputs['loss'] / gradient_accumulation_steps
                
    
                scaler.scale(loss).backward()
                
        
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
            
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
                    
               
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    
               
                    epoch_losses.append(outputs['base_loss'].item())
                    if 'length_penalty' in outputs:
                        epoch_length_penalties.append(outputs['length_penalty'].item())
                
                    pbar.set_postfix({
                        'loss': f"{outputs['base_loss'].item():.4f}",
                        'len_pen': f"{outputs.get('length_penalty', 0).item():.4f}",
                        'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
                    })
                
         
                if global_step % config['training']['print_interval'] == 0 and global_step > 0:
                    model.eval()
                    with torch.no_grad():
                     
                        sample_images = images[:2]
                        sample_captions = captions[:2]
                        
                        generated_captions = model.generate_caption(sample_images)
                        
                        print(f"\n{'='*80}")
                        print(f"[Step {global_step}] Caption Comparison:")
                        print(f"{'='*80}")
                        
                        for i in range(len(generated_captions)):
                            print(f"\n Image {i+1}:")
                            print(f" Real Caption: {sample_captions[i]}")
                            print(f" Generated Caption: {generated_captions[i]}")
                            print(f" Generated Length: {len(generated_captions[i].split())} words")
                        
                 
                        if METRICS_AVAILABLE and config['data'].get('val_ann_file'):
                            print("\n Computing metrics on validation set...")
                            val_generated = []
                            val_references = []
                            
                            try:
                               
                                for i in range(min(10, len(val_dataset))): 
                                    val_img, _, val_img_id = val_dataset[i]
                                    val_img = val_img.unsqueeze(0).to(device)
                                    
                                    gen_cap = model.generate_caption(val_img)[0]
                                    ref_caps = val_dataset.img_id_to_anns[val_img_id]
                                    
                                    val_generated.append(gen_cap)
                                    val_references.append(ref_caps)
                                
                               
                                metrics = compute_metrics_safe(val_generated, val_references)
                                
                                print("\n Validation Metrics:")
                                for metric_name, score in metrics.items():
                                    print(f"  {metric_name}: {score:.4f}")
                                    
                            except Exception as e:
                                print(f"Metrics computation failed: {e}")
                                print("Continuing training without metrics...")
                        
                        print(f"{'='*80}\n")
                    
                    model.train()
                
                global_step += 1
                
               
                if global_step % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
       
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_length_penalty = np.mean(epoch_length_penalties) if epoch_length_penalties else 0
        print(f"\n Epoch {epoch+1} completed.")
        print(f"   Average loss: {avg_epoch_loss:.4f}")
        print(f"   Average length penalty: {avg_length_penalty:.4f}")
        
 
        if (epoch + 1) % config['training']['checkpoint_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'qformer_state_dict': model.qformer.state_dict(),
                'visual_projection_state_dict': model.visual_projection.state_dict(),
                'llama_lora_state_dict': model.llama.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt'))
            print(f" Saved checkpoint for epoch {epoch+1}")
            
          
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":

    config = {
  
        'vqgan': {
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
            }
        },
        

        'vqgan_checkpoint': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/VQ_GAN5/outputs/checkpoint_epoch_60.pt',
        
 
        'qformer': {
            'd_model': 512, 
            'n_heads': 8,    
            'n_layers': 4,   
            'n_queries': 32,
            'codebook_dims': {
                'global': 768,
                'object': 512,
                'color': 256,
                'spatial': 384,
                'texture': 256
            },
            'llama_dim': 4096
        },
        
      
        'llama_model_path': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/Meta-Llama-3-8B-Instruct',  
        
    
        'data': {
            'train_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/train/data',
            'train_ann_file': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/raw/captions_train2017.json', 
            'val_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/validation/data',
            'val_ann_file': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/coco-2017/raw/captions_val2017.json',  
            'image_size': 128,
            'num_workers': 2, 
            'max_samples': 120000,  
        },
        
       
        'training': {
            'batch_size': 2,  
            'gradient_accumulation_steps': 8,  
            'num_epochs': 10,
            'learning_rate': 2e-4,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'print_interval': 10000,
            'checkpoint_interval': 2,
        },
        
     
        'output_dir': '/media/a/484ab06d-9814-4785-89b6-d05139c8bca2/a/PengYX/VQ-LLM/Former4/checkpoint'
    }
    

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
  
    train_qformer_llama3_memory_optimized(config)
