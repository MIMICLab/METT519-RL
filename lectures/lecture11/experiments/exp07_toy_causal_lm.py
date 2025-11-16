#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 07 - Toy Causal Language Model for DPO

Implements a minimal causal language model and synthetic preference dataset
for demonstrating Direct Preference Optimization (DPO) without requiring
large models or datasets.

Learning objectives:
- Build minimal transformer/GRU language model
- Create synthetic preference datasets
- Understand tokenization and sequence processing
- Prepare foundation for DPO experiments

Prerequisites: PyTorch 2.x, transformers (optional)
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
import json
import string
from collections import defaultdict

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available()
setup_seed(42)

class SimpleTokenizer:
    """
    Minimal character-level tokenizer for toy experiments.
    
    Vocabulary:
    - Characters: a-z, A-Z, 0-9, space, punctuation
    - Special tokens: <PAD>, <BOS>, <EOS>, <UNK>
    """
    
    def __init__(self, vocab_size: int = 512):
        # Define base character vocabulary
        chars = string.ascii_letters + string.digits + string.punctuation + ' '
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
        
        # Build vocabulary
        vocab_chars = list(set(chars))[:vocab_size - len(self.special_tokens)]
        self.vocab = self.special_tokens + vocab_chars
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        self.vocab_size = len(self.vocab)
        self.pad_id = self.token_to_id['<PAD>']
        self.bos_id = self.token_to_id['<BOS>']
        self.eos_id = self.token_to_id['<EOS>']
        self.unk_id = self.token_to_id['<UNK>']
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_id)
        
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_id)
        
        if add_special_tokens:
            tokens.append(self.eos_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        text = ""
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                text += token
            else:
                text += "<UNK>"
        return text
    
    def __call__(self, texts: List[str], max_length: int = 64, 
                 padding: bool = True, truncation: bool = True,
                 return_tensors: str = 'pt') -> Dict[str, torch.Tensor]:
        """Tokenize batch of texts."""
        encoded_texts = []
        attention_masks = []
        
        for text in texts:
            tokens = self.encode(text, add_special_tokens=True)
            
            # Truncate if necessary
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length-1] + [self.eos_id]
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * len(tokens)
            
            # Pad if necessary
            if padding and len(tokens) < max_length:
                pad_length = max_length - len(tokens)
                tokens.extend([self.pad_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            encoded_texts.append(tokens)
            attention_masks.append(attention_mask)
        
        result = {
            'input_ids': torch.tensor(encoded_texts, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
        }
        
        return result

class SimpleTransformerLM(nn.Module):
    """
    Minimal causal transformer language model.
    
    Architecture:
    - Token embeddings + positional embeddings
    - Multi-head self-attention layers with causal masking
    - Feed-forward networks
    - Layer normalization
    - Output projection to vocabulary
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 2, max_seq_length: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            
        Returns:
            logits: [B, L, V] vocabulary logits
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)  # [B, L, E]
        position_embeds = self.position_embedding(position_ids)  # [B, L, E]
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(batch_size, seq_length, input_ids.device)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to match causal mask format
            extended_attention_mask = attention_mask[:, None, None, :].float()
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            causal_mask = causal_mask + extended_attention_mask
        
        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_mask)
        
        # Final layer norm and projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)  # [B, L, V]
        
        return logits
    
    def create_causal_mask(self, batch_size: int, seq_length: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        mask = mask * -10000.0  # Large negative values for masked positions
        return mask.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)  # [B, 1, L, L]

class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: [B, L, E] input embeddings
            attention_mask: [B, 1, L, L] attention mask
            
        Returns:
            [B, L, E] output embeddings
        """
        # Self-attention with residual connection
        attn_x = self.ln_1(x)
        # For MultiheadAttention with multiple heads, expand mask: [B*num_heads, L, L]
        batch_size, seq_len = attn_x.shape[:2]
        num_heads = self.attn.num_heads
        mask_expanded = attention_mask.squeeze(1).unsqueeze(1).repeat(1, num_heads, 1, 1)
        mask_expanded = mask_expanded.view(batch_size * num_heads, seq_len, seq_len)
        attn_output, _ = self.attn(attn_x, attn_x, attn_x, 
                                  attn_mask=mask_expanded,
                                  need_weights=False)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_x = self.ln_2(x)
        ff_output = self.mlp(ff_x)
        x = x + ff_output
        
        return x

class PreferenceDataset:
    """Synthetic preference dataset for DPO experiments."""
    
    def __init__(self, tokenizer: SimpleTokenizer, size: int = 1000):
        self.tokenizer = tokenizer
        self.size = size
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic preference data."""
        data = []
        
        # Template-based generation for consistency
        templates = [
            "The weather today is {weather}",
            "I like {food} for dinner",
            "My favorite color is {color}",
            "The best movie is {movie}",
            "I prefer {activity} over {alternative}"
        ]
        
        # Response quality patterns
        good_patterns = [
            "sunny and beautiful",
            "pasta with fresh herbs",
            "deep blue like the ocean",
            "a classic story with great acting",
            "reading books to watching TV"
        ]
        
        bad_patterns = [
            "bad weather stuff",
            "food thing",
            "some color",
            "movie or whatever",
            "doing stuff instead of other stuff"
        ]
        
        for i in range(self.size):
            # Create context (input prompt)
            template_idx = i % len(templates)
            template = templates[template_idx]
            
            if "{" in template and "}" in template:
                # Extract placeholder
                start = template.find("{")
                end = template.find("}")
                placeholder = template[start+1:end]
                context = template[:start] + "{" + placeholder + "}"
            else:
                context = template
            
            # Create preferred and dispreferred responses
            good_response = good_patterns[template_idx % len(good_patterns)]
            bad_response = bad_patterns[template_idx % len(bad_patterns)]
            
            # Randomly swap to avoid positional bias
            if random.random() > 0.5:
                y_pos, y_neg = good_response, bad_response
            else:
                y_pos, y_neg = bad_response, good_response
                # Actually, let's keep good as preferred for clarity
                y_pos, y_neg = good_response, bad_response
            
            data.append({
                'input': context,
                'chosen': y_pos,
                'rejected': y_neg
            })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    
    def get_batch(self, batch_size: int, max_length: int = 64) -> Dict[str, torch.Tensor]:
        """Get a batch of tokenized preference data."""
        batch_indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        
        inputs, chosen, rejected = [], [], []
        
        for idx in batch_indices:
            item = self.data[idx]
            inputs.append(item['input'])
            chosen.append(item['chosen'])
            rejected.append(item['rejected'])
        
        # Tokenize
        input_tokens = self.tokenizer(inputs, max_length=max_length, 
                                     padding=True, truncation=True)
        chosen_tokens = self.tokenizer(chosen, max_length=max_length,
                                      padding=True, truncation=True)
        rejected_tokens = self.tokenizer(rejected, max_length=max_length,
                                        padding=True, truncation=True)
        
        return {
            'input_ids': input_tokens['input_ids'].to(device),
            'input_attention_mask': input_tokens['attention_mask'].to(device),
            'chosen_ids': chosen_tokens['input_ids'].to(device),
            'chosen_attention_mask': chosen_tokens['attention_mask'].to(device),
            'rejected_ids': rejected_tokens['input_ids'].to(device),
            'rejected_attention_mask': rejected_tokens['attention_mask'].to(device)
        }

def compute_log_likelihood(model: nn.Module, input_ids: torch.Tensor, 
                          attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute log-likelihood of a sequence.
    
    Args:
        model: Language model
        input_ids: [B, L] token IDs
        attention_mask: [B, L] attention mask
        
    Returns:
        log_likelihood: [B] log-likelihood per sequence
    """
    with torch.no_grad():
        logits = model(input_ids, attention_mask)  # [B, L, V]
        
        # Shift logits and labels for causal LM
        shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, V]
        shift_labels = input_ids[:, 1:].contiguous()   # [B, L-1]
        shift_mask = attention_mask[:, 1:].contiguous()  # [B, L-1]
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        gathered_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Apply mask and sum
        masked_log_probs = gathered_log_probs * shift_mask.float()
        sequence_log_likelihood = masked_log_probs.sum(dim=-1)  # [B]
        
        return sequence_log_likelihood

def test_tokenizer():
    """Test the simple tokenizer."""
    print("Testing simple tokenizer...")
    
    tokenizer = SimpleTokenizer(vocab_size=128)
    
    # Test basic encoding/decoding
    text = "Hello world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    assert len(tokens) > 0, "Should produce tokens"
    assert decoded.replace(" ", "") == text.replace(" ", ""), "Should decode properly"
    
    # Test batch tokenization
    texts = ["Hello", "world", "test"]
    batch = tokenizer(texts, max_length=10, padding=True)
    
    assert batch['input_ids'].shape[0] == 3, "Should have 3 sequences"
    assert batch['input_ids'].shape[1] == 10, "Should be padded to max_length"
    assert batch['attention_mask'].shape == batch['input_ids'].shape, "Attention mask shape should match"
    
    print(f"  Tokenizer: ✓ (vocab_size: {tokenizer.vocab_size})")

def test_transformer_model():
    """Test the transformer language model."""
    print("Testing transformer model...")
    
    tokenizer = SimpleTokenizer(vocab_size=128)
    model = SimpleTransformerLM(vocab_size=tokenizer.vocab_size, embed_dim=64, 
                               num_heads=2, num_layers=2).to(device)
    
    # Test forward pass
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 16)).to(device)  # [B=2, L=16]
    attention_mask = torch.ones_like(input_ids).to(device)
    
    logits = model(input_ids, attention_mask)
    
    assert logits.shape == (2, 16, tokenizer.vocab_size), f"Wrong logits shape: {logits.shape}"
    
    # Test parameter count
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    
    # Test log-likelihood computation
    log_likelihood = compute_log_likelihood(model, input_ids, attention_mask)
    assert log_likelihood.shape == (2,), f"Wrong log-likelihood shape: {log_likelihood.shape}"
    
    print("  Transformer model: ✓")

def test_preference_dataset():
    """Test preference dataset generation."""
    print("Testing preference dataset...")
    
    tokenizer = SimpleTokenizer(vocab_size=128)
    dataset = PreferenceDataset(tokenizer, size=50)
    
    assert len(dataset) == 50, f"Wrong dataset size: {len(dataset)}"
    
    # Test data item format
    item = dataset[0]
    assert 'input' in item and 'chosen' in item and 'rejected' in item
    assert isinstance(item['input'], str)
    assert isinstance(item['chosen'], str)
    assert isinstance(item['rejected'], str)
    
    # Test batch generation
    batch = dataset.get_batch(batch_size=4, max_length=32)
    
    expected_keys = ['input_ids', 'input_attention_mask', 'chosen_ids', 
                    'chosen_attention_mask', 'rejected_ids', 'rejected_attention_mask']
    for key in expected_keys:
        assert key in batch, f"Missing key: {key}"
        assert batch[key].shape[0] == 4, f"Wrong batch size for {key}"
    
    print("  Preference dataset: ✓")

def demonstrate_usage():
    """Demonstrate the toy language model usage."""
    print("\nDemonstrating toy language model:")
    print("="*40)
    
    # Setup
    tokenizer = SimpleTokenizer(vocab_size=256)
    model = SimpleTransformerLM(
        vocab_size=tokenizer.vocab_size, 
        embed_dim=128, 
        num_heads=4, 
        num_layers=3
    ).to(device)
    
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Generate some text (random sampling)
    model.eval()
    prompt = "The weather today is"
    print(f"\nPrompt: '{prompt}'")
    
    # Tokenize prompt
    input_tokens = tokenizer([prompt], max_length=32)
    input_ids = input_tokens['input_ids'].to(device)
    
    # Generate continuation (greedy)
    generated_ids = input_ids.clone()
    max_new_tokens = 10
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Stop at EOS token
            if next_token.item() == tokenizer.eos_id:
                break
                
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    print(f"Generated: '{generated_text}'")
    
    # Demonstrate preference data
    print(f"\nPreference dataset samples:")
    dataset = PreferenceDataset(tokenizer, size=10)
    
    for i in range(3):
        item = dataset[i]
        print(f"  Input: {item['input']}")
        print(f"  Chosen: {item['chosen']}")
        print(f"  Rejected: {item['rejected']}")
        print()

def main():
    print("="*60)
    print("Experiment 07: Toy Causal Language Model for DPO")
    print("="*60)
    
    # Run tests
    test_tokenizer()
    test_transformer_model()
    test_preference_dataset()
    
    print("\nAll tests passed! ✓")
    
    # Demonstrate usage
    demonstrate_usage()
    
    print(f"\nToy language model ready for DPO!")
    print("Model specifications:")
    print(f"  Architecture: Causal transformer with multi-head attention")
    print(f"  Vocabulary: Character-level tokenization (~256-512 tokens)")
    print(f"  Preference data: Synthetic template-based pairs")
    print(f"  Log-likelihood: Efficient sequence probability computation")
    print(f"  Device: {device}")

if __name__ == "__main__":
    main()