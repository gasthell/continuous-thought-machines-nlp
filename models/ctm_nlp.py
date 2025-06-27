import torch
import torch.nn as nn
import numpy as np
import math

from models.ctm import ContinuousThoughtMachine
from models.modules import Identity

class CTM_NLP(ContinuousThoughtMachine):
    """
    Adaptation of Continuous Thought Machine for natural language processing (NLP) tasks.

    This class inherits from ContinuousThoughtMachine and replaces the image-specific backend (ResNet) with standard NLP layers:

    1. Token Embeddings

    2. Positional Embeddings

    The internal mechanics of CTM "thinking" (recurrence, NLM, synchronization)
    remain unchanged. The model takes `input_ids` (token indices) as input
    and generates predictions at each "thought step".

    Args:
    vocab_size (int): Vocabulary size for token embeddings.
    max_seq_len (int): Maximum sequence length for positional embeddings.

    # --- Arguments inherited from ContinuousThoughtMachine ---
    iterations (int): Number of internal 'thought steps' (T).
    d_model (int): Dimensionality of internal latent space (D).
    d_input (int): Dimensionality of embeddings and attention outputs. Should be equal to d_embedding.
    heads (int): Number of attention heads.
    n_synch_out (int): Number of neurons for output synchronization.
    n_synch_action (int): Number of neurons for action/attention synchronization.
    synapse_depth (int): Depth of synapse model (U-Net).
    memory_length (int): History length for Neuron-Level Models (M).
    deep_nlms (bool): Whether to use deep (2-layer) NLMs.
    memory_hidden_dims (int): Hidden dimension for deep NLMs.
    dropout (float): Dropout.
    ... (and other base class arguments)
    """
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 # CTM Arguments
                 iterations: int,
                 d_model: int,
                 d_input: int,
                 heads: int,
                 n_synch_out: int,
                 n_synch_action: int,
                 synapse_depth: int,
                 memory_length: int,
                 deep_nlms: bool,
                 memory_hidden_dims: int,
                 dropout: float = 0.1,
                 **kwargs):
        
        # --- 1 step: Initialization CTM ---
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            dropout=dropout,
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=vocab_size,
            **kwargs
        )
        
        print("Initializing CTM for NLP tasks...")

        # --- 2 step: Swapping backend layers to NLP-specified layers ---
        self.token_embedding = nn.Embedding(vocab_size, d_input)
        self.position_embedding_nlp = nn.Embedding(max_seq_len, d_input)
        
        self.backbone = Identity()
        self.positional_embedding = Identity()
        
        print(f"CTM_NLP initialized with vocab_size={vocab_size}, max_seq_len={max_seq_len}")
        print(f"Output projection layer will map to {self.output_projector[0].out_features} logits.")

    def compute_features(self, input_ids: torch.Tensor):
        """
        Override the method to compute features from text data.
        Instead of running the image through ResNet, we get token embeddings.

        Args:
        input_ids (torch.Tensor): Tensor with token indices.
        Shape: (batch_size, sequence_length).

        Returns:
        torch.Tensor: Keys/values ​​(kv) for the attention mechanism.
        Shape: (batch_size, sequence_length, d_input).
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # 1. Получаем эмбеддинги токенов
        token_embeds = self.token_embedding(input_ids) # (B, S, D_input)
        
        # 2. Создаем и получаем позиционные эмбеддинги
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, -1) # (B, S)
        position_embeds = self.position_embedding_nlp(positions) # (B, S, D_input)
        
        # 3. Суммируем эмбеддинги
        combined_features = token_embeds + position_embeds # (B, S, D_input)
        
        # 4. Проецируем в key-value пространство для внимания
        # Слой kv_proj уже создан в родительском классе __init__
        kv = self.kv_proj(combined_features) # (B, S, D_input)
        
        return kv

    def forward(self, input_ids: torch.Tensor, track=False):
        """
        Override forward to explicitly specify that input_ids are passed as input.
        Internal logic calls `super().forward`, which will execute the entire CTM
        recursion loop.

        Args:
        input_ids (torch.Tensor): Tensor with token indices.
        Shape: (batch_size, sequence_length).
        track (bool): Flag for tracking internal states.

        Returns:
        Tuple: Tuple with predictions, confidence, and other debug data,
        same as parent class.
        """
        return super().forward(input_ids, track=track)