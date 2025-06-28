import torch
import torch.nn as nn
import numpy as np
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.ctm import ContinuousThoughtMachine
from models.modules import Identity, Squeeze

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
                 # --- NLP-Specific Arguments ---
                 vocab_size: int,
                 num_classes: int,
                 max_seq_len: int = 512,
                 padding_idx: int = 0,
                 
                 # --- CTM Core Arguments ---
                 iterations: int = 16,
                 d_model: int = 1024,
                 d_input: int = 768,
                 heads: int = 12,
                 synapse_depth: int = 6, # Depth of the new Transformer synapse
                 n_synch_out: int = 256,
                 n_synch_action: int = 256,
                 memory_length: int = 16,
                 deep_nlms: bool = True,
                 memory_hidden_dims: int = 128,
                 do_layernorm_nlm: bool = True,
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
            synapse_depth=synapse_depth, # Pass this along
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            do_layernorm_nlm=do_layernorm_nlm,
            dropout=dropout,
            # out_dims=vocab_size,
            out_dims=num_classes,
            # CRITICAL: Disable vision-specific components in the base class
            backbone_type='none',
            positional_embedding_type='none',
            # We will handle synapses ourselves
            **kwargs 
        )

        print("Initializing CTM for NLP...")

        # --- 2. Create NLP-Native Input Layers ---
        self.token_embedding = nn.Embedding(vocab_size, d_input, padding_idx=padding_idx)
        
        # Using fixed sinusoidal positional embeddings - a SOTA standard
        pe = self._create_sinusoidal_positional_embedding(max_seq_len, d_input)
        self.register_buffer('nlp_positional_embedding', pe)
        
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_layernorm = nn.LayerNorm(d_input)

        # --- 3. Override the Synapse Model with a Transformer Encoder ---
        print(f"Replacing SynapseUnet with TransformerEncoder ({synapse_depth} layers)...")
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, # The synapse operates on the core latent state 'z'
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-LN is a modern best practice
        )
        self.synapses = TransformerEncoder(encoder_layer, num_layers=synapse_depth)

        # --- 4. Override the Output Projector for Language Modeling ---
        self.trace_processor = nn.Sequential(nn.Linear(memory_length, 1), Squeeze(-1), nn.Tanh())

        # --- Override Output Projector for Classification ---
        self.output_projector = nn.Linear(self.synch_representation_size_out, num_classes)
        print(f"Model configured for {num_classes}-class classification.")

    def _create_sinusoidal_positional_embedding(self, max_len: int, d_model: int):
        """Creates a static sinusoidal positional embedding matrix."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        return pe

    def compute_features(self, input_ids: torch.Tensor):
        """
        Override the feature computation for text data.
        This method converts token IDs into embeddings for the CTM attention mechanism.
        """
        seq_length = input_ids.shape[1]
        
        # 1. Get token embeddings
        token_embeds = self.token_embedding(input_ids) # (B, S, D_input)
        
        # 2. Add sinusoidal positional embeddings
        pos_embeds = self.nlp_positional_embedding[:, :seq_length, :]
        
        # 3. Combine and apply LayerNorm + Dropout (standard Transformer practice)
        combined_features = self.embedding_layernorm(token_embeds + pos_embeds)
        combined_features = self.embedding_dropout(combined_features)
        
        # 4. Project into key-value space for the CTM's attention mechanism
        # The kv_proj layer is already created in the parent class __init__
        kv = self.kv_proj(combined_features) # (B, S, D_input)
        
        return kv

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, track: bool = False):
        """
        The main forward pass for the NLP-adapted CTM.

        Args:
            input_ids (torch.Tensor): Token indices. Shape: (batch_size, sequence_length).
            attention_mask (torch.Tensor, optional): Mask to avoid attending to padding tokens.
                                                     Shape: (batch_size, sequence_length).
                                                     `1` for tokens to attend to, `0` for padding.
            track (bool): Flag to track internal CTM states for analysis.

        Returns:
            Tuple: (predictions, certainties, final_synchronization_state)
        """
        # We explicitly pass the input_ids and the crucial attention_mask to the
        # base class's forward method. This requires a small modification to the
        # base class to accept the mask.
        return super().forward(x=input_ids, attention_mask=attention_mask, track=track)
