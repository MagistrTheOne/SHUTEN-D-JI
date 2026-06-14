"""
Persistent Memory for SHUTEN-DŌJI.

Learned working memory that persists across generation steps.
The model decides what to store (write gate) and reads via cross-attention.
NOT retrieval augmentation — this is end-to-end trained working memory.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentMemory(nn.Module):
    """
    Working memory with gated write and cross-attention read.

    Memory slots persist across forward passes (multi-turn interactions).
    The model learns when and what to memorize.
    """

    def __init__(self, config):
        super().__init__()
        self.num_slots = config.num_memory_slots
        self.memory_dim = config.memory_dim
        self.hidden_size = config.hidden_size

        # Memory bank — initialized as learnable parameters, updated during inference
        self.memory_slots = nn.Parameter(
            torch.randn(1, self.num_slots, self.memory_dim) * 0.02
        )

        # Read: cross-attention from hidden states to memory
        self.read_query = nn.Linear(config.hidden_size, self.memory_dim, bias=False)
        self.read_key = nn.Linear(self.memory_dim, self.memory_dim, bias=False)
        self.read_value = nn.Linear(self.memory_dim, config.hidden_size, bias=False)
        self.read_gate = nn.Linear(config.hidden_size, 1, bias=True)

        # Write: gate + projection from hidden states to memory
        self.write_proj = nn.Linear(config.hidden_size, self.memory_dim, bias=False)
        self.write_gate = nn.Linear(config.hidden_size, self.num_slots, bias=True)

        # Decay factor for older memories
        self.decay = nn.Parameter(torch.tensor(0.95))

    def read(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Cross-attend to memory slots, gated addition to hidden states.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            hidden_states + gated memory contribution
        """
        bsz = hidden_states.shape[0]
        memory = self.memory_slots.expand(bsz, -1, -1)

        # Cross-attention
        q = self.read_query(hidden_states)  # [bsz, seq, mem_dim]
        k = self.read_key(memory)  # [bsz, num_slots, mem_dim]
        v = self.read_value(memory)  # [bsz, num_slots, hidden_size]

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.memory_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        memory_output = torch.matmul(attn_weights, v)  # [bsz, seq, hidden_size]

        # Gated addition
        gate = torch.sigmoid(self.read_gate(hidden_states))
        return hidden_states + gate * memory_output

    def write(self, hidden_states: torch.Tensor) -> None:
        """
        Update memory slots based on current hidden states.
        Uses gated write — model decides what to remember.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Pool across sequence for write candidates
        # Use last token as summary (causal models attend to everything before)
        summary = hidden_states[:, -1, :]  # [bsz, hidden_size]

        # Project to memory dimension
        write_content = self.write_proj(summary)  # [bsz, memory_dim]

        # Write gate — which slots to update
        gate = torch.sigmoid(self.write_gate(summary))  # [bsz, num_slots]

        # Decay existing memory, add new content
        with torch.no_grad():
            decay = torch.clamp(self.decay, 0.8, 0.99)
            new_memory = decay * self.memory_slots.data

            # Write new content to most-activated slots
            write_update = gate.unsqueeze(-1) * write_content.unsqueeze(1)
            new_memory = new_memory + (1 - decay) * write_update.mean(dim=0, keepdim=True)
            self.memory_slots.data.copy_(new_memory)

    def reset(self) -> None:
        """Reset memory slots (between unrelated conversations)."""
        nn.init.normal_(self.memory_slots, mean=0.0, std=0.02)
