"""
SHUTEN-DŌJI Model Architecture

Custom Mixture-of-Experts transformer designed for strategic intelligence.
NOT based on any pretrained architecture — derived from the training objective:
  world understanding + simulation + planning + prediction + agentic execution.

Key innovations:
  - Cognitive expert groups (soft-partitioned by reasoning mode)
  - Typed token routing (token type biases expert selection)
  - Persistent memory slots (learned working memory across turns)
  - GQA attention with RoPE for long context
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.routing import CognitiveRouter, TokenType
from src.model.experts import ExpertLayer, SharedExpert
from src.model.memory import PersistentMemory


@dataclass
class ShutenDojiConfig:
    """Model configuration — all hyperparameters in one place."""

    # Core dimensions
    hidden_size: int = 5120
    intermediate_size: int = 13824  # SwiGLU: 2/3 * 4 * hidden_size rounded
    num_layers: int = 48
    vocab_size: int = 128256  # 128K + special tokens

    # Attention
    num_attention_heads: int = 48
    num_kv_heads: int = 8  # GQA ratio 6:1
    head_dim: int = 128  # hidden_size // num_attention_heads ~= 106, use power-of-2
    max_position_embeddings: int = 131072  # 128K context
    rope_theta: float = 500000.0
    rope_scaling: Optional[dict] = None  # YaRN config for extended context

    # MoE
    num_experts: int = 128
    num_shared_experts: int = 2
    num_experts_per_token: int = 6
    expert_hidden_size: int = 2048
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001

    # Cognitive expert groups (soft assignment, influences initialization + aux loss)
    expert_groups: dict = field(default_factory=lambda: {
        "state_encoding": list(range(0, 20)),
        "analysis": list(range(20, 45)),
        "simulation": list(range(45, 70)),
        "planning": list(range(70, 95)),
        "evaluation": list(range(95, 115)),
        "general": list(range(115, 128)),
    })

    # Memory
    num_memory_slots: int = 256
    memory_dim: int = 512

    # Token types (for typed routing)
    num_token_types: int = 9  # state, analysis, simulation, planning, prediction, memory, tool, role, critique

    # Training
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    # Dense layers (first N layers are dense, no MoE)
    num_dense_layers: int = 1

    @property
    def total_params_estimate(self) -> int:
        """Rough parameter count estimate."""
        embed = self.vocab_size * self.hidden_size * 2  # embed + lm_head
        attn_per_layer = (
            self.hidden_size * self.head_dim * self.num_attention_heads  # Q
            + self.hidden_size * self.head_dim * self.num_kv_heads  # K
            + self.hidden_size * self.head_dim * self.num_kv_heads  # V
            + self.head_dim * self.num_attention_heads * self.hidden_size  # O
        )
        expert_params = 3 * self.hidden_size * self.expert_hidden_size  # gate+up+down per expert
        moe_per_layer = self.num_experts * expert_params + self.num_shared_experts * expert_params
        dense_ffn = 3 * self.hidden_size * self.intermediate_size
        total = embed
        total += self.num_dense_layers * (attn_per_layer + dense_ffn)
        total += (self.num_layers - self.num_dense_layers) * (attn_per_layer + moe_per_layer)
        return total

    @property
    def active_params_estimate(self) -> int:
        """Active parameters per forward pass."""
        embed = self.vocab_size * self.hidden_size * 2
        attn_per_layer = (
            self.hidden_size * self.head_dim * self.num_attention_heads
            + self.hidden_size * self.head_dim * self.num_kv_heads * 2
            + self.head_dim * self.num_attention_heads * self.hidden_size
        )
        active_expert_params = (
            (self.num_experts_per_token + self.num_shared_experts)
            * 3 * self.hidden_size * self.expert_hidden_size
        )
        dense_ffn = 3 * self.hidden_size * self.intermediate_size
        total = embed
        total += self.num_dense_layers * (attn_per_layer + dense_ffn)
        total += (self.num_layers - self.num_dense_layers) * (attn_per_layer + active_expert_params)
        return total


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE implementation with optional YaRN scaling."""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 500000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(position_ids.float().squeeze(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GQAttention(nn.Module):
    """Grouped Query Attention with RoPE."""

    def __init__(self, config: ShutenDojiConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)

        # Expand KV for GQA
        k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output), past_key_value


class DenseFFN(nn.Module):
    """SwiGLU feed-forward for dense layers."""

    def __init__(self, config: ShutenDojiConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """Mixture-of-Experts feed-forward with cognitive routing."""

    def __init__(self, config: ShutenDojiConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.router = CognitiveRouter(config, layer_idx)
        self.experts = nn.ModuleList([
            ExpertLayer(config.hidden_size, config.expert_hidden_size)
            for _ in range(config.num_experts)
        ])
        self.shared_experts = nn.ModuleList([
            SharedExpert(config.hidden_size, config.expert_hidden_size)
            for _ in range(config.num_shared_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: scalar router balancing loss
        """
        bsz, seq_len, hidden_dim = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_dim)

        # Route tokens to experts
        router_output = self.router(flat_hidden, token_type_ids)
        expert_indices = router_output.expert_indices  # [num_tokens, top_k]
        expert_weights = router_output.expert_weights  # [num_tokens, top_k]
        aux_loss = router_output.aux_loss

        # Compute expert outputs
        output = torch.zeros_like(flat_hidden)

        for idx in range(self.config.num_experts_per_token):
            expert_idx = expert_indices[:, idx]
            weight = expert_weights[:, idx].unsqueeze(-1)

            for e_id in range(self.config.num_experts):
                mask = expert_idx == e_id
                if mask.any():
                    expert_input = flat_hidden[mask]
                    expert_output = self.experts[e_id](expert_input)
                    output[mask] += weight[mask] * expert_output

        # Add shared expert contributions (always active)
        for shared in self.shared_experts:
            output = output + shared(flat_hidden) / self.config.num_shared_experts

        return output.view(bsz, seq_len, hidden_dim), aux_loss


class ShutenDojiLayer(nn.Module):
    """Single transformer layer — attention + (dense FFN or MoE FFN)."""

    def __init__(self, config: ShutenDojiConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_dense = layer_idx < config.num_dense_layers

        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GQAttention(config, layer_idx)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if self.is_dense:
            self.ffn = DenseFFN(config)
        else:
            self.ffn = MoEFFN(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple], torch.Tensor]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, past_key_value = self.attention(
            hidden_states, position_ids, attention_mask, past_key_value
        )
        hidden_states = residual + hidden_states

        # FFN with residual
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)

        aux_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.is_dense:
            hidden_states = self.ffn(hidden_states)
        else:
            hidden_states, aux_loss = self.ffn(hidden_states, token_type_ids)

        hidden_states = residual + hidden_states
        return hidden_states, past_key_value, aux_loss


class ShutenDojiModel(nn.Module):
    """
    SHUTEN-DŌJI: Full model architecture.

    Custom MoE transformer for strategic intelligence production.
    Training via LLaMA Factory — this module is registered as a custom model.
    """

    def __init__(self, config: ShutenDojiConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embed = nn.Embedding(config.num_token_types, config.hidden_size)

        self.layers = nn.ModuleList([
            ShutenDojiLayer(config, i) for i in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Persistent memory module
        self.memory = PersistentMemory(config)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        if token_type_ids is not None:
            hidden_states = hidden_states + self.token_type_embed(token_type_ids)

        # Causal attention mask
        if attention_mask is None:
            attention_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
            ).unsqueeze(0).unsqueeze(0)

        # Memory integration (cross-attention to persistent slots)
        if use_memory:
            hidden_states = self.memory.read(hidden_states)

        # Transformer layers
        total_aux_loss = torch.tensor(0.0, device=device)
        new_past_key_values = []

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_past_kv, aux_loss = layer(
                hidden_states, position_ids, attention_mask, token_type_ids, past_kv
            )
            new_past_key_values.append(new_past_kv)
            total_aux_loss = total_aux_loss + aux_loss

        # Memory write (model decides what to store)
        if use_memory:
            self.memory.write(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add router auxiliary loss
            loss = loss + self.config.router_aux_loss_coef * total_aux_loss / self.config.num_layers

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": new_past_key_values,
            "aux_loss": total_aux_loss,
        }

    def num_parameters(self, only_trainable: bool = True) -> int:
        params = sum(p.numel() for p in self.parameters() if not only_trainable or p.requires_grad)
        return params
