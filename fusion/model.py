from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def _encoder_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module] | None:
    candidates: list[Any] = [
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(model, "encoder", None), "layers", None),
        getattr(getattr(getattr(model, "bert", None), "encoder", None), "layer", None),
        getattr(
            getattr(getattr(model, "wav2vec2", None), "encoder", None),
            "layers",
            None,
        ),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    return None


def configure_encoder_trainability(model: nn.Module, unfreeze_last_layers: int) -> None:
    """Freeze an encoder and optionally unfreeze the last N transformer layers.

    Use -1 to fine-tune the full encoder, 0 to freeze it completely.
    """

    if unfreeze_last_layers < 0:
        _set_requires_grad(model, True)
        return

    _set_requires_grad(model, False)
    if unfreeze_last_layers == 0:
        return

    layers = _encoder_layers(model)
    if layers is None:
        return

    for layer in list(layers)[-unfreeze_last_layers:]:
        _set_requires_grad(layer, True)


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)

    mask = mask.to(dtype=values.dtype, device=values.device).unsqueeze(-1)
    summed = (values * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def audio_feature_mask(
    audio_encoder: nn.Module,
    audio_hidden: torch.Tensor,
    raw_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, feature_length, _ = audio_hidden.shape
    device = audio_hidden.device

    if raw_attention_mask is None:
        return torch.ones(batch_size, feature_length, dtype=torch.bool, device=device)

    raw_attention_mask = raw_attention_mask.to(device)
    mask_method = getattr(audio_encoder, "_get_feature_vector_attention_mask", None)
    if mask_method is not None:
        try:
            feature_mask = mask_method(
                feature_length,
                raw_attention_mask,
                add_adapter=False,
            )
        except TypeError:
            feature_mask = mask_method(feature_length, raw_attention_mask)
        return feature_mask.to(device=device, dtype=torch.bool)

    raw_lengths = raw_attention_mask.sum(dim=-1)
    max_raw_length = raw_attention_mask.shape[-1]
    feature_lengths = torch.ceil(raw_lengths.float() * feature_length / max_raw_length)
    positions = torch.arange(feature_length, device=device).unsqueeze(0)
    return positions < feature_lengths.long().unsqueeze(1)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.text_to_audio = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_to_text = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.text_norm_attn = nn.LayerNorm(hidden_size)
        self.audio_norm_attn = nn.LayerNorm(hidden_size)
        self.text_norm_ffn = nn.LayerNorm(hidden_size)
        self.audio_norm_ffn = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        inner_size = hidden_size * ffn_multiplier
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
        )
        self.audio_ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(
        self,
        text_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_context, _ = self.text_to_audio(
            query=text_tokens,
            key=audio_tokens,
            value=audio_tokens,
            key_padding_mask=~audio_mask,
            need_weights=False,
        )
        audio_context, _ = self.audio_to_text(
            query=audio_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=~text_mask,
            need_weights=False,
        )

        text_tokens = self.text_norm_attn(text_tokens + self.dropout(text_context))
        audio_tokens = self.audio_norm_attn(audio_tokens + self.dropout(audio_context))
        text_tokens = self.text_norm_ffn(
            text_tokens + self.dropout(self.text_ffn(text_tokens))
        )
        audio_tokens = self.audio_norm_ffn(
            audio_tokens + self.dropout(self.audio_ffn(audio_tokens))
        )
        return text_tokens, audio_tokens


class BertWav2VecCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        text_model_name: str = "bert-base-multilingual-cased",
        audio_model_name: str = "facebook/wav2vec2-base",
        num_labels: int = 2,
        fusion_dim: int = 256,
        cross_attention_heads: int = 4,
        cross_attention_layers: int = 2,
        dropout: float = 0.2,
        unfreeze_text_layers: int = 0,
        unfreeze_audio_layers: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name,
            cache_dir=cache_dir,
        )
        self.audio_encoder = AutoModel.from_pretrained(
            audio_model_name,
            cache_dir=cache_dir,
        )

        configure_encoder_trainability(self.text_encoder, unfreeze_text_layers)
        configure_encoder_trainability(self.audio_encoder, unfreeze_audio_layers)

        text_hidden = self.text_encoder.config.hidden_size
        audio_hidden = self.audio_encoder.config.hidden_size
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_attention = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_size=fusion_dim,
                    num_heads=cross_attention_heads,
                    dropout=dropout,
                )
                for _ in range(cross_attention_layers)
            ]
        )

        pair_dim = fusion_dim * 4
        self.fusion_head = nn.Sequential(
            nn.Linear(pair_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_values: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        audio_outputs = self.audio_encoder(
            input_values=audio_values,
            attention_mask=audio_attention_mask,
        )

        text_mask = attention_mask.to(dtype=torch.bool)
        audio_mask = audio_feature_mask(
            self.audio_encoder,
            audio_outputs.last_hidden_state,
            audio_attention_mask,
        )

        text_tokens = self.text_projection(text_outputs.last_hidden_state)
        audio_tokens = self.audio_projection(audio_outputs.last_hidden_state)

        for layer in self.cross_attention:
            text_tokens, audio_tokens = layer(
                text_tokens=text_tokens,
                audio_tokens=audio_tokens,
                text_mask=text_mask,
                audio_mask=audio_mask,
            )

        text_pool = masked_mean(text_tokens, text_mask)
        audio_pool = masked_mean(audio_tokens, audio_mask)
        pair_features = torch.cat(
            [
                text_pool,
                audio_pool,
                text_pool * audio_pool,
                torch.abs(text_pool - audio_pool),
            ],
            dim=-1,
        )
        fused = self.fusion_head(pair_features)
        logits = self.classifier(fused)

        output = {"logits": logits, "features": fused}
        if labels is not None:
            output["loss"] = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
            )
        return output

