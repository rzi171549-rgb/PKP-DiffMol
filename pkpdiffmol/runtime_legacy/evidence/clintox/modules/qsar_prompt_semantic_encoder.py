# -*- coding: utf-8 -*-
"""
真正的 descriptor prompt 语义编码器。

实现边界：
1. 只接收 descriptor prompt 文本，不接收分子主干 hidden states。
2. 使用真正的预训练 BERT 类文本编码器，不允许静默降级到轻量方案。
3. 默认按本地离线路径加载 SciBERT。
4. prompt 使用 12 维原始 descriptor 的四舍五入值，不使用 z-score 后的值。
"""

import os
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    from transformers import BertModel, BertTokenizer
    _TRANSFORMERS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    BertModel = None
    BertTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc


class QSARPromptSemanticEncoder(nn.Module):
    """
    将 descriptor 数值组织为 prompt 文本，再用预训练 BERT 编码。
    """

    def __init__(
        self,
        descriptor_names: Sequence[str],
        text_encoder_name_or_path: str,
        text_pooling: str = "pooler",
        text_proj_dim: int = 64,
        text_dropout: float = 0.1,
        prompt_round_digits: int = 1,
        freeze_text_encoder: bool = True,
        prompt_max_length: int = 128,
    ):
        super().__init__()

        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise ImportError(
                "当前环境无法导入 transformers，无法构建真正的 BERT 文本编码器。"
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.descriptor_names = list(descriptor_names)
        if not self.descriptor_names:
            raise ValueError("descriptor_names 不能为空。")

        self.text_encoder_name_or_path = os.path.abspath(
            os.path.expanduser(str(text_encoder_name_or_path))
        )
        self.text_pooling = str(text_pooling).lower()
        self.prompt_round_digits = int(prompt_round_digits)
        self.freeze_text_encoder = bool(freeze_text_encoder)
        self.prompt_max_length = int(prompt_max_length)

        if self.text_pooling not in {"pooler", "cls", "mean"}:
            raise ValueError(
                f"text_pooling 只支持 pooler/cls/mean，当前收到: {text_pooling}"
            )
        if self.prompt_max_length <= 0:
            raise ValueError("prompt_max_length 必须大于 0。")

        self.tokenizer, self.text_encoder = self._load_text_encoder(
            self.text_encoder_name_or_path
        )

        hidden_size = int(self.text_encoder.config.hidden_size)
        self.text_proj = nn.Linear(hidden_size, int(text_proj_dim))
        self.text_dropout = nn.Dropout(p=float(text_dropout))
        self.output_dim = int(text_proj_dim)

        if self.freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
            self.text_encoder.eval()

    def _load_text_encoder(self, text_encoder_name_or_path: str):
        """
        按本地路径显式加载 BertTokenizer 和 BertModel。
        """
        if not text_encoder_name_or_path:
            raise ValueError("text_encoder_name_or_path 不能为空。")
        if not os.path.isdir(text_encoder_name_or_path):
            raise FileNotFoundError(
                f"text_encoder_name_or_path 不存在或不是目录: {text_encoder_name_or_path}"
            )

        try:
            tokenizer = BertTokenizer.from_pretrained(
                text_encoder_name_or_path,
                local_files_only=True,
            )
            text_encoder = BertModel.from_pretrained(
                text_encoder_name_or_path,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "本地 BERT 文本编码器加载失败，且当前实现不允许回退到 lightweight 方案。"
                f" 路径: {text_encoder_name_or_path}"
            ) from exc

        return tokenizer, text_encoder

    def train(self, mode: bool = True):
        """
        冻结文本编码器时，始终保持其处于 eval，避免内部 dropout 漂移。
        """
        super().train(mode)
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        return self

    def named_text_encoder_parameters(self):
        """
        仅暴露预训练 BERT 编码器参数，供训练脚本单独设置学习率。
        """
        return self.text_encoder.named_parameters()

    def _format_descriptor_value(self, value: float) -> str:
        """
        将原始 descriptor 值四舍五入为稳定文本。
        """
        rounded_value = round(float(value), self.prompt_round_digits)
        if self.prompt_round_digits <= 0:
            return str(int(round(rounded_value)))
        return f"{rounded_value:.{self.prompt_round_digits}f}"

    def build_prompt_text(
        self,
        descriptor_values: Sequence[float],
        is_valid: bool = True,
    ) -> str:
        """
        用真实 descriptor 名称组织 prompt。
        """
        descriptor_parts: List[str] = []
        for name, value in zip(self.descriptor_names, descriptor_values):
            descriptor_parts.append(
                f"{name} is {self._format_descriptor_value(value)}"
            )

        if is_valid:
            prefix = "Molecular descriptor profile:"
        else:
            prefix = "Invalid molecule descriptor profile:"

        return prefix + " " + "; ".join(descriptor_parts) + "."

    def build_prompt_texts(
        self,
        raw_descriptors: Union[torch.Tensor, Sequence[Sequence[float]]],
        valid_flags: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    ) -> List[str]:
        """
        批量构建 prompt 文本。
        """
        if isinstance(raw_descriptors, torch.Tensor):
            descriptor_rows = raw_descriptors.detach().cpu().tolist()
        else:
            descriptor_rows = [list(row) for row in raw_descriptors]

        if valid_flags is None:
            valid_list = [True] * len(descriptor_rows)
        elif isinstance(valid_flags, torch.Tensor):
            valid_list = [bool(x) for x in valid_flags.detach().cpu().view(-1).tolist()]
        else:
            valid_list = [bool(x) for x in valid_flags]

        prompt_texts: List[str] = []
        for descriptor_values, is_valid in zip(descriptor_rows, valid_list):
            prompt_texts.append(
                self.build_prompt_text(
                    descriptor_values=descriptor_values,
                    is_valid=is_valid,
                )
            )
        return prompt_texts

    def tokenizer_prompt(
        self,
        prompt_texts: Union[str, Sequence[str]],
    ):
        """
        保留与 MolPrompt 接近的 tokenizer_prompt 接口形态。
        """
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]

        return self.tokenizer(
            text=list(prompt_texts),
            truncation=True,
            padding=True,
            add_special_tokens=True,
            max_length=self.prompt_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

    def _pool_text_features(
        self,
        model_outputs,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        按配置做 pooler/cls/mean pooling。
        """
        if self.text_pooling == "pooler":
            pooler_output = getattr(model_outputs, "pooler_output", None)
            if pooler_output is not None:
                return pooler_output
            return model_outputs.last_hidden_state[:, 0, :]

        if self.text_pooling == "cls":
            return model_outputs.last_hidden_state[:, 0, :]

        token_embeddings = model_outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        raw_descriptors: torch.Tensor,
        valid_flags: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        输入原始 descriptor 数值，输出文本语义表示。
        """
        prompt_texts = self.build_prompt_texts(
            raw_descriptors=raw_descriptors,
            valid_flags=valid_flags,
        )
        tokenized = self.tokenizer_prompt(prompt_texts)

        model_inputs = {
            "input_ids": tokenized["input_ids"].to(raw_descriptors.device),
            "attention_mask": tokenized["attention_mask"].to(raw_descriptors.device),
        }
        if "token_type_ids" in tokenized:
            model_inputs["token_type_ids"] = tokenized["token_type_ids"].to(
                raw_descriptors.device
            )

        if self.freeze_text_encoder:
            with torch.no_grad():
                model_outputs = self.text_encoder(**model_inputs, return_dict=True)
        else:
            model_outputs = self.text_encoder(**model_inputs, return_dict=True)

        pooled_text = self._pool_text_features(
            model_outputs=model_outputs,
            attention_mask=model_inputs["attention_mask"],
        )
        text_repr = self.text_proj(pooled_text)
        text_repr = self.text_dropout(text_repr)
        return text_repr
