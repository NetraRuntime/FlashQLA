# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from .chunk import chunk_gated_delta_rule
from .fused_recurrent import recurrent_gated_delta_rule


__all__ = ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"]
