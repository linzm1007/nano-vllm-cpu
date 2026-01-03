import torch
import logging
from torch import nn
from nanovllm.utils.context import get_context

# 配置日志（输出关键调试信息）
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AttentionDebug")


class Attention(nn.Module):

    def __init__(
            self,
            num_heads,
            head_dim,
            scale,
            num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([], device="cpu")
        # 预计算KV展平后的维度
        self.kv_flat_dim = self.num_kv_heads * self.head_dim

        # 计算每个KV头需要重复的次数（用于GQA）
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # ====================== 1. 关键参数日志打印（调试核心） ======================
        # logger.debug(f"=== Attention Forward Debug Info ===")
        # logger.debug(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
        # logger.debug(f"num_heads: {self.num_heads}, num_kv_heads: {self.num_kv_heads}, head_dim: {self.head_dim}")
        # logger.debug(f"kv_flat_dim (num_kv_heads*head_dim): {self.kv_flat_dim}")
        # logger.debug(f"is_prefill: {context.is_prefill}")
        # logger.debug(f"k_cache shape: {k_cache.shape}, v_cache shape: {v_cache.shape}")

        # 校验KV维度合法性
        assert k.ndim == 3, f"K must be 3D (N, num_kv_heads, head_dim), got {k.ndim}D"
        assert v.ndim == 3, f"V must be 3D (N, num_kv_heads, head_dim), got {v.ndim}D"
        assert k.shape[-2:] == (self.num_kv_heads, self.head_dim), \
            f"K shape {k.shape} mismatch with kv_heads/head_dim: ({self.num_kv_heads}, {self.head_dim})"
        assert v.shape[-2:] == (self.num_kv_heads, self.head_dim), \
            f"V shape {v.shape} mismatch with kv_heads/head_dim: ({self.num_kv_heads}, {self.head_dim})"

        # 简化KV Cache存储（CPU下无需Triton内核）
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None and context.slot_mapping.numel() > 0:
            # logger.debug(f"slot_mapping raw shape: {context.slot_mapping.shape}, values: {context.slot_mapping[:10]}")

            # 校验slot_mapping范围
            slot_mapping = context.slot_mapping.clamp(min=0)
            max_valid_slot = k_cache.view(-1, self.kv_flat_dim).shape[0] - 1
            valid_slots = (slot_mapping >= 0) & (slot_mapping <= max_valid_slot)

            # logger.debug(f"valid_slots mask: {valid_slots.sum()} valid slots / {len(valid_slots)} total")
            # logger.debug(f"max valid slot index: {max_valid_slot}, slot_mapping max: {slot_mapping.max()}")

            if valid_slots.any():
                # 将K/V展平为 [N, kv_flat_dim]，匹配k_cache.view后的形状
                k_flat = k[valid_slots].reshape(-1, self.kv_flat_dim)
                v_flat = v[valid_slots].reshape(-1, self.kv_flat_dim)

                # logger.debug(f"k_flat shape: {k_flat.shape}, v_flat shape: {v_flat.shape}")
                # logger.debug(
                #     f"k_cache target shape: {k_cache.view(-1, self.kv_flat_dim)[slot_mapping[valid_slots]].shape}")

                # 赋值到KV Cache（形状匹配）
                k_cache.view(-1, self.kv_flat_dim)[slot_mapping[valid_slots]] = k_flat
                v_cache.view(-1, self.kv_flat_dim)[slot_mapping[valid_slots]] = v_flat

        if context.is_prefill:
            # Prefill阶段：使用原生SDPA
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            # 校验context参数合法性
            assert context.cu_seqlens_q is not None, "cu_seqlens_q must not be None in prefill"
            batch_size_q = context.cu_seqlens_q.shape[0] - 1
            # logger.debug(f"Prefill batch_size_q: {batch_size_q}")

            # 调整形状适配SDPA: (batch * seq_len, num_heads/num_kv_heads, head_dim) → (batch, num_heads/num_kv_heads, seq_len, head_dim)
            # 计算每个批次的序列长度
            q_seq_len = q.shape[0] // batch_size_q
            k_seq_len = k.shape[0] // batch_size_q

            q_reshaped = q.view(batch_size_q, q_seq_len, self.num_heads, self.head_dim).transpose(1,
                                                                                                  2)  # (B, H_q, S_q, D)
            k_reshaped = k.view(batch_size_q, k_seq_len, self.num_kv_heads, self.head_dim).transpose(1,
                                                                                                     2)  # (B, H_kv, S_k, D)
            v_reshaped = v.view(batch_size_q, k_seq_len, self.num_kv_heads, self.head_dim).transpose(1,
                                                                                                     2)  # (B, H_kv, S_k, D)

            # logger.debug(
            #     f"Prefill Q reshaped: {q_reshaped.shape}, K reshaped: {k_reshaped.shape}, V reshaped: {v_reshaped.shape}")

            # 重复 K 和 V 的头维度，使其与 Q 的头数匹配
            if self.num_queries_per_kv > 1:
                k_reshaped = k_reshaped.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1).flatten(1,
                                                                                                             2)  # (B, H_q, S_k, D)
                v_reshaped = v_reshaped.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1, -1).flatten(1,
                                                                                                             2)  # (B, H_q, S_k, D)

            # logger.debug(
            #     f"Prefill Q reshaped (after GQA): {q_reshaped.shape}, K reshaped (after GQA): {k_reshaped.shape}, V reshaped (after GQA): {v_reshaped.shape}")

            # CPU下的Scaled Dot-Product Attention
            o = torch.nn.functional.scaled_dot_product_attention(
                q_reshaped, k_reshaped, v_reshaped,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
                scale=self.scale
            )
            o = o.transpose(1, 2).reshape(-1, self.num_heads * self.head_dim)
        else:
            # Decode阶段：单token解码
            # logger.debug(f"Decode stage Q shape: {q.shape}")

            # 在 Decode 阶段，Q 的形状是 [batch_size, num_heads, head_dim]
            # 所以 batch_size 就是 q.shape[0]
            batch_size = q.shape[0]
            # 重塑 Q 为 [batch_size, num_heads, 1, head_dim] (1是因为是单token)
            q_reshaped = q.unsqueeze(2)  # (batch_size, num_heads, 1, head_dim)

            # 使用 block_tables 获取当前批次token的完整KV缓存
            # block_tables: [batch_size, max_blocks_per_sequence]
            # context.block_tables 可能是None或实际的block索引表
            if context.block_tables is not None:
                block_tables = context.block_tables  # Shape: [batch_size, max_blocks]
                # 获取block大小 (从k_cache的形状推断)
                num_blocks, block_size_per_block, num_kv_heads_cache, head_dim_cache = k_cache.shape

                # 检查形状是否匹配
                assert num_kv_heads_cache == self.num_kv_heads, f"Cache num_kv_heads {num_kv_heads_cache} != self.num_kv_heads {self.num_kv_heads}"
                assert head_dim_cache == self.head_dim, f"Cache head_dim {head_dim_cache} != self.head_dim {self.head_dim}"

                # 获取当前批次所有token的KV缓存
                # block_ids: [batch_size, max_blocks] -> [batch_size, max_blocks, 1, 1, 1]
                # k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
                # 使用高级索引获取对应的blocks
                # block_tables_expanded = block_tables.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, block_size_per_block, num_kv_heads_cache, head_dim_cache)
                # k_cache_selected = k_cache[block_tables_expanded.flatten()].view(batch_tables.shape[0], -1, num_kv_heads_cache, head_dim_cache)

                # 更高效的实现，逐批次获取
                k_selected_list = []
                v_selected_list = []
                for i in range(batch_size):
                    block_indices = block_tables[i]  # [max_blocks]
                    valid_blocks = block_indices[block_indices != -1]  # 假设-1是padding
                    if valid_blocks.numel() == 0:  # 如果没有有效块，跳过或使用空张量
                        # 创建一个虚拟的KV，长度为0
                        k_selected_list.append(torch.empty(0, num_kv_heads_cache, head_dim_cache, dtype=k_cache.dtype,
                                                           device=k_cache.device))
                        v_selected_list.append(torch.empty(0, num_kv_heads_cache, head_dim_cache, dtype=v_cache.dtype,
                                                           device=v_cache.device))
                        continue

                    # 选择对应的blocks
                    k_blocks = k_cache[valid_blocks]  # [num_valid_blocks, block_size_per_block, num_kv_heads, head_dim]
                    v_blocks = v_cache[valid_blocks]  # [num_valid_blocks, block_size_per_block, num_kv_heads, head_dim]

                    # 将blocks展平成序列
                    k_selected_list.append(k_blocks.reshape(-1, num_kv_heads_cache, head_dim_cache))
                    v_selected_list.append(v_blocks.reshape(-1, num_kv_heads_cache, head_dim_cache))

                k_cache_selected = torch.stack(k_selected_list,
                                               dim=0)  # [batch_size, total_cached_tokens, num_kv_heads, head_dim]
                v_cache_selected = torch.stack(v_selected_list,
                                               dim=0)  # [batch_size, total_cached_tokens, num_kv_heads, head_dim]

            else:
                # 如果没有block_tables，无法确定有效缓存，抛出错误
                raise ValueError(
                    "context.block_tables is None, cannot retrieve KV cache for decode stage without block tables.")

            # logger.debug(
            #     f"Decode K cache selected shape: {k_cache_selected.shape}, V cache selected shape: {v_cache_selected.shape}")

            # Transpose KV caches to match SDPA input: (B, T, H_kv, D) -> (B, H_kv, T, D)
            k_cache_selected = k_cache_selected.transpose(1,
                                                          2)  # (batch_size, num_kv_heads, total_cached_tokens, head_dim)
            v_cache_selected = v_cache_selected.transpose(1,
                                                          2)  # (batch_size, num_kv_heads, total_cached_tokens, head_dim)

            # 重复 K 和 V 的头维度，使其与 Q 的头数匹配
            if self.num_queries_per_kv > 1:
                k_cache_selected = k_cache_selected.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1,
                                                                        -1).flatten(1,
                                                                                    2)  # (batch_size, num_heads, total_cached_tokens, head_dim)
                v_cache_selected = v_cache_selected.unsqueeze(2).expand(-1, -1, self.num_queries_per_kv, -1,
                                                                        -1).flatten(1,
                                                                                    2)  # (batch_size, num_heads, total_cached_tokens, head_dim)

            # logger.debug(
            #     f"Decode Q reshaped: {q_reshaped.shape}, K reshaped (from cache): {k_cache_selected.shape}, V reshaped (from cache): {v_cache_selected.shape}")

            o = torch.nn.functional.scaled_dot_product_attention(
                q_reshaped, k_cache_selected, v_cache_selected,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,  # Decode阶段通常不需要因果掩码，因为新token只看缓存
                scale=self.scale
            )
            o = o.squeeze(2).reshape(-1,
                                     self.num_heads * self.head_dim)  # (batch_size, num_heads, head_dim) -> (batch_size * num_heads, head_dim)

        # logger.debug(f"Attention output shape: {o.shape}")
        return o
