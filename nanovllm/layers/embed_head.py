import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = 0
        self.tp_size = 1
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings
        self.vocab_start_idx = 0
        self.vocab_end_idx = num_embeddings
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device="cpu"))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight.cpu())

    def forward(self, x: torch.Tensor):
        x = x.cpu()
        return F.embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        x = x.cpu()
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return F.linear(x, self.weight)
