import torch
from torch import nn
import torch.nn.functional as F


def divide(numerator, denominator):
    # 适配CPU场景：分母固定为1，直接返回分子
    assert denominator == 1, "CPU模式下无需张量并行，分母必须为1"
    return numerator


class LinearBase(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ):
        super().__init__()
        # 移除TP相关的tp_dim/tp_rank/tp_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
        # 基础权重加载逻辑：直接复制（移除TP分片逻辑）
        param.data.copy_(loaded_weight)


class ReplicatedLinear(LinearBase):
    """CPU版本：普通线性层（原ReplicatedLinear无TP逻辑，仅保留原生实现）"""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """CPU版本：移除TP列并行分片，退化为普通线性层"""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ):
        # 移除TP_SIZE切分，直接使用完整的output_size
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """CPU版本：合并列并行线性层，移除TP分片逻辑"""

    def __init__(
            self,
            input_size: int,
            output_sizes: list[int],
            bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 直接使用求和后的完整output_size
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # 移除TP分片偏移/切分逻辑，直接复制对应分片的完整权重
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        param_data = param.data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """CPU版本：QKV并行线性层，移除TP头数切分"""

    def __init__(
            self,
            hidden_size: int,
            head_size: int,
            total_num_heads: int,
            total_num_kv_heads: int | None = None,
            bias: bool = False,
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 移除TP_SIZE切分，直接使用完整的head数
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        # 移除TP分片逻辑，直接加载对应Q/K/V的完整权重
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        param_data = param_data.narrow(0, shard_offset, shard_size)
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """CPU版本：移除TP行并行，退化为普通线性层"""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
    ):
        # 移除TP_SIZE切分，直接使用完整的input_size
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 移除TP的all_reduce和bias的rank判断
        return F.linear(x, self.weight, self.bias)
