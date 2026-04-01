import math
import torch
from typing import Optional, Tuple

from .dot_production_attention import get_multi_stage_dot_production_attention
from model.profiling import profile_section


# Allocate a fixed-size block of GPU memory specifically for storing the KV-Cache of the local_window.
class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units  # n_block
        self.unit_size = unit_size  # block_size * hidden_dim * 2
        self.dtype = dtype
        self.data = torch.empty(
            (num_units, unit_size),
            device = "cuda",
            dtype=dtype
        )
        self.idle_set = set(list(range(num_units)))

    def alloc(self):
        assert len(self.idle_set) > 0
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)


# The KV-Cache management unit supports data transfer between the CPU and GPU.
class MemoryUnit:
    # Initialize the KV-Cache management unit and store it on the CPU.
    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        if pin_memory:
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    # Load data from the CPU to the GPU and copy it to 'target' when necessary.
    # target: 2x (n_head, n_token, head_dim), on GPU
    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None

            return False, target_event

        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)
        target_event = None
        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)

        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    # Get the KV-Cache stored on GPU
    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    # Clear the KV-Cache stored on GPU
    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def calculate_cpu_memory(self):
        return len(self.cpu_data) * self.cpu_data[0].numel() * self.cpu_data[0].element_size()


# A dynamically growing vector cache on the GPU, used to store representative vectors of video frames.
class VectorTensor:
    # Initialize an empty cache of size (16, hidden_dim) on the GPU.
    def __init__(
        self, 
        hidden_size,
        element_dtype,
        device
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    # Double the size of the cache.
    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size,...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    # Append a frame vector to the cache, and expand the cache if it exceeds the current cache size.
    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size, f'{tensor.size(1)}, {self.hidden_size}'
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)

        self.length += append_l

    # Get the cached frame vectors
    def get_data(self):
        return self.data[:self.length, ...]

    def get_cosine_similarity(self, tensor: torch.Tensor):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        key = self.data[:self.length].float()  # (T, D), convert to fp32 to prevent numerical overflow
        query = tensor[None, :].float()  # (1, D)

        logits = torch.matmul(query, key.T)[0]  # (T,)

        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def get_similarity_token_q(self, query_tokens: torch.Tensor, agg: str = "topk", topk_count: Optional[int] = None):
        assert query_tokens.dim() == 2 and query_tokens.size(1) == self.hidden_size
        if self.length == 0:
            return torch.empty((0,), device=self.data.device, dtype=torch.float32)

        key = self.data[:self.length].float()  # (n_frame, hidden_dim)
        query = query_tokens.float()  # (q_len, hidden_dim)
        sim = torch.matmul(query, key.T).transpose(0, 1)  # (n_frame, q_len)
        if agg == "mean":
            return sim.mean(dim=-1)
        if agg == "topk":
            if topk_count is None:
                raise ValueError("topk_count must be provided when agg='topk'.")
            keep = max(1, min(topk_count, sim.size(-1)))
            return sim.topk(keep, dim=-1).values.mean(dim=-1)
        raise ValueError(f"Unsupported agg: {agg}. Choose from {{'mean', 'topk'}}.")

    def __len__(self):
        return self.length


class HeadVectorTensor:
    def __init__(
        self,
        num_heads_kv,
        dim_head,
        element_dtype,
        device
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, num_heads_kv, dim_head),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head

    def append_cache(self):
        new_cache_size = self.cache_size * 2
        new_data = torch.empty(
            (new_cache_size, self.num_heads_kv, self.dim_head),
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size, ...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.dim() == 3
        assert tensor.size(1) == self.num_heads_kv
        assert tensor.size(2) == self.dim_head
        assert tensor.is_contiguous()

        append_l = tensor.size(0)
        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length + append_l, ...].copy_(tensor)
        self.length += append_l

    def get_cosine_similarity(self, query_heads: torch.Tensor):
        assert query_heads.dim() == 2 and query_heads.size(0) == self.num_heads_kv and query_heads.size(1) == self.dim_head
        key = self.data[:self.length].float()  # (n_block, n_head_kv, dim_head)
        query = query_heads.float()  # (n_head_kv, dim_head)
        logits = (key * query[None, :, :]).sum(dim=-1).mean(dim=-1)  # (n_block,)
        return logits

    def get_head_similarity_mean_q(self, query_heads: torch.Tensor):
        assert query_heads.dim() == 2 and query_heads.size(0) == self.num_heads_kv and query_heads.size(1) == self.dim_head
        key = self.data[:self.length].float()  # (n_block, n_head_kv, dim_head)
        query = query_heads.float()  # (n_head_kv, dim_head)
        logits = (key * query[None, :, :]).sum(dim=-1)  # (n_block, n_head_kv)
        return logits.transpose(0, 1).contiguous()  # (n_head_kv, n_block)

    def get_similarity_token_q(self, query_tokens: torch.Tensor, agg: str = "topk", topk_count: Optional[int] = None):
        assert query_tokens.dim() == 3
        assert query_tokens.size(0) == self.num_heads_kv
        assert query_tokens.size(2) == self.dim_head
        if self.length == 0:
            return torch.empty((0,), device=self.data.device, dtype=torch.float32)

        key = self.data[:self.length].float()  # (n_block, n_head_kv, dim_head)
        query = query_tokens.float()  # (n_head_kv, q_len, dim_head)
        sim = torch.einsum("bhd,hqd->bhq", key, query)  # (n_block, n_head_kv, q_len)
        sim = sim.mean(dim=1)  # (n_block, q_len)
        if agg == "mean":
            return sim.mean(dim=-1)
        if agg == "topk":
            if topk_count is None:
                raise ValueError("topk_count must be provided when agg='topk'.")
            keep = max(1, min(topk_count, sim.size(-1)))
            return sim.topk(keep, dim=-1).values.mean(dim=-1)
        raise ValueError(f"Unsupported agg: {agg}. Choose from {{'mean', 'topk'}}.")

    def get_head_similarity_token_q(self, query_tokens: torch.Tensor, agg: str = "topk", topk_count: Optional[int] = None,
                                    k_agg: str = "max", k_topk_count: Optional[int] = None):
        assert query_tokens.dim() == 3
        assert query_tokens.size(0) == self.num_heads_kv
        assert query_tokens.size(2) == self.dim_head
        if self.length == 0:
            return torch.empty((self.num_heads_kv, 0), device=self.data.device, dtype=torch.float32)

        if k_agg not in {"max", "topk"}:
            raise ValueError(f"Unsupported k_agg: {k_agg}. Choose from {{'max', 'topk'}}.")

        key = self.data[:self.length].float()  # (n_block, n_head_kv, dim_head)
        query = query_tokens.float()  # (n_head_kv, q_len, dim_head)
        sim = torch.einsum("bhd,hqd->bhq", key, query).transpose(1, 2)  # (n_block, q_len, n_head_kv)
        sim = sim.transpose(0, 2).contiguous()  # (n_head_kv, q_len, n_block)
        if agg == "mean":
            return sim.mean(dim=1)
        if agg == "topk":
            if topk_count is None:
                raise ValueError("topk_count must be provided when agg='topk'.")
            keep = max(1, min(topk_count, sim.size(1)))
            return sim.topk(keep, dim=1).values.mean(dim=1)
        raise ValueError(f"Unsupported agg: {agg}. Choose from {{'mean', 'topk'}}.")

    def get_per_head_token_scores(self, query_tokens: torch.Tensor):
        assert query_tokens.dim() == 3
        assert query_tokens.size(0) == self.num_heads_kv
        assert query_tokens.size(2) == self.dim_head
        if self.length == 0:
            return torch.empty((self.num_heads_kv, 0, query_tokens.size(1)), device=self.data.device, dtype=torch.float32)

        key = self.data[:self.length].float()  # (n_block, n_head_kv, dim_head)
        query = query_tokens.float()  # (n_head_kv, q_len, dim_head)
        sim = torch.einsum("bhd,hqd->bhq", key, query)  # (n_block, n_head_kv, q_len)
        return sim.permute(1, 0, 2).contiguous()  # (n_head_kv, n_block, q_len)

    def __len__(self):
        return self.length


class BlockTokenTensor:
    def __init__(
        self,
        block_token_count,
        hidden_size,
        element_dtype,
        device
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, block_token_count, hidden_size),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.block_token_count = block_token_count
        self.hidden_size = hidden_size

    def append_cache(self):
        new_cache_size = self.cache_size * 2
        new_data = torch.empty(
            (new_cache_size, self.block_token_count, self.hidden_size),
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size, ...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.dim() == 3
        assert tensor.size(1) == self.block_token_count
        assert tensor.size(2) == self.hidden_size
        assert tensor.is_contiguous()

        append_l = tensor.size(0)
        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length:self.length + append_l, ...].copy_(tensor)
        self.length += append_l

    def get_similarity_mean_q(self, query: torch.Tensor, k_agg: str = "max", k_topk_count: Optional[int] = None):
        assert query.dim() == 1 and query.size(0) == self.hidden_size
        key = self.data[:self.length].float()  # (n_block, n_token, hidden_dim)
        sim = torch.einsum("bkd,d->bk", key, query.float())  # (n_block, n_token)
        if k_agg == "max":
            return sim.max(dim=-1).values
        if k_agg == "topk":
            if k_topk_count is None:
                raise ValueError("k_topk_count must be provided when k_agg='topk'.")
            keep = max(1, min(k_topk_count, sim.size(-1)))
            return sim.topk(keep, dim=-1).values.mean(dim=-1)
        raise ValueError(f"Unsupported k_agg: {k_agg}. Choose from {{'max', 'topk'}}.")

    def get_similarity_token_q(self, query_tokens: torch.Tensor, agg: str = "topk", topk_count: Optional[int] = None,
                               k_agg: str = "max", k_topk_count: Optional[int] = None):
        assert query_tokens.dim() == 2 and query_tokens.size(1) == self.hidden_size
        if self.length == 0:
            return torch.empty((0,), device=self.data.device, dtype=torch.float32)

        key = self.data[:self.length].float()  # (n_block, n_token, hidden_dim)
        query = query_tokens.float()  # (q_len, hidden_dim)
        sim = torch.einsum("qd,bkd->bqk", query, key)  # (n_block, q_len, n_token)
        if k_agg == "max":
            maxsim = sim.max(dim=-1).values  # (n_block, q_len)
        elif k_agg == "topk":
            if k_topk_count is None:
                raise ValueError("k_topk_count must be provided when k_agg='topk'.")
            keep_k = max(1, min(k_topk_count, sim.size(-1)))
            maxsim = sim.topk(keep_k, dim=-1).values.mean(dim=-1)  # (n_block, q_len)
        else:
            raise ValueError(f"Unsupported k_agg: {k_agg}. Choose from {{'max', 'topk'}}.")
        if agg == "mean":
            return maxsim.mean(dim=-1)
        if agg == "topk":
            if topk_count is None:
                raise ValueError("topk_count must be provided when agg='topk'.")
            keep = max(1, min(topk_count, maxsim.size(-1)))
            return maxsim.topk(keep, dim=-1).values.mean(dim=-1)
        raise ValueError(f"Unsupported agg: {agg}. Choose from {{'mean', 'topk'}}.")

    def __len__(self):
        return self.length


class HeadBlockTokenTensor:
    def __init__(
        self,
        num_heads_kv,
        block_token_count,
        dim_head,
        element_dtype,
        device
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, num_heads_kv, block_token_count, dim_head),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.num_heads_kv = num_heads_kv
        self.block_token_count = block_token_count
        self.dim_head = dim_head

    def append_cache(self):
        new_cache_size = self.cache_size * 2
        new_data = torch.empty(
            (new_cache_size, self.num_heads_kv, self.block_token_count, self.dim_head),
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size, ...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.dim() == 4
        assert tensor.size(1) == self.num_heads_kv
        assert tensor.size(2) == self.block_token_count
        assert tensor.size(3) == self.dim_head
        assert tensor.is_contiguous()

        append_l = tensor.size(0)
        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length:self.length + append_l, ...].copy_(tensor)
        self.length += append_l

    def get_head_similarity_mean_q(self, query_heads: torch.Tensor, k_agg: str = "max", k_topk_count: Optional[int] = None):
        assert query_heads.dim() == 2
        assert query_heads.size(0) == self.num_heads_kv
        assert query_heads.size(1) == self.dim_head
        if self.length == 0:
            return torch.empty((self.num_heads_kv, 0), device=self.data.device, dtype=torch.float32)

        key = self.data[:self.length].float()  # (n_block, n_head_kv, n_token, dim_head)
        query = query_heads.float()  # (n_head_kv, dim_head)
        sim = torch.einsum("bhkd,hd->bhk", key, query)  # (n_block, n_head_kv, n_token)
        if k_agg == "max":
            score = sim.max(dim=-1).values
        elif k_agg == "topk":
            if k_topk_count is None:
                raise ValueError("k_topk_count must be provided when k_agg='topk'.")
            keep = max(1, min(k_topk_count, sim.size(-1)))
            score = sim.topk(keep, dim=-1).values.mean(dim=-1)
        else:
            raise ValueError(f"Unsupported k_agg: {k_agg}. Choose from {{'max', 'topk'}}.")
        return score.transpose(0, 1).contiguous()  # (n_head_kv, n_block)

    def get_head_similarity_token_q(self, query_tokens: torch.Tensor, agg: str = "topk", topk_count: Optional[int] = None,
                                    k_agg: str = "max", k_topk_count: Optional[int] = None):
        assert query_tokens.dim() == 3
        assert query_tokens.size(0) == self.num_heads_kv
        assert query_tokens.size(2) == self.dim_head
        if self.length == 0:
            return torch.empty((self.num_heads_kv, 0), device=self.data.device, dtype=torch.float32)

        key = self.data[:self.length].float()  # (n_block, n_head_kv, n_token, dim_head)
        query = query_tokens.float()  # (n_head_kv, q_len, dim_head)
        sim = torch.einsum("bhqd,bhkd->bhqk", query[None, ...].expand(key.size(0), -1, -1, -1), key)  # (n_block, n_head_kv, q_len, n_token)
        if k_agg == "max":
            maxsim = sim.max(dim=-1).values.transpose(0, 1).contiguous()  # (n_head_kv, n_block, q_len)
        elif k_agg == "topk":
            if k_topk_count is None:
                raise ValueError("k_topk_count must be provided when k_agg='topk'.")
            keep_k = max(1, min(k_topk_count, sim.size(-1)))
            maxsim = sim.topk(keep_k, dim=-1).values.mean(dim=-1).transpose(0, 1).contiguous()  # (n_head_kv, n_block, q_len)
        else:
            raise ValueError(f"Unsupported k_agg: {k_agg}. Choose from {{'max', 'topk'}}.")
        if agg == "mean":
            return maxsim.mean(dim=-1)  # (n_head_kv, n_block)
        if agg == "topk":
            if topk_count is None:
                raise ValueError("topk_count must be provided when agg='topk'.")
            keep = max(1, min(topk_count, maxsim.size(-1)))
            return maxsim.topk(keep, dim=-1).values.mean(dim=-1)  # (n_head_kv, n_block)
        raise ValueError(f"Unsupported agg: {agg}. Choose from {{'mean', 'topk'}}.")

    def get_per_head_token_scores(self, query_tokens: torch.Tensor, k_agg: str = "max", k_topk_count: Optional[int] = None):
        assert query_tokens.dim() == 3
        assert query_tokens.size(0) == self.num_heads_kv
        assert query_tokens.size(2) == self.dim_head
        if self.length == 0:
            return torch.empty((self.num_heads_kv, 0, query_tokens.size(1)), device=self.data.device, dtype=torch.float32)

        key = self.data[:self.length].float()  # (n_block, n_head_kv, n_token, dim_head)
        query = query_tokens.float()  # (n_head_kv, q_len, dim_head)
        sim = torch.einsum("bhqd,bhkd->bhqk", query[None, ...].expand(key.size(0), -1, -1, -1), key)  # (n_block, n_head_kv, q_len, n_token)
        if k_agg == "max":
            scores = sim.max(dim=-1).values  # (n_block, n_head_kv, q_len)
        elif k_agg == "topk":
            if k_topk_count is None:
                raise ValueError("k_topk_count must be provided when k_agg='topk'.")
            keep_k = max(1, min(k_topk_count, sim.size(-1)))
            scores = sim.topk(keep_k, dim=-1).values.mean(dim=-1)  # (n_block, n_head_kv, q_len)
        else:
            raise ValueError(f"Unsupported k_agg: {k_agg}. Choose from {{'max', 'topk'}}.")
        return scores.permute(1, 0, 2).contiguous()  # (n_head_kv, n_block, q_len)

    def __len__(self):
        return self.length


GLOBAL_STREAM = None


class ContextManager:
    def __init__(self, 
                 position_embedding,
                 n_init, n_local, 
                 block_size, max_cached_block, topk, chunk_size, exc_block_size, 
                 kv_repr: str = "mean",
                 q_repr: str = "mean",
                 q_token_agg: str = "topk",
                 q_topk_ratio: float = 0.3,
                 k_token_agg: str = "max",
                 k_topk_ratio: float = 0.3,
                 head_specific_retrieval: bool = False,
                 retrieval_fusion: str = "none",
                 fusion_mean_topk: Optional[int] = None,
                 fusion_token_topk: Optional[int] = None,
                 fattn: bool = False,
                 async_global_stream: bool = False,
                 pin_memory: bool = False,
    ):

        self.length = 0  # number of tokens in the KV-Cache
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local # no global token in input
        self.topk = topk
        self.chunk_size = chunk_size
        self.kv_repr = kv_repr
        self.q_repr = q_repr
        self.q_token_agg = q_token_agg
        self.q_topk_ratio = q_topk_ratio
        self.k_token_agg = k_token_agg
        self.k_topk_ratio = k_topk_ratio
        self.head_specific_retrieval = head_specific_retrieval
        self.retrieval_fusion = retrieval_fusion
        if fusion_mean_topk is None and fusion_token_topk is None:
            fusion_mean_topk = topk // 2
            fusion_token_topk = topk - fusion_mean_topk
        elif fusion_mean_topk is None:
            fusion_mean_topk = topk - fusion_token_topk
        elif fusion_token_topk is None:
            fusion_token_topk = topk - fusion_mean_topk
        self.fusion_mean_topk = fusion_mean_topk
        self.fusion_token_topk = fusion_token_topk
        if self.kv_repr not in {"mean", "token"}:
            raise ValueError(f"Unsupported kv_repr: {self.kv_repr}. Choose from {{'mean', 'token'}}.")
        if self.q_repr not in {"mean", "token"}:
            raise ValueError(f"Unsupported q_repr: {self.q_repr}. Choose from {{'mean', 'token'}}.")
        if self.q_token_agg not in {"mean", "topk"}:
            raise ValueError(f"Unsupported q_token_agg: {self.q_token_agg}. Choose from {{'mean', 'topk'}}.")
        if not (0 < self.q_topk_ratio <= 1):
            raise ValueError(f"Unsupported q_topk_ratio: {self.q_topk_ratio}. It must be in (0, 1].")
        if self.k_token_agg not in {"max", "topk"}:
            raise ValueError(f"Unsupported k_token_agg: {self.k_token_agg}. Choose from {{'max', 'topk'}}.")
        if not (0 < self.k_topk_ratio <= 1):
            raise ValueError(f"Unsupported k_topk_ratio: {self.k_topk_ratio}. It must be in (0, 1].")
        if self.retrieval_fusion not in {"none", "quota"}:
            raise ValueError(
                f"Unsupported retrieval_fusion: {self.retrieval_fusion}. Choose from {{'none', 'quota'}}."
            )
        if self.retrieval_fusion != "none":
            if self.fusion_mean_topk < 0 or self.fusion_token_topk < 0:
                raise ValueError("fusion_mean_topk and fusion_token_topk must be non-negative.")
            if self.fusion_mean_topk + self.fusion_token_topk != self.topk:
                raise ValueError(
                    f"fusion_mean_topk + fusion_token_topk must equal topk: "
                    f"{self.fusion_mean_topk} + {self.fusion_token_topk} != {self.topk}"
                )
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.reset_retrieval()

    def _required_cached_blocks_per_unit(self):
        # Head-specific retrieval can request up to topk distinct blocks per KV head.
        # Size the GPU cache for that worst case so loading selected blocks cannot
        # exhaust the pool before LRU eviction has a chance to run.
        if self._use_head_specific_retrieval():
            return max(self.max_cached_block, self.topk * self.num_heads_kv)
        return max(self.max_cached_block, self.topk)

    def _build_block_storage(self, kv_repr: str, element_dtype, device):
        if kv_repr == "mean":
            return [
                HeadVectorTensor(self.num_heads_kv, self.dim_head, element_dtype, device)
                for _ in range(self.num_units)
            ]
        return [
            HeadBlockTokenTensor(
                self.num_heads_kv,
                self.block_size,
                self.dim_head,
                element_dtype,
                device,
            )
            for _ in range(self.num_units)
        ]

    def _append_block_representations(self, global_block_k: torch.Tensor):
        if self.retrieval_fusion != "none":
            mean_block_k = global_block_k.mean(dim=-2, keepdim=False).contiguous()
            token_block_k = global_block_k.contiguous()
            for u in range(self.num_units):
                self.block_k_mean[u].append(mean_block_k[u:u + 1])
                self.block_k_token[u].append(token_block_k[u:u + 1])
            return

        if self.kv_repr == "mean":
            global_block_k = global_block_k.mean(dim=-2, keepdim=False).contiguous()
        else:
            global_block_k = global_block_k.contiguous()
        for u in range(self.num_units):
            self.block_k[u].append(global_block_k[u:u + 1])

    def _get_q_token_topk_for_ratio(self, q_len: int, ratio: float) -> int:
        return max(1, math.ceil(ratio * q_len))

    def _get_k_token_topk_for_ratio(self, k_len: int, ratio: float) -> int:
        return max(1, math.ceil(ratio * k_len))

    def _compute_logits_from_block_k(
        self,
        block_k,
        global_h_q,
        kv_repr: str,
        q_repr: str,
        q_token_agg: str,
        q_topk_ratio: float,
        k_token_agg: str,
        k_topk_ratio: float,
    ):
        q_len = global_h_q.size(2)
        query_group_reduction = "mean" if self._use_head_specific_retrieval() else "sum"
        if q_repr == "mean":
            global_h_q = global_h_q.mean(dim=2, keepdim=False)
            global_h_q = self._group_query_heads(global_h_q, reduction=query_group_reduction)
        else:
            global_h_q = self._group_query_heads(global_h_q, reduction=query_group_reduction)

        if q_repr == "mean":
            if kv_repr == "mean":
                if self._use_head_specific_retrieval():
                    logits = torch.stack([
                        block_k[u].get_head_similarity_mean_q(global_h_q[u])
                        for u in range(self.num_units)
                    ])  # (batch_size, num_heads_kv, block_num)
                else:
                    logits = torch.stack([
                        block_k[u].get_head_similarity_mean_q(global_h_q[u]).sum(dim=0)
                        for u in range(self.num_units)
                    ])  # (batch_size, block_num)
            else:
                logits = torch.stack([
                    block_k[u].get_head_similarity_mean_q(
                        global_h_q[u],
                        k_agg=k_token_agg,
                        k_topk_count=self._get_k_token_topk_for_ratio(self.block_size, k_topk_ratio)
                        if k_token_agg == "topk" else None,
                    )
                    for u in range(self.num_units)
                ])  # (batch_size, num_heads_kv, block_num)
                if not self._use_head_specific_retrieval():
                    logits = logits.sum(dim=1)  # (batch_size, block_num)
        else:
            if kv_repr == "mean":
                if self._use_head_specific_retrieval():
                    logits = torch.stack([
                        block_k[u].get_head_similarity_token_q(
                            global_h_q[u],
                            agg=q_token_agg,
                            topk_count=self._get_q_token_topk_for_ratio(q_len, q_topk_ratio) if q_token_agg == "topk" else None,
                            k_agg=k_token_agg,
                            k_topk_count=self._get_k_token_topk_for_ratio(self.block_size, k_topk_ratio)
                            if k_token_agg == "topk" else None,
                        )
                        for u in range(self.num_units)
                    ])  # (batch_size, num_heads_kv, block_num)
                else:
                    logits = torch.stack([
                        (
                            block_k[u].get_per_head_token_scores(global_h_q[u]).sum(dim=0).mean(dim=-1)
                            if q_token_agg == "mean"
                            else block_k[u].get_per_head_token_scores(global_h_q[u]).sum(dim=0).topk(
                                max(1, min(self._get_q_token_topk_for_ratio(q_len, q_topk_ratio), q_len)), dim=-1
                            ).values.mean(dim=-1)
                        )
                        for u in range(self.num_units)
                    ])  # (batch_size, block_num)
            else:
                if self._use_head_specific_retrieval():
                    logits = torch.stack([
                        block_k[u].get_head_similarity_token_q(
                            global_h_q[u],
                            agg=q_token_agg,
                            topk_count=self._get_q_token_topk_for_ratio(q_len, q_topk_ratio) if q_token_agg == "topk" else None,
                            k_agg=k_token_agg,
                            k_topk_count=self._get_k_token_topk_for_ratio(self.block_size, k_topk_ratio)
                            if k_token_agg == "topk" else None,
                        )
                        for u in range(self.num_units)
                    ])  # (batch_size, num_heads_kv, block_num)
                else:
                    logits = torch.stack([
                        (
                            block_k[u].get_per_head_token_scores(
                                global_h_q[u],
                                k_agg=k_token_agg,
                                k_topk_count=self._get_k_token_topk_for_ratio(self.block_size, k_topk_ratio)
                                if k_token_agg == "topk" else None,
                            ).sum(dim=0).mean(dim=-1)
                            if q_token_agg == "mean"
                            else block_k[u].get_per_head_token_scores(
                                global_h_q[u],
                                k_agg=k_token_agg,
                                k_topk_count=self._get_k_token_topk_for_ratio(self.block_size, k_topk_ratio)
                                if k_token_agg == "topk" else None,
                            ).sum(dim=0).topk(
                                max(1, min(self._get_q_token_topk_for_ratio(q_len, q_topk_ratio), q_len)), dim=-1
                            ).values.mean(dim=-1)
                        )
                        for u in range(self.num_units)
                    ])  # (batch_size, block_num)
        return logits

    def _rank_blocks_from_logits(self, logits: torch.Tensor):
        rankings = []
        for u in range(self.num_units):
            unit_logits = logits[u]
            if unit_logits.dim() == 1:
                unit_logits = unit_logits.unsqueeze(0)
                unwrap_head = True
            else:
                unwrap_head = False

            unit_rankings = []
            for head_logits in unit_logits:
                remainder_size = head_logits.shape[-1] % self.chunk_size
                prefix_len = head_logits.shape[-1] - remainder_size
                if prefix_len > 0:
                    chunked_logits = head_logits[:prefix_len].reshape(-1, self.chunk_size).mean(dim=-1)
                else:
                    chunked_logits = head_logits.new_empty((0,))
                if remainder_size > 0:
                    remainder_logits = head_logits[-remainder_size:].mean(dim=-1, keepdim=True)
                    chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=0)
                ranked_chunks = chunked_logits.argsort(dim=0, descending=True).tolist()
                ranked_blocks = []
                for chunk_idx in ranked_chunks:
                    start = chunk_idx * self.chunk_size
                    ranked_blocks.extend([
                        idx for idx in range(start, start + self.chunk_size) if idx < head_logits.shape[-1]
                    ])
                unit_rankings.append(ranked_blocks)
            rankings.append(unit_rankings[0] if unwrap_head else unit_rankings)
        return rankings

    def _pick_next_unseen(self, ranked_indices, ptr, seen):
        while ptr < len(ranked_indices):
            block_idx = ranked_indices[ptr]
            ptr += 1
            if block_idx not in seen:
                return block_idx, ptr
        return None, ptr

    def _select_branch(self, mean_count, token_count):
        if mean_count >= self.fusion_mean_topk:
            return "token"
        if token_count >= self.fusion_token_topk:
            return "mean"
        mean_progress = mean_count / self.fusion_mean_topk if self.fusion_mean_topk > 0 else 1.0
        token_progress = token_count / self.fusion_token_topk if self.fusion_token_topk > 0 else 1.0
        return "mean" if mean_progress <= token_progress else "token"

    def _fuse_ranked_block_indices(self, mean_rankings, token_rankings):
        fused = []
        for u in range(self.num_units):
            if self._use_head_specific_retrieval():
                unit_fused = []
                for h in range(self.num_heads_kv):
                    selected = []
                    seen = set()
                    mean_ptr = 0
                    token_ptr = 0
                    mean_count = 0
                    token_count = 0
                    mean_head_rank = mean_rankings[u][h]
                    token_head_rank = token_rankings[u][h]

                    while len(selected) < self.topk and (mean_ptr < len(mean_head_rank) or token_ptr < len(token_head_rank)):
                        branch = self._select_branch(mean_count, token_count)
                        if branch == "mean":
                            block_idx, mean_ptr = self._pick_next_unseen(mean_head_rank, mean_ptr, seen)
                            if block_idx is None:
                                branch = "token"
                            else:
                                selected.append(block_idx)
                                seen.add(block_idx)
                                mean_count += 1
                                continue
                        block_idx, token_ptr = self._pick_next_unseen(token_head_rank, token_ptr, seen)
                        if block_idx is None:
                            block_idx, mean_ptr = self._pick_next_unseen(mean_head_rank, mean_ptr, seen)
                            if block_idx is None:
                                break
                            selected.append(block_idx)
                            seen.add(block_idx)
                            mean_count += 1
                        else:
                            selected.append(block_idx)
                            seen.add(block_idx)
                            token_count += 1
                    unit_fused.append(sorted(selected))
                fused.append(unit_fused)
                continue

            selected = []
            seen = set()
            mean_ptr = 0
            token_ptr = 0
            mean_count = 0
            token_count = 0

            while len(selected) < self.topk and (mean_ptr < len(mean_rankings[u]) or token_ptr < len(token_rankings[u])):
                branch = self._select_branch(mean_count, token_count)
                if branch == "mean":
                    block_idx, mean_ptr = self._pick_next_unseen(mean_rankings[u], mean_ptr, seen)
                    if block_idx is None:
                        branch = "token"
                    else:
                        selected.append(block_idx)
                        seen.add(block_idx)
                        mean_count += 1
                        continue
                block_idx, token_ptr = self._pick_next_unseen(token_rankings[u], token_ptr, seen)
                if block_idx is None:
                    block_idx, mean_ptr = self._pick_next_unseen(mean_rankings[u], mean_ptr, seen)
                    if block_idx is None:
                        break
                    selected.append(block_idx)
                    seen.add(block_idx)
                    mean_count += 1
                else:
                    selected.append(block_idx)
                    seen.add(block_idx)
                    token_count += 1

            fused.append(sorted(selected))
        return fused

    def _calc_hybrid_block_topk(self, global_h_q):
        if self.num_global_block <= self.topk:
            if self._use_head_specific_retrieval():
                return [[list(range(self.num_global_block)) for _ in range(self.num_heads_kv)] for _ in range(self.num_units)]
            return [list(range(self.num_global_block)) for _ in range(self.num_units)]

        mean_logits = self._compute_logits_from_block_k(
            self.block_k_mean,
            global_h_q,
            kv_repr="mean",
            q_repr="mean",
            q_token_agg="mean",
            q_topk_ratio=1.0,
            k_token_agg="max",
            k_topk_ratio=1.0,
        )
        token_logits = self._compute_logits_from_block_k(
            self.block_k_token,
            global_h_q,
            kv_repr="token",
            q_repr="token",
            q_token_agg=self.q_token_agg,
            q_topk_ratio=self.q_topk_ratio,
            k_token_agg=self.k_token_agg,
            k_topk_ratio=self.k_topk_ratio,
        )
        self.similarity = {"mean": mean_logits, "token": token_logits}
        mean_rankings = self._rank_blocks_from_logits(mean_logits)
        token_rankings = self._rank_blocks_from_logits(token_logits)
        return self._fuse_ranked_block_indices(mean_rankings, token_rankings)

    def _remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks = None):
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    # handle GQA, k: (batch_size, n_head_kv, length, dim_head) -> (batch_size, n_head, length, dim_head)
    def _from_group_kv(self, tensor):
        # tensor: (batch_size, n_head_kv, length, dim_head)
        assert tensor.dim() == 4 
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_units, self.unit_size_kv, 1, length, dim_head))  # (batch_size, n_head_kv, 1, length, dim_head)
        tensor = tensor.expand((self.num_units, self.unit_size_kv, num_group, length, dim_head)).reshape((self.num_units, self.num_heads, length, dim_head))  # (batch_size, n_head, length, dim_head)
        return tensor

    def _group_query_heads(self, tensor, reduction: str = "mean"):
        assert tensor.dim() in {3, 4}
        assert tensor.size(1) == self.num_heads
        if self.num_heads == self.num_heads_kv:
            return tensor
        num_group = self.num_heads // self.num_heads_kv
        if tensor.dim() == 3:
            batch_size, _, dim_head = tensor.shape
            tensor = tensor.view(batch_size, self.num_heads_kv, num_group, dim_head)
            if reduction == "mean":
                return tensor.mean(dim=2)
            if reduction == "sum":
                return tensor.sum(dim=2)
            raise ValueError(f"Unsupported reduction: {reduction}. Choose from {{'mean', 'sum'}}.")
        batch_size, _, q_len, dim_head = tensor.shape
        tensor = tensor.view(batch_size, self.num_heads_kv, num_group, q_len, dim_head)
        if reduction == "mean":
            return tensor.mean(dim=2)
        if reduction == "sum":
            return tensor.sum(dim=2)
        raise ValueError(f"Unsupported reduction: {reduction}. Choose from {{'mean', 'sum'}}.")
    
    def init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        """
        Only use the metadata of these parameters, such as shape, dtype, and device.
        """
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv

        self.global_blocks = [[] for _ in range(self.num_units)] # context memory's KV-Cache: [ batch_size x [memory_unit] ]
        self.cached_blocks = [{} for _ in range(self.num_units)] # relavency scores of blocks: batch_size x {block_id: block_score}
        self.num_global_block = 0

        # context memory's representative keys: batch_size x (n_blocks, hidden_dim)
        if self.retrieval_fusion != "none":
            self.block_k_mean = self._build_block_storage("mean", global_k.dtype, global_k.device)
            self.block_k_token = self._build_block_storage("token", global_k.dtype, global_k.device)
            self.block_k = self.block_k_token
        elif self.kv_repr == "mean":
            self.block_k = [HeadVectorTensor(
                self.num_heads_kv, dim_head, global_k.dtype, global_k.device
            ) for _ in range(self.num_units)]
        elif self.kv_repr == "token":
            self.block_k = [HeadBlockTokenTensor(
                self.num_heads_kv, self.block_size, dim_head, global_k.dtype, global_k.device
            ) for _ in range(self.num_units)]
        else:
            block_hidden = dim_head * self.unit_size
            self.block_k = [BlockTokenTensor(
                self.block_size, block_hidden, global_k.dtype, global_k.device
            ) for _ in range(self.num_units)]

        # local KV
        self.local_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)  # (batch_size, n_head_kv, 0, dim_head)
        self.local_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        # global KV that are not yet processed into blocks.
        # 2 x (batch_size, n_head_kv, length, dim_head)
        self.global_remainder = (
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )

        # init KV
        self.init_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        # buffering global KV during attention computations
        # (2, batch_size, n_head_kv, L, dim_head)
        # L = n_init + n_retrieve
        buffer_len = self.topk * self.block_size + self.n_init
        self.global_buffer = torch.zeros(
                (2, self.num_units, self.unit_size_kv, buffer_len , dim_head),
                dtype = global_k.dtype, device=global_k.device
            )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        self.max_cached_block = self._required_cached_blocks_per_unit()
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype
        )  # (max_cached_block * batch_size, block_size * D * 2)

        self.initialized = True

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.similarity = None
        self.retrieved_block_indices = None
        self.to_retrieve = False

    def set_retrieved_block_indices(self, retrieved_block_indices):
        # shared retrieval: batch_size x n_frames
        # head-specific retrieval: batch_size x n_head_kv x n_frames
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices

    def _use_head_specific_retrieval(self):
        return self.head_specific_retrieval

    def _unit_selected_blocks(self, unit_indices):
        if len(unit_indices) == 0:
            return []
        if isinstance(unit_indices[0], list):
            return sorted({b_idx for head_indices in unit_indices for b_idx in head_indices})
        return list(unit_indices)

    def _unit_slot_count(self, unit_indices):
        if len(unit_indices) == 0:
            return 0
        if isinstance(unit_indices[0], list):
            return max((len(head_indices) for head_indices in unit_indices), default=0)
        return len(unit_indices)

    def _get_slot_block_index(self, unit_indices, slot_idx, head_idx=None):
        if len(unit_indices) == 0:
            return None
        if isinstance(unit_indices[0], list):
            assert head_idx is not None
            head_indices = unit_indices[head_idx]
            if len(head_indices) == 0:
                return None
            if slot_idx < len(head_indices):
                return head_indices[slot_idx]
            return head_indices[-1]
        if slot_idx < len(unit_indices):
            return unit_indices[slot_idx]
        return unit_indices[-1]

    def get_retrieved_kv(self, query=None):
        """retrieve context blocks with retrieved_block_indices
        query: (batch_size, num_heads, length, dim_head)
        return [init_k, retrieved_k] and the respective v
        """
        with profile_section("retrieval.total"):
            if query is not None:  # retrieve based on the attention score between query and context's representative keys
                block_topk = self._calc_block_topk(query)
                self.set_retrieved_block_indices(block_topk)

            assert len(self.retrieved_block_indices) == self.num_units

            global_h_k = self.global_buffer[0]
            global_h_v = self.global_buffer[1]
            slot_counts = [self._unit_slot_count(self.retrieved_block_indices[u]) for u in range(self.num_units)]
            max_slot_count = max(slot_counts, default=0)

            with profile_section("retrieval.kv_load"):
                with torch.cuda.stream(GLOBAL_STREAM):
                    if self.init_exc:  # init KV were loaded in global_h_k, context KV were offloaded in global_blocks
                        # offload LRU blocks
                        for u in range(self.num_units):
                            num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                            selected_blocks = self._unit_selected_blocks(self.retrieved_block_indices[u])
                            for b_idx in selected_blocks:
                                if b_idx not in self.cached_blocks[u]:
                                    num_remove += 1
                            self._remove_lru_blocks(u, num_remove, selected_blocks)

                        self.load_count += 1
                        for u in range(self.num_units):
                            for b_idx in self._unit_selected_blocks(self.retrieved_block_indices[u]):
                                self.cached_blocks[u][b_idx] = self.load_count

                        # no need to load init KV
                        init_st = 0
                        init_ed = init_st + self.init_k.size(-2)
                        ed = init_ed
                        assert self.global_buffer_init_st == init_st or self.global_buffer_init_ed == init_ed
                        if max_slot_count > 0:
                            global_h_k[:, :, init_ed:init_ed + max_slot_count * self.block_size, :].zero_()
                            global_h_v[:, :, init_ed:init_ed + max_slot_count * self.block_size, :].zero_()

                        # load retrieved context KV
                        for u in range(self.num_units):
                            selected_blocks = self._unit_selected_blocks(self.retrieved_block_indices[u])
                            if len(selected_blocks) == 0:
                                continue
                            assert selected_blocks[-1] < self.num_global_block, f'{selected_blocks[-1]}, {self.num_global_block}'
                            for b_idx in selected_blocks:
                                self.global_blocks[u][b_idx].load()
                            for cnt in range(slot_counts[u]):
                                st = init_ed + cnt * self.block_size
                                ed = st + self.block_size
                                if self._use_head_specific_retrieval():
                                    for head_idx in range(self.num_heads_kv):
                                        b_idx = self._get_slot_block_index(self.retrieved_block_indices[u], cnt, head_idx=head_idx)
                                        if b_idx is None:
                                            continue
                                        block_gpu = self.global_blocks[u][b_idx].get()
                                        global_h_k[u, head_idx, st:ed, :].copy_(block_gpu[0, head_idx], non_blocking=True)
                                        global_h_v[u, head_idx, st:ed, :].copy_(block_gpu[1, head_idx], non_blocking=True)
                                else:
                                    b_idx = self._get_slot_block_index(self.retrieved_block_indices[u], cnt)
                                    self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))

                    else:  # init KV and context are in self.global_remainder
                        # load init KV
                        init_st = 0
                        init_ed = init_st + self.n_init
                        global_h_k[:, :, init_st:init_ed] = self.global_remainder[0][:, :, init_st:init_ed]
                        global_h_v[:, :, init_st:init_ed] = self.global_remainder[1][:, :, init_st:init_ed]
                        ed = init_ed
                        if max_slot_count > 0:
                            global_h_k[:, :, init_ed:init_ed + max_slot_count * self.block_size, :].zero_()
                            global_h_v[:, :, init_ed:init_ed + max_slot_count * self.block_size, :].zero_()

                        # load retrieved context KV
                        for u in range(self.num_units):
                            for cnt in range(slot_counts[u]):
                                st = init_ed + cnt * self.block_size
                                ed = st + self.block_size
                                if self._use_head_specific_retrieval():
                                    for head_idx in range(self.num_heads_kv):
                                        b_idx = self._get_slot_block_index(self.retrieved_block_indices[u], cnt, head_idx=head_idx)
                                        if b_idx is None:
                                            continue
                                        remainder_st = init_ed + b_idx * self.block_size
                                        remainder_ed = remainder_st + self.block_size
                                        if remainder_st >= self.global_remainder[0].size(2):
                                            continue
                                        global_h_k[u, head_idx, st:ed] = self.global_remainder[0][u, head_idx, remainder_st:remainder_ed]
                                        global_h_v[u, head_idx, st:ed] = self.global_remainder[1][u, head_idx, remainder_st:remainder_ed]
                                else:
                                    b_idx = self._get_slot_block_index(self.retrieved_block_indices[u], cnt)
                                    remainder_st = init_ed + b_idx * self.block_size
                                    remainder_ed = remainder_st + self.block_size
                                    if remainder_st >= self.global_remainder[0].size(2):
                                        break
                                    global_h_k[u, :, st:ed] = self.global_remainder[0][u, :, remainder_st:remainder_ed]
                                    global_h_v[u, :, st:ed] = self.global_remainder[1][u, :, remainder_st:remainder_ed]

                    global_h_k = global_h_k[:, :, :ed, :]
                    global_h_v = global_h_v[:, :, :ed, :]

                if self.async_global_stream:
                    torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

            assert global_h_k.size(-2) <= self.n_init + self.n_local
            return global_h_k, global_h_v

    def _get_q_token_topk(self, q_len: int) -> int:
        return max(1, math.ceil(self.q_topk_ratio * q_len))

    def _get_k_token_topk(self, k_len: int) -> int:
        return max(1, math.ceil(self.k_topk_ratio * k_len))

    # Get the indices of the top-k vectors in self.block_k[u] that have the highest similarity with global_h_q[u].
    # shared retrieval: batch_size x topk
    # head-specific retrieval: batch_size x n_head_kv x topk
    def _calc_block_topk(
        self, global_h_q
    ):
        with profile_section("retrieval.block_topk_total"):
            return self._calc_block_topk_impl(global_h_q)

    def _calc_block_topk_impl(self, global_h_q):
        if self.retrieval_fusion != "none":
            return self._calc_hybrid_block_topk(global_h_q)

        q_len = global_h_q.size(2)
        query_group_reduction = "mean" if self._use_head_specific_retrieval() else "sum"
        if self.q_repr == "mean":
            global_h_q = global_h_q.mean(dim=2, keepdim=False)  # (batch_size, num_heads, dim_head)
            if self.kv_repr in {"mean", "token"}:
                global_h_q = self._group_query_heads(global_h_q, reduction=query_group_reduction)  # (batch_size, num_heads_kv, dim_head)
            else:
                assert global_h_q.shape == (self.num_units, self.unit_size, self.dim_head)
                global_h_q = global_h_q.reshape(self.num_units, self.dim_head * self.unit_size)  # (batch_size, dim_head * num_heads)
        else:
            if self.kv_repr in {"mean", "token"}:
                global_h_q = self._group_query_heads(global_h_q, reduction=query_group_reduction)  # (batch_size, num_heads_kv, q_len, dim_head)
            else:
                global_h_q = global_h_q.transpose(1, 2).contiguous().reshape(
                    self.num_units, q_len, self.dim_head * self.unit_size
                )  # (batch_size, q_len, dim_head * num_heads)
        logits = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, f'{self.global_remainder[0].shape}'
                global_k = self.global_remainder[0][:, :, self.n_init:, :]  # (batch_size, n_head_kv, length - n_init, dim_head)
                assert global_k.size(-2) % self.block_size == 0, f'{global_k.shape}'
                block_num = global_k.size(-2) // self.block_size  # number of frames in local window
                if block_num <= self.topk:
                    if self._use_head_specific_retrieval():
                        ret = [[list(range(block_num)) for _ in range(self.num_heads_kv)] for _ in range(self.num_units)]
                    else:
                        ret = [list(range(block_num)) for _ in range(self.num_units)]
                else:
                    if self.kv_repr == "mean":
                        global_k = global_k.reshape(self.num_units, self.num_heads_kv, block_num, self.block_size, self.dim_head)
                        global_k = global_k.mean(dim=-2).permute(0, 2, 1, 3).contiguous()  # (batch_size, block_num, num_heads_kv, dim_head)
                    if self.q_repr == "mean":
                        if self.kv_repr == "mean":
                            logits = (global_k.float() * global_h_q[:, None, :, :].float()).sum(dim=-1)  # (batch_size, block_num, num_heads_kv)
                            if self._use_head_specific_retrieval():
                                logits = logits.permute(0, 2, 1).contiguous()  # (batch_size, num_heads_kv, block_num)
                            else:
                                logits = logits.sum(dim=-1)  # (batch_size, block_num)
                        else:
                            global_k = global_k.reshape(
                                self.num_units, self.num_heads_kv, block_num, self.block_size, self.dim_head
                            ).contiguous()  # (batch_size, num_heads_kv, block_num, block_size, dim_head)
                            sim = torch.einsum("bhd,bhnkd->bhnk", global_h_q.float(), global_k.float())  # (batch_size, num_heads_kv, block_num, block_size)
                            if self.k_token_agg == "max":
                                logits = sim.max(dim=-1).values  # (batch_size, num_heads_kv, block_num)
                            else:
                                keep_k = max(1, min(self._get_k_token_topk(self.block_size), sim.size(-1)))
                                logits = sim.topk(keep_k, dim=-1).values.mean(dim=-1)  # (batch_size, num_heads_kv, block_num)
                            if not self._use_head_specific_retrieval():
                                logits = logits.sum(dim=1)  # (batch_size, block_num)
                    else:
                        if self.kv_repr == "mean":
                            sim = torch.einsum("bnhd,bhqd->bnhq", global_k.float(), global_h_q.float())  # (batch_size, block_num, num_heads_kv, q_len)
                            if self._use_head_specific_retrieval():
                                maxsim = sim.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_heads_kv, block_num, q_len)
                            else:
                                maxsim = sim.sum(dim=2)  # (batch_size, block_num, q_len)
                        else:
                            global_k = global_k.reshape(
                                self.num_units, self.num_heads_kv, block_num, self.block_size, self.dim_head
                            ).permute(0, 1, 2, 3, 4).contiguous()  # (batch_size, num_heads_kv, block_num, block_size, dim_head)
                            sim = torch.einsum("bhqd,bhnkd->bhnqk", global_h_q.float(), global_k.float())  # (batch_size, num_heads_kv, block_num, q_len, block_size)
                            if self.k_token_agg == "max":
                                maxsim = sim.max(dim=-1).values  # (batch_size, num_heads_kv, block_num, q_len)
                            else:
                                keep_k = max(1, min(self._get_k_token_topk(self.block_size), sim.size(-1)))
                                maxsim = sim.topk(keep_k, dim=-1).values.mean(dim=-1)  # (batch_size, num_heads_kv, block_num, q_len)
                            if not self._use_head_specific_retrieval():
                                maxsim = maxsim.sum(dim=1)  # (batch_size, block_num, q_len)
                        if self.q_token_agg == "mean":
                            logits = maxsim.mean(dim=-1)  # kv mean: (batch_size, num_heads_kv, block_num); kv token: (batch_size, block_num)
                        else:
                            q_topk = self._get_q_token_topk(q_len)
                            keep = max(1, min(q_topk, maxsim.size(-1)))
                            logits = maxsim.topk(keep, dim=-1).values.mean(dim=-1)  # kv mean: (batch_size, num_heads_kv, block_num); kv token: (batch_size, block_num)
            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                if self._use_head_specific_retrieval():
                    ret = [[list(range(len(self.global_blocks[0]))) for _ in range(self.num_heads_kv)] for _ in range(self.num_units)]
                else:
                    ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        else:
            if self.q_repr == "mean":
                if self.kv_repr == "mean":
                    if self._use_head_specific_retrieval():
                        logits = torch.stack([self.block_k[u].get_head_similarity_mean_q(global_h_q[u]) for u in range(self.num_units)])  # (batch_size, num_heads_kv, block_num)
                    else:
                        logits = torch.stack([
                            self.block_k[u].get_head_similarity_mean_q(global_h_q[u]).sum(dim=0)
                            for u in range(self.num_units)
                        ])  # (batch_size, block_num)
                else:
                    logits = torch.stack([
                        self.block_k[u].get_head_similarity_mean_q(
                            global_h_q[u],
                            k_agg=self.k_token_agg,
                            k_topk_count=self._get_k_token_topk(self.block_size) if self.k_token_agg == "topk" else None,
                        )
                        for u in range(self.num_units)
                    ])  # (batch_size, num_heads_kv, block_num)
                    if not self._use_head_specific_retrieval():
                        logits = logits.sum(dim=1)  # (batch_size, block_num)
            else:
                if self.kv_repr == "mean":
                    if self._use_head_specific_retrieval():
                        logits = torch.stack([
                            self.block_k[u].get_head_similarity_token_q(
                                global_h_q[u],
                                agg=self.q_token_agg,
                                topk_count=self._get_q_token_topk(q_len) if self.q_token_agg == "topk" else None,
                                k_agg=self.k_token_agg,
                                k_topk_count=self._get_k_token_topk(self.block_size) if self.k_token_agg == "topk" else None,
                            )
                            for u in range(self.num_units)
                        ])  # (batch_size, num_heads_kv, block_num)
                    else:
                        logits = torch.stack([
                            (
                                self.block_k[u].get_per_head_token_scores(global_h_q[u]).sum(dim=0).mean(dim=-1)
                                if self.q_token_agg == "mean"
                                else self.block_k[u].get_per_head_token_scores(global_h_q[u]).sum(dim=0).topk(
                                    max(1, min(self._get_q_token_topk(q_len), q_len)), dim=-1
                                ).values.mean(dim=-1)
                            )
                            for u in range(self.num_units)
                        ])  # (batch_size, block_num)
                else:
                    if self._use_head_specific_retrieval():
                        logits = torch.stack([
                            self.block_k[u].get_head_similarity_token_q(
                                global_h_q[u],
                                agg=self.q_token_agg,
                                topk_count=self._get_q_token_topk(q_len) if self.q_token_agg == "topk" else None,
                                k_agg=self.k_token_agg,
                                k_topk_count=self._get_k_token_topk(self.block_size) if self.k_token_agg == "topk" else None,
                            )
                            for u in range(self.num_units)
                        ])  # (batch_size, num_heads_kv, block_num)
                    else:
                        logits = torch.stack([
                            (
                                self.block_k[u].get_per_head_token_scores(
                                    global_h_q[u],
                                    k_agg=self.k_token_agg,
                                    k_topk_count=self._get_k_token_topk(self.block_size) if self.k_token_agg == "topk" else None,
                                ).sum(dim=0).mean(dim=-1)
                                if self.q_token_agg == "mean"
                                else self.block_k[u].get_per_head_token_scores(
                                    global_h_q[u],
                                    k_agg=self.k_token_agg,
                                    k_topk_count=self._get_k_token_topk(self.block_size) if self.k_token_agg == "topk" else None,
                                ).sum(dim=0).topk(
                                    max(1, min(self._get_q_token_topk(q_len), q_len)), dim=-1
                                ).values.mean(dim=-1)
                            )
                            for u in range(self.num_units)
                        ])  # (batch_size, block_num)

        if logits is not None:
            self.similarity = logits
            assert self.topk % self.chunk_size == 0
            if self._use_head_specific_retrieval() and logits.dim() == 3:
                remainder_size = logits.shape[-1] % self.chunk_size
                chunked_logits = logits[..., :logits.shape[-1] - remainder_size].reshape(
                    self.num_units, self.num_heads_kv, -1, self.chunk_size
                ).mean(dim=-1)  # (batch_size, num_heads_kv, block_num // chunk_size)
                if remainder_size > 0:
                    remainder_logits = logits[..., -remainder_size:].mean(dim=-1, keepdim=True)  # (batch_size, num_heads_kv, 1)
                    chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=-1)
                ret = chunked_logits.topk(self.topk // self.chunk_size, dim=-1).indices
                ret = ret.sort(dim=-1)[0][..., None]  # (batch_size, num_heads_kv, topk//chunk_size, 1)
                ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, None, :]
                ret = ret.reshape(self.num_units, self.num_heads_kv, -1)  # (batch_size, num_heads_kv, topk)
                ret = ret.cpu().tolist()
                for u in range(self.num_units):
                    for h in range(self.num_heads_kv):
                        ret[u][h] = list(filter(lambda idx: idx < logits.shape[-1], ret[u][h]))
            else:
                remainder_size = logits.shape[1] % self.chunk_size
                chunked_logits = logits[:, :logits.shape[1]-remainder_size].reshape(self.num_units, -1, self.chunk_size).mean(dim=-1)  # (batch_size, block_num // chunk_size)
                if remainder_size > 0:
                    remainder_logits = logits[:, -remainder_size:].mean(dim=-1, keepdim=True)  # (batch_size, 1)
                    chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
                ret = chunked_logits.topk(self.topk//self.chunk_size, dim=1).indices
                ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
                ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, :]  # (batch_size, topk//chunk_size, chunk_size)
                ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
                ret = ret.cpu().tolist()

                # NOTE: The last chunk might cause an index overflow
                for u in range(self.num_units):
                    ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

            return ret

    # load init KV
    def get_global_hidden_and_mask(self, exc_length):
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # prepare init KV-Cache until it's full
        if not self.init_exc and global_remainder_len > self.n_local:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = min(
                self.n_init - self.init_k.size(-2),
                global_remainder_len - self.n_local
            )
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True  # init KV-Cache is full

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

        # load init KV
        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:  # init KV haven't been loaded into global_h_kv
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)

        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        global_h_k = global_h_k[:, :, :init_ed, :]
        global_h_v = global_h_v[:, :, :init_ed, :]

        return global_h_k, global_h_v

    def _append(
        self,
        local_q, local_k, local_v, global_q,
    ):
        """calculate attention results 

        Args:
            local_q (_type_): (batch_size, num_heads, length, dim_head)
            local_k (_type_): (batch_size, num_heads, length, dim_head)
            local_v (_type_): (batch_size, num_heads, length, dim_head)
            global_q (_type_): (batch_size, num_heads, length, dim_head)

        Returns:
            chunk_o: (batch_size, num_heads, length, dim_head)
        """

        # apply RoPE to input QKV
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        # input Q attends to input + local KV
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            get_score=False, sliding_window=self.n_local
        )

        # load init KV
        with torch.cuda.stream(GLOBAL_STREAM):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(exc_length=global_q.size(-2))

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # input Q attends to init KV
        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True,  # the final append operation
            get_score=False, 
            sliding_window=None,
            complement_sliding_window=True,
        )

        o, _ = attn.get_result()

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def _append_global(
        self
    ):
        """offload context memory
        """

        global_remainder_ed = self._global_remainder_ed
        global_remainder_st = self._global_remainder_st

        global_remainder_len = global_remainder_ed - global_remainder_st

        # offload context KV to CPU
        if self.init_exc:
            assert global_remainder_len % self.block_size == 0, f'global_remainder_len: {global_remainder_len}, block_size: {self.block_size}'
            while global_remainder_len > 0:
                global_remainder_len -= self.block_size

                # Context KV-Cache
                for u in range(self.num_units):
                    self.global_blocks[u].append((
                        MemoryUnit(
                            (
                                self.global_remainder[0][u, :, global_remainder_st:global_remainder_st + self.block_size, :],
                                self.global_remainder[1][u, :, global_remainder_st:global_remainder_st + self.block_size, :]
                            ),
                            self.cuda_cache,
                            False,
                            self.pin_memory
                        )
                    ))

                global_block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :]
                if self.retrieval_fusion != "none":
                    self._append_block_representations(global_block_k)
                elif self.kv_repr in {"mean", "token"}:
                    self._append_block_representations(global_block_k)
                else:
                    global_block_k = self._from_group_kv(global_block_k)  # (batch_size, num_heads, length, dim_head)
                    global_block_k = global_block_k.transpose(1, 2).contiguous().reshape(
                        self.num_units, self.block_size, self.unit_size * self.dim_head
                    )  # (batch_size, block_size, num_heads * dim_head)
                    global_block_k = global_block_k.contiguous()
                    for u in range(self.num_units):
                        self.block_k[u].append(global_block_k[u:u+1])
                
                self.num_global_block += 1
                global_remainder_st += self.block_size

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        # Pre-allocate GPU Memory.
        if not self.initialized:
            self.init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        input_length = local_q.size(-2)
        
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # append local KV
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

        # apply RoPE to global_q
        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        o_list = []
        for st in range(0, input_length, self.exc_block_size):  # Process the input tokens in blocks.
            ed = min(st + self.exc_block_size, input_length)

            # calculate attention results
            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length
            chunk_o = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st: kv_ed, :],
                self.local_v[:, :, kv_st: kv_ed, :],
                global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            # offload context memory
            with torch.cuda.stream(GLOBAL_STREAM):
                self._append_global()

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        self.length += input_length

        # restrict the length of local KV-cache to self.n_local
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :]

        # update global remainder
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        assert not self.init_exc or self._global_remainder_st == self._global_remainder_ed, f'self.init_exc: {self.init_exc}, global_remainder_st: {self._global_remainder_st}, global_remainder_ed: {self._global_remainder_ed}'
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :],
                self.global_remainder[1][:, :, self._global_remainder_st:, :]
            )

        ret = torch.cat(o_list, dim=-2)
        
        return ret
    
    def size(self, *args, **kwargs):
        return self.length

    def calculate_cpu_memory(self):
        memory = 0
        for u in range(self.num_units):
            for block in self.global_blocks[u]:
                memory += block.calculate_cpu_memory()
        return memory
