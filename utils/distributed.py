import os
import torch
import torch.distributed as dist

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank % torch.cuda.device_count())
        print(f"Initialized DDP with rank {rank} and world size {world_size}")
    else:
        print("Not using distributed mode")

def synchronize():
    """Synchronize all processes."""
    dist.barrier()

def get_rank():
    """Get the current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    """Get the total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1

def all_reduce_tensor(tensor):
    """Perform all-reduce on a tensor."""
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor

def cleanup():
    """Cleanup the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()