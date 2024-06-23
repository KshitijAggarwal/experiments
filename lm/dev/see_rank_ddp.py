import os
import torch.distributed as dist


def print_info():
    print(f"RANK: {os.environ.get('RANK', 'Not set')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    print(f"LOCAL_WORLD_SIZE: {os.environ.get('LOCAL_WORLD_SIZE', 'Not set')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'Not set')}")


def is_ddp_enabled():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def main():
    ddp_enabled = is_ddp_enabled()
    print(f"DDP Enabled: {ddp_enabled}")
    if ddp_enabled:
        dist.init_process_group(backend="gloo")
        print_info()
        dist.destroy_process_group()
    else:
        print("Running in single-process mode")


if __name__ == "__main__":
    main()

# torchrun --nproc_per_node=2 see_rank_ddp.py
