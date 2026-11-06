import gc
import torch


def memory_cleanup(verbose: bool = False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"[VRAM] allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")


def print_vram(tag: str = ""):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[VRAM {tag}] allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")