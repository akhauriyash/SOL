import random
import torch
import shutil
from typing import Optional, List
import os


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def snapshot_code(ckpt_dir: str, root_dir: Optional[str] = None, skip_dirs: Optional[List[str]] = None) -> None:
    """
    Copy all *.py files from the repo into ckpt_dir/'code', preserving
    relative paths and skipping any paths under 'checkpoints'.
    """
    if root_dir is None:
        root_dir = os.getcwd()

    code_root = os.path.join(ckpt_dir, "code")
    os.makedirs(code_root, exist_ok=True)

    skip = {"checkpoints", "__pycache__"}
    if skip_dirs:
        skip.update(skip_dirs)
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        # prune directories we don't want to traverse (by name)
        dirnames[:] = [d for d in dirnames if d not in skip]
        rel = os.path.relpath(dirpath, root_dir)
        dest_dir = code_root if rel == "." else os.path.join(code_root, rel)
        os.makedirs(dest_dir, exist_ok=True)

        for fn in filenames:
            if not fn.endswith((".py", ".yml", ".yaml")):
                continue
            src = os.path.join(dirpath, fn)
            dst = os.path.join(dest_dir, fn)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                # non-fatal; keep training even if a file can't be copied
                print(f"[snapshot_code] Skip {src} -> {dst}: {e}")
