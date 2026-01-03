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

def snapshot_code(
    ckpt_dir: str,
    root_dir: Optional[str] = None,
    skip_dirs: Optional[List[str]] = None,
    include_top_level: Optional[List[str]] = None,
    ) -> None:
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
    allowed = set(include_top_level) if include_top_level is not None else None
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        rel = os.path.relpath(dirpath, root_dir)
        # At the repo root, optionally only descend into a small whitelist
        if rel == ".":
            if allowed is not None:
                dirnames[:] = [
                    d for d in dirnames
                    if d not in skip and d in allowed
                ]
            else:
                dirnames[:] = [d for d in dirnames if d not in skip]
        else:
            # Below the root, just respect the skip set
            dirnames[:] = [d for d in dirnames if d not in skip]

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
