import os
import time
from pathlib import Path
from typing import List, Optional


def build_files_list(
    base_dir: str,
    files_list_path: str,
    extension: Optional[str] = None,
    overwrite: bool = False,
    depth: int = 1,
):
    """
    Scan base_dir and save relative file paths into files_list_path
    """

    base_dir = Path(base_dir)
    files_list_path = Path(files_list_path)

    if files_list_path.exists() and not overwrite:
        return

    files = []

    if depth == 1:
        for p in sorted(base_dir.iterdir()):
            if p.is_file():
                if extension is None or p.suffix == extension:
                    files.append(p.name)
    else:
        for p in sorted(base_dir.rglob("*")):
            if p.is_file():
                if extension is None or p.suffix == extension:
                    files.append(str(p.relative_to(base_dir)))

    files_list_path.parent.mkdir(parents=True, exist_ok=True)

    with open(files_list_path, "w", encoding="utf-8") as f:
        for fp in files:
            f.write(fp + "\n")


def load_files_list(
    base_dir: str,
    files_list_path: str,
    with_extension: Optional[str] = None,
    mkdir: bool = False,
) -> List[str]:
    """
    Load file list and return full paths
    """

    base_dir = Path(base_dir)
    files_list_path = Path(files_list_path)

    if not files_list_path.exists():
        raise FileNotFoundError(f"File list not found: {files_list_path}")

    with open(files_list_path, "r", encoding="utf-8") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    full_paths = []

    for rp in rel_paths:
        p = base_dir / rp
        if with_extension is not None:
            p = p.with_suffix(with_extension)

        if mkdir:
            p.parent.mkdir(parents=True, exist_ok=True)

        full_paths.append(str(p))

    return full_paths


def wait_for_file(path: str, sleep_time: float = 1.0, timeout: Optional[int] = None):
    """
    Block until file exists
    """

    path = Path(path)
    start = time.time()

    while not path.exists():
        time.sleep(sleep_time)
        if timeout is not None and (time.time() - start) > timeout:
            raise TimeoutError(f"Timeout waiting for file: {path}")
