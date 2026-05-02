import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple
import numpy as np
import torch
from .utils import ensure_dir, resolve_path


VOCAB_SIZE = 256  # byte-level tokenizer


@dataclass
class PreparedDataset:
    source_dir: Path
    prepared_dir: Path
    train_path: Path
    valid_path: Path
    test_path: Path
    lengths: Dict[str, int]
    vocab_size: int = VOCAB_SIZE


class ByteTokenizer:
    vocab_size: int = VOCAB_SIZE

    def encode_bytes(self, data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.uint8).copy()

    def decode_bytes(self, array: np.ndarray) -> str:
        return bytes(array.tolist()).decode("utf-8", errors="ignore")


class RandomChunkBatcher:
    def __init__(
        self,
        array: np.ndarray,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if len(array) <= block_size + 1:
            raise ValueError(
                f"Need at least block_size+2 tokens, got {len(array)} and block_size={block_size}."
            )
        self.array = array
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.max_start = len(array) - block_size - 1

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        starts = np.random.randint(0, self.max_start, size=self.batch_size)
        x = np.stack([self.array[i : i + self.block_size] for i in starts], axis=0)
        y = np.stack([self.array[i + 1 : i + 1 + self.block_size] for i in starts], axis=0)
        xb = torch.from_numpy(x.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
        yb = torch.from_numpy(y.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
        return xb, yb


class SequentialChunkBatcher:
    def __init__(
        self,
        array: np.ndarray,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if len(array) <= block_size + 1:
            raise ValueError(
                f"Need at least block_size+2 tokens, got {len(array)} and block_size={block_size}."
            )
        self.array = array
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.max_start = len(array) - block_size - 1

    def iter_batches(self, num_batches: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        total_needed = num_batches * self.batch_size
        if total_needed <= 1:
            starts = np.array([0], dtype=np.int64)
        else:
            starts = np.linspace(0, self.max_start - 1, num=total_needed, dtype=np.int64)
        for i in range(num_batches):
            batch_starts = starts[i * self.batch_size : (i + 1) * self.batch_size]
            x = np.stack([self.array[s : s + self.block_size] for s in batch_starts], axis=0)
            y = np.stack(
                [self.array[s + 1 : s + 1 + self.block_size] for s in batch_starts], axis=0
            )
            xb = torch.from_numpy(x.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
            yb = torch.from_numpy(y.astype(np.int64, copy=False)).to(self.device, non_blocking=True)
            yield xb, yb



def _encode_text_file(input_path: Path, output_path: Path) -> int:
    tokenizer = ByteTokenizer()
    with open(input_path, "rb") as f:
        data = f.read()
    encoded = tokenizer.encode_bytes(data)
    np.save(output_path, encoded)
    return int(encoded.shape[0])



def prepare_dataset(data_config: Dict[str, object]) -> PreparedDataset:
    source_dir = resolve_path(str(data_config["source_dir"]))
    prepared_dir = ensure_dir(resolve_path(str(data_config["prepared_dir"])))

    split_filenames = {
        "train": source_dir / str(data_config.get("train_file", "train.txt")),
        "valid": source_dir / str(data_config.get("valid_file", "valid.txt")),
        "test": source_dir / str(data_config.get("test_file", "test.txt")),
    }
    for split, path in split_filenames.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split at {path}")

    manifest_path = prepared_dir / "manifest.json"
    overwrite = bool(data_config.get("overwrite_prepared", False))

    if manifest_path.exists() and not overwrite:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return PreparedDataset(
            source_dir=source_dir,
            prepared_dir=prepared_dir,
            train_path=prepared_dir / manifest["train_file"],
            valid_path=prepared_dir / manifest["valid_file"],
            test_path=prepared_dir / manifest["test_file"],
            lengths=manifest["lengths"],
            vocab_size=manifest.get("vocab_size", VOCAB_SIZE),
        )

    lengths: Dict[str, int] = {}
    output_files = {}
    for split, input_path in split_filenames.items():
        output_path = prepared_dir / f"{split}.npy"
        lengths[split] = _encode_text_file(input_path, output_path)
        output_files[split] = output_path.name

    manifest = {
        "source_dir": str(source_dir),
        "prepared_dir": str(prepared_dir),
        "train_file": output_files["train"],
        "valid_file": output_files["valid"],
        "test_file": output_files["test"],
        "lengths": lengths,
        "vocab_size": VOCAB_SIZE,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return PreparedDataset(
        source_dir=source_dir,
        prepared_dir=prepared_dir,
        train_path=prepared_dir / output_files["train"],
        valid_path=prepared_dir / output_files["valid"],
        test_path=prepared_dir / output_files["test"],
        lengths=lengths,
        vocab_size=VOCAB_SIZE,
    )



def load_split_array(path: str | Path) -> np.ndarray:
    array = np.load(path, mmap_mode=None)
    return np.asarray(array, dtype=np.uint8)
