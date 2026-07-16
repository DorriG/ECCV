from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Example:
    sample_id: str
    video_path: str
    label: int
    transcript: str
    audio_path: Path


def parse_manifest_line(line: str) -> tuple[str, int, str] | None:
    line = line.strip()
    if not line:
        return None

    lower = line.lower()
    if lower.startswith("video_path,") or lower.startswith("video-path,"):
        return None

    parts = line.split(",", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid manifest line: {line}")

    video_path = parts[0].strip()
    label = int(parts[1].strip())
    transcript = parts[2].strip() if len(parts) == 3 else ""
    return video_path, label, transcript


def video_path_to_audio_path(
    video_path: str,
    audio_root: Path,
    audio_extension: str = ".wav",
    strip_video_prefix: str = "Videos",
) -> Path:
    posix_path = PurePosixPath(video_path.replace("\\", "/"))
    parts = list(posix_path.parts)
    if parts and parts[0] == strip_video_prefix:
        parts = parts[1:]
    return audio_root / Path(*parts).with_suffix(audio_extension)


def sample_id_from_video_path(video_path: str) -> str:
    return Path(PurePosixPath(video_path.replace("\\", "/")).name).stem


def load_examples(
    manifest_path: Path,
    audio_root: Path,
    audio_extension: str = ".wav",
    strip_video_prefix: str = "Videos",
    limit: int | None = None,
) -> list[Example]:
    examples: list[Example] = []
    with manifest_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parsed = parse_manifest_line(line)
            if parsed is None:
                continue
            video_path, label, transcript = parsed
            audio_path = video_path_to_audio_path(
                video_path=video_path,
                audio_root=audio_root,
                audio_extension=audio_extension,
                strip_video_prefix=strip_video_prefix,
            )
            examples.append(
                Example(
                    sample_id=sample_id_from_video_path(video_path),
                    video_path=video_path,
                    label=label,
                    transcript=transcript,
                    audio_path=audio_path,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(info.min), info.max)
    else:
        audio = audio.astype(np.float32)
    return audio


def load_audio_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sampling_rate = sf.read(str(audio_path), always_2d=False)
    audio = _to_float32(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, int(sampling_rate)


def resample_audio(
    audio: np.ndarray,
    source_sampling_rate: int,
    target_sampling_rate: int,
) -> np.ndarray:
    if source_sampling_rate == target_sampling_rate:
        return audio.astype(np.float32, copy=False)

    gcd = math.gcd(source_sampling_rate, target_sampling_rate)
    up = target_sampling_rate // gcd
    down = source_sampling_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def crop_audio(
    audio: np.ndarray,
    sampling_rate: int,
    max_seconds: float | None,
    crop_mode: str,
) -> np.ndarray:
    if max_seconds is None or max_seconds <= 0:
        return audio

    max_samples = int(round(max_seconds * sampling_rate))
    if max_samples <= 0 or audio.shape[0] <= max_samples:
        return audio

    if crop_mode == "random":
        start = random.randint(0, audio.shape[0] - max_samples)
    elif crop_mode == "center":
        start = (audio.shape[0] - max_samples) // 2
    elif crop_mode == "first":
        start = 0
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")
    return audio[start : start + max_samples]


class BAHTextAudioDataset(Dataset):
    """BAH split file dataset using transcript text and matching wav files."""

    def __init__(
        self,
        manifest_path: Path,
        audio_root: Path,
        target_sampling_rate: int,
        max_audio_seconds: float | None = 30.0,
        crop_mode: str = "center",
        limit: int | None = None,
        audio_extension: str = ".wav",
        strip_video_prefix: str = "Videos",
    ) -> None:
        self.examples = load_examples(
            manifest_path=manifest_path,
            audio_root=audio_root,
            audio_extension=audio_extension,
            strip_video_prefix=strip_video_prefix,
            limit=limit,
        )
        self.target_sampling_rate = target_sampling_rate
        self.max_audio_seconds = max_audio_seconds
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        audio, sampling_rate = load_audio_mono(example.audio_path)
        audio = resample_audio(audio, sampling_rate, self.target_sampling_rate)
        audio = crop_audio(
            audio=audio,
            sampling_rate=self.target_sampling_rate,
            max_seconds=self.max_audio_seconds,
            crop_mode=self.crop_mode,
        )
        return {
            "sample_id": example.sample_id,
            "video_path": example.video_path,
            "audio_path": str(example.audio_path),
            "transcript": example.transcript,
            "label": int(example.label),
            "audio": audio,
        }


class TextAudioCollator:
    def __init__(
        self,
        tokenizer: Any,
        audio_feature_extractor: Any,
        sampling_rate: int,
        max_text_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.audio_feature_extractor = audio_feature_extractor
        self.sampling_rate = sampling_rate
        self.max_text_length = max_text_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        transcripts = [item["transcript"] for item in batch]
        tokenized = self.tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        audio_arrays = [item["audio"] for item in batch]
        try:
            audio_features = self.audio_feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        except TypeError:
            audio_features = self.audio_feature_extractor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                padding=True,
                return_tensors="pt",
            )

        audio_values = audio_features["input_values"]
        audio_attention_mask = audio_features.get("attention_mask")
        if audio_attention_mask is None:
            audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "audio_values": audio_values,
            "audio_attention_mask": audio_attention_mask,
            "labels": labels,
            "sample_ids": [item["sample_id"] for item in batch],
            "video_paths": [item["video_path"] for item in batch],
            "audio_paths": [item["audio_path"] for item in batch],
            "transcripts": transcripts,
        }


def build_dataloader(
    dataset: Dataset,
    collator: TextAudioCollator,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

