"""WER computation against ground-truth reference transcripts."""

from jiwer import wer
from pathlib import Path
import pandas as pd


def load_references(ref_dir: str) -> dict[str, str]:
    """Load all .txt reference transcripts from ref_dir."""
    ref_dir = Path(ref_dir)
    return {
        f.stem: f.read_text(encoding="utf-8").strip()
        for f in sorted(ref_dir.glob("*.txt"))
    }


def compute_wer_report(
    hypotheses: dict[str, str],
    references: dict[str, str],
    model_name: str,
    audio_dir: str,
) -> pd.DataFrame:
    """
    Compute per-clip WER and return a tidy DataFrame.

    Columns: clip, model, wer_pct, duration_s, word_count_ref
    """
    audio_dir = Path(audio_dir)
    rows = []

    for clip_id, hyp in hypotheses.items():
        ref = references.get(clip_id)
        if ref is None:
            print(f"  ⚠  No reference found for {clip_id}, skipping.")
            continue

        clip_wer = wer(ref, hyp) * 100  # percent

        # Duration: try to get from audio file via wave / mutagen
        duration = _get_duration(audio_dir / f"{clip_id}.wav")
        if duration is None:
            duration = _get_duration(audio_dir / f"{clip_id}.mp3")

        rows.append(
            {
                "clip": clip_id,
                "model": model_name,
                "wer_pct": round(clip_wer, 2),
                "duration_s": round(duration, 1) if duration else None,
                "word_count_ref": len(ref.split()),
            }
        )

    return pd.DataFrame(rows)


def _get_duration(path: Path) -> float | None:
    """Return audio duration in seconds, or None if unavailable."""
    if not path.exists():
        return None
    try:
        import wave, contextlib

        with contextlib.closing(wave.open(str(path))) as f:
            return f.getnframes() / f.getframerate()
    except Exception:
        return None
