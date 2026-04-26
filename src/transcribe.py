"""Whisper inference logic for batch transcription."""

import whisper
from pathlib import Path
from tqdm import tqdm


def transcribe_clips(audio_dir: str, model_name: str) -> dict[str, str]:
    """
    Transcribe all audio clips in audio_dir using the specified Whisper model.

    Args:
        audio_dir: Path to directory containing .wav / .mp3 files.
        model_name: Whisper model size — 'small' or 'medium'.

    Returns:
        Dict mapping clip stem (e.g. 'clip_001') to transcription string.
    """
    model = whisper.load_model(model_name)
    audio_dir = Path(audio_dir)
    clips = sorted(audio_dir.glob("*.wav")) + sorted(audio_dir.glob("*.mp3"))

    if not clips:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    results = {}
    for clip in tqdm(clips, desc=f"[{model_name}]", unit="clip"):
        result = model.transcribe(str(clip))
        results[clip.stem] = result["text"].strip()

    return results
