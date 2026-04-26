"""whisper-wer-bench — main entry point."""

import argparse
import pandas as pd
from pathlib import Path

from src.transcribe import transcribe_clips
from src.evaluate import load_references, compute_wer_report
from src.visualize import plot_per_clip_comparison, plot_wer_distribution

AUDIO_DIR = "data/audio"
REF_DIR = "data/references"
OUTPUT_CSV = "outputs/report.csv"
PLOTS_DIR = "outputs/plots"


def main():
    parser = argparse.ArgumentParser(description="Whisper WER Benchmark")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["small"],
        choices=["tiny", "base", "small", "medium", "large"],
        help="One or more Whisper model sizes to evaluate.",
    )
    parser.add_argument(
        "--export",
        choices=["csv", "none"],
        default="csv",
        help="Export format for the report.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    args = parser.parse_args()

    references = load_references(REF_DIR)
    print(f"✔ Loaded {len(references)} reference transcripts\n")

    all_frames: list[pd.DataFrame] = []

    for model_name in args.model:
        print(f"── Running Whisper {model_name} {'─' * 40}")
        hypotheses = transcribe_clips(AUDIO_DIR, model_name)
        df = compute_wer_report(hypotheses, references, model_name, AUDIO_DIR)
        avg = df["wer_pct"].mean()
        print(f"   avg WER [{model_name}]: {avg:.2f}%\n")
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    if args.export == "csv":
        Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUT_CSV, index=False)
        print(f"✔ Report exported → {OUTPUT_CSV}")

    if not args.no_plots:
        Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
        print(f"\n── Generating plots ────────────────────────────────────")
        plot_per_clip_comparison(combined, f"{PLOTS_DIR}/model_comparison.png")
        plot_wer_distribution(combined, f"{PLOTS_DIR}/wer_distribution.png")

    print("\nDone. ✔")


if __name__ == "__main__":
    main()
