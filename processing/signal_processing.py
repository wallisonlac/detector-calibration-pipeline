"""
Processes raw detector waveforms to extract calibration-relevant features:
peak amplitude, peak time, integrated charge, and a signal/noise
classification based on a threshold on peak amplitude and pulse shape.

This mirrors the type of feature-extraction and calibration validation
routines used in detector/laser calibration pipelines (PhD/CERN work),
adapted here to fully synthetic data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/raw_detector_waveforms.parquet")
PROCESSED_PATH = Path("data/processed/detector_features.parquet")

NOISE_RMS_ESTIMATE = 1.5
SIGNAL_THRESHOLD_SIGMA = 5.0


def extract_features(waveform: np.ndarray) -> dict:
    peak_amplitude = float(np.max(waveform))
    peak_sample = int(np.argmax(waveform))
    integrated_charge = float(np.trapz(np.clip(waveform, 0, None)))
    baseline_rms = float(np.std(waveform[:20]))  # first 20 samples assumed pre-pulse baseline

    is_signal = peak_amplitude > SIGNAL_THRESHOLD_SIGMA * NOISE_RMS_ESTIMATE

    return {
        "peak_amplitude": peak_amplitude,
        "peak_sample": peak_sample,
        "integrated_charge": integrated_charge,
        "baseline_rms": baseline_rms,
        "is_signal_detected": is_signal,
    }


def run() -> None:
    df = pd.read_parquet(RAW_PATH)

    features = df["waveform"].apply(lambda wf: extract_features(np.array(wf)))
    features_df = pd.DataFrame(features.tolist())

    result = pd.concat([df[["event_id", "channel_id", "has_signal_truth"]], features_df], axis=1)

    # Validation: compare detected signal vs ground truth (used only because this is synthetic data)
    true_positive = ((result["is_signal_detected"]) & (result["has_signal_truth"])).sum()
    false_positive = ((result["is_signal_detected"]) & (~result["has_signal_truth"])).sum()
    false_negative = ((~result["is_signal_detected"]) & (result["has_signal_truth"])).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0

    print(f"Signal detection validation on synthetic ground truth:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PROCESSED_PATH, index=False)
    print(f"Processed features for {len(result)} channel-waveforms saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    run()
