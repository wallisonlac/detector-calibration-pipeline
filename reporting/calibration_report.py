"""
Generates a calibration summary report from processed detector features:
per-channel signal detection rate, average peak amplitude, and flags
channels with anomalous behaviour (potential hardware issues), mirroring
the type of automated calibration QA routine used in detector operations.
"""
import pandas as pd
from pathlib import Path

PROCESSED_PATH = Path("data/processed/detector_features.parquet")
REPORT_PATH = Path("data/processed/calibration_summary.csv")

ANOMALY_DETECTION_RATE_THRESHOLD = (0.15, 0.55)  # expected fraction of events with signal


def run() -> None:
    df = pd.read_parquet(PROCESSED_PATH)

    summary = (
        df.groupby("channel_id")
        .agg(
            n_events=("event_id", "count"),
            detection_rate=("is_signal_detected", "mean"),
            avg_peak_amplitude=("peak_amplitude", "mean"),
            avg_integrated_charge=("integrated_charge", "mean"),
            avg_baseline_rms=("baseline_rms", "mean"),
        )
        .reset_index()
    )

    low, high = ANOMALY_DETECTION_RATE_THRESHOLD
    summary["anomalous_channel"] = ~summary["detection_rate"].between(low, high)

    n_anomalous = summary["anomalous_channel"].sum()
    print(f"Calibration summary generated for {len(summary)} channels.")
    print(f"Flagged {n_anomalous} channel(s) as anomalous (detection rate outside [{low}, {high}]).")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORT_PATH, index=False)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    run()
