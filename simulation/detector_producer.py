"""
Generates synthetic particle-detector waveform data, simulating the type of
high-volume signal output produced by photomultiplier tubes (PMTs) or
similar sensors in a particle-detector calibration setup.

This is a fully synthetic dataset - it contains no real experiment data,
detector geometry, or calibration constants from any real facility.
"""
import numpy as np
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw")

N_CHANNELS = 32
N_EVENTS = 5000
SAMPLES_PER_WAVEFORM = 200
SAMPLING_RATE_NS = 2.0


def simulate_pulse(t: np.ndarray, amplitude: float, t0: float, tau_rise: float, tau_fall: float) -> np.ndarray:
    """Simple bi-exponential pulse shape, typical of PMT/detector signals."""
    pulse = np.zeros_like(t)
    mask = t >= t0
    rise = 1 - np.exp(-(t[mask] - t0) / tau_rise)
    fall = np.exp(-(t[mask] - t0) / tau_fall)
    pulse[mask] = amplitude * rise * fall
    return pulse


def generate_event(event_id: int, rng: np.random.Generator) -> list[dict]:
    t = np.arange(SAMPLES_PER_WAVEFORM) * SAMPLING_RATE_NS
    rows = []

    for channel in range(N_CHANNELS):
        noise = rng.normal(0, 1.5, size=SAMPLES_PER_WAVEFORM)
        has_signal = rng.random() < 0.35

        if has_signal:
            amplitude = rng.uniform(20, 200)
            t0 = rng.uniform(20, 150)
            waveform = simulate_pulse(t, amplitude, t0, tau_rise=5.0, tau_fall=25.0) + noise
        else:
            waveform = noise

        rows.append({
            "event_id": event_id,
            "channel_id": channel,
            "waveform": waveform.tolist(),
            "has_signal_truth": has_signal,
        })

    return rows


def run(n_events: int = N_EVENTS, seed: int = 42) -> None:
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_rows = []
    for event_id in range(n_events):
        all_rows.extend(generate_event(event_id, rng))
        if event_id % 500 == 0:
            print(f"Generated {event_id}/{n_events} events...")

    df = pd.DataFrame(all_rows)
    out_file = RAW_PATH / "raw_detector_waveforms.parquet"
    df.to_parquet(out_file, index=False)
    print(f"Saved {len(df)} channel-waveforms ({n_events} events x {N_CHANNELS} channels) to {out_file}")


if __name__ == "__main__":
    run()
