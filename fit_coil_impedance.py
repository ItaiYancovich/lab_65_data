import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit RL and RLC coil models to PicoScope frequency-sweep CSV data."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the PicoScope CSV file.")
    parser.add_argument(
        "--rref",
        type=float,
        required=True,
        help="Reference resistor value in ohms.",
    )
    parser.add_argument(
        "--freq-col",
        default="Frequency",
        help="Frequency column name in Hz.",
    )
    parser.add_argument(
        "--va-amp-col",
        default="Channel A Amplitude",
        help="Channel A amplitude column name.",
    )
    parser.add_argument(
        "--va-phase-col",
        default="Channel A Phase",
        help="Channel A phase column name in degrees.",
    )
    parser.add_argument(
        "--vb-amp-col",
        default="Channel B Amplitude",
        help="Channel B amplitude column name.",
    )
    parser.add_argument(
        "--vb-phase-col",
        default="Channel B Phase",
        help="Channel B phase column name in degrees.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot instead of showing it.",
    )
    return parser


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        available = ", ".join(data.columns)
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {available}"
        )


def complex_voltage(amplitude: np.ndarray, phase_deg: np.ndarray) -> np.ndarray:
    phase_rad = np.deg2rad(phase_deg)
    return amplitude * np.exp(1j * phase_rad)


def model_rl_mag(frequency_hz: np.ndarray, resistance: float, inductance: float) -> np.ndarray:
    omega = 2.0 * np.pi * frequency_hz
    impedance = resistance + 1j * omega * inductance
    return np.abs(impedance)


def model_rlc_mag(
    frequency_hz: np.ndarray, resistance: float, inductance: float, capacitance: float
) -> np.ndarray:
    omega = 2.0 * np.pi * frequency_hz
    series_rl = resistance + 1j * omega * inductance
    admittance = 1.0 / series_rl + 1j * omega * capacitance
    impedance = 1.0 / admittance
    return np.abs(impedance)


def parameter_errors(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(covariance)
    diagonal = np.where(diagonal < 0.0, np.nan, diagonal)
    return np.sqrt(diagonal)


def initial_rl_guess(frequency_hz: np.ndarray, impedance_mag: np.ndarray) -> tuple[float, float]:
    resistance_guess = max(float(np.nanmin(impedance_mag)), 1e-9)
    inductance_guess = max(
        float(np.nanmedian(impedance_mag / (2.0 * np.pi * np.maximum(frequency_hz, 1e-12)))),
        1e-12,
    )
    return resistance_guess, inductance_guess


def initial_rlc_guess(
    frequency_hz: np.ndarray, impedance_mag: np.ndarray
) -> tuple[float, float, float]:
    resistance_guess, inductance_guess = initial_rl_guess(frequency_hz, impedance_mag)
    capacitance_guess = 1e-9
    return resistance_guess, inductance_guess, capacitance_guess


def fit_models(
    frequency_hz: np.ndarray, impedance_mag: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rl_guess = initial_rl_guess(frequency_hz, impedance_mag)
    rlc_guess = initial_rlc_guess(frequency_hz, impedance_mag)

    rl_params, rl_cov = curve_fit(
        model_rl_mag,
        frequency_hz,
        impedance_mag,
        p0=rl_guess,
        bounds=([0.0, 0.0], [np.inf, np.inf]),
        maxfev=20000,
    )

    rlc_params, rlc_cov = curve_fit(
        model_rlc_mag,
        frequency_hz,
        impedance_mag,
        p0=rlc_guess,
        bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        maxfev=40000,
    )

    return (rl_params, rl_cov), (rlc_params, rlc_cov)


def print_fit_results(name: str, params: np.ndarray, covariance: np.ndarray, labels: list[str]) -> None:
    errors = parameter_errors(covariance)
    print(name)
    for label, value, error in zip(labels, params, errors):
        print(f"  {label} = {value:.6g} +/- {error:.3g}")


def load_experimental_impedance(
    csv_path: Path,
    freq_col: str,
    va_amp_col: str,
    va_phase_col: str,
    vb_amp_col: str,
    vb_phase_col: str,
    rref: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(csv_path)
    require_columns(
        data,
        [freq_col, va_amp_col, va_phase_col, vb_amp_col, vb_phase_col],
    )

    frequency_hz = data[freq_col].to_numpy(dtype=float)
    va = complex_voltage(
        data[va_amp_col].to_numpy(dtype=float),
        data[va_phase_col].to_numpy(dtype=float),
    )
    vb = complex_voltage(
        data[vb_amp_col].to_numpy(dtype=float),
        data[vb_phase_col].to_numpy(dtype=float),
    )

    current = (vb - va) / rref
    valid = np.isfinite(frequency_hz) & np.isfinite(np.abs(va)) & np.isfinite(np.abs(vb))
    valid &= np.abs(current) > 0.0
    valid &= frequency_hz > 0.0

    frequency_hz = frequency_hz[valid]
    current = current[valid]
    va = va[valid]

    impedance = va / current
    impedance_mag = np.abs(impedance)
    return frequency_hz, impedance_mag


def plot_results(
    frequency_hz: np.ndarray,
    impedance_mag: np.ndarray,
    rl_params: np.ndarray,
    rlc_params: np.ndarray,
    output: Path | None,
) -> None:
    freq_fit = np.logspace(np.log10(frequency_hz.min()), np.log10(frequency_hz.max()), 500)

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.loglog(frequency_hz, impedance_mag, "o", label="Experimental", alpha=0.8)
    axis.loglog(freq_fit, model_rl_mag(freq_fit, *rl_params), label="Series RL fit", linewidth=2)
    axis.loglog(freq_fit, model_rlc_mag(freq_fit, *rlc_params), label="RLC fit", linewidth=2)
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("|Z| [Ohm]")
    axis.set_title("Coil impedance magnitude from PicoScope sweep")
    axis.grid(True, which="both", linestyle="--", alpha=0.4)
    axis.legend()
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=200)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.rref <= 0.0:
        raise ValueError("Reference resistor must be positive.")

    frequency_hz, impedance_mag = load_experimental_impedance(
        csv_path=args.csv_path,
        freq_col=args.freq_col,
        va_amp_col=args.va_amp_col,
        va_phase_col=args.va_phase_col,
        vb_amp_col=args.vb_amp_col,
        vb_phase_col=args.vb_phase_col,
        rref=args.rref,
    )

    if frequency_hz.size < 3:
        raise ValueError("Need at least three valid sweep points for fitting.")

    (rl_params, rl_cov), (rlc_params, rlc_cov) = fit_models(frequency_hz, impedance_mag)

    print_fit_results("Model 1: Series RL", rl_params, rl_cov, ["R [Ohm]", "L [H]"])
    print_fit_results("Model 2: RLC", rlc_params, rlc_cov, ["R [Ohm]", "L [H]", "C [F]"])

    plot_results(frequency_hz, impedance_mag, rl_params, rlc_params, args.output)


if __name__ == "__main__":
    main()