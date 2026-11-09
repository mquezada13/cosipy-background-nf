"""Sample one of Pascal's time-independent NF background components.

Run from the project root, for example:

    .venv/bin/python src/sample_time_independent_background.py --part 1 --samples 100

The output contains CDS coordinates, not physical event times or expected
counts for an observation. The number of samples is chosen by the user; it
does not use the constant rate stored in the model. Existing files are never
overwritten. For large samples, increase --samples and use --batch-size to
control memory use.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from cosipy.background_estimation.ml import NFBackground


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COORDINATES = ("Energies", "Phi", "Chi local", "Psi local")


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--samples", type=positive_integer, required=True)
    parser.add_argument("--batch-size", type=positive_integer, default=100_000)
    parser.add_argument("--output", type=Path, help="Output HDF5 path; must not already exist.")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / f"model/nfbackground_part-{args.part:02d}_v3-00.fits"
    output_path = args.output or (
        PROJECT_ROOT / "data/NF_background" / f"part-{args.part:02d}_samples_{args.samples}.hdf5"
    )
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    background = NFBackground(model_path, devices=["cpu"], density_compile_mode=None, show_progress=False)
    if background._version.split(".")[0] != "3":
        raise ValueError(f"Expected a version 3 model, got {background._version}")

    # COSIpy's interface requires one context value per event, although the
    # version 3 density ignores time. These zeros are not event times.
    with h5py.File(output_path, "x") as output:
        output.attrs["model_file"] = model_path.name
        output.attrs["model_version"] = background._version
        output.attrs["part"] = args.part
        output.attrs["requested_samples"] = args.samples
        output.attrs["completed_samples"] = 0
        output.attrs["description"] = "Time-independent NF shape samples; no physical times or rate normalization"
        for name in COORDINATES:
            dataset = output.create_dataset(name, shape=(args.samples,), dtype="f4", chunks=True)
            dataset.attrs["unit"] = "keV" if name == "Energies" else "rad"

        background.init_compute_pool(["cpu"])
        try:
            for start in range(0, args.samples, args.batch_size):
                stop = min(start + args.batch_size, args.samples)
                context = torch.zeros((stop - start, 1), dtype=torch.float64)
                samples = background.sample_density(context).numpy()
                if samples.shape != (stop - start, 4) or not np.isfinite(samples).all():
                    raise ValueError("Sampler returned invalid coordinates.")
                for index, name in enumerate(COORDINATES):
                    output[name][start:stop] = samples[:, index]
                output.attrs["completed_samples"] = stop
                output.flush()
                print(f"Sampled {stop:,} / {args.samples:,} events", flush=True)
        finally:
            background.shutdown_compute_pool()

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
