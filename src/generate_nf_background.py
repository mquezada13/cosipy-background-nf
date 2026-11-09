"""Generate a three-month DC4 event sample from Pascal's NF checkpoint.

The ``.pt`` checkpoint is a trained model, not an event file. COSIpy's
``NFBackground`` class reads the checkpoint and exposes its two components:

1. ``evaluate_rate(time)`` predicts the background event rate.
2. ``sample_density(time)`` generates event coordinates at supplied times.

This file provides the surrounding workflow: it combines Pascal's trained
model with the DC4 observation times, draws one statistical realization, and
writes the generated events progressively to HDF5.

Responsibility map
------------------
Pascal's checkpoint
    Learned weights and metadata for the time-dependent rate and event density.
COSIpy
    ``NFBackground`` loads and evaluates the checkpoint. ``SpacecraftHistory``
    reads the DC4 observation times and livetimes.
This script
    Integrates the rate, draws event times, calls the NF, and writes the output.

Run from the repository root with::

    python src/generate_nf_background.py
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

# COSIpy reads the spacecraft history used to define the DC4 observation.
from cosipy import SpacecraftHistory

# Pascal implemented NFBackground inside COSIpy. This is the interface to his
# trained rate and density models stored in the .pt checkpoint.
from cosipy.background_estimation.ml import NFBackground
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Pascal's checkpoint: model architecture metadata plus trained weights.
MODEL_PATH = PROJECT_ROOT / "model/nfbackground_v2-00.pt"

# DC4 observation timeline and detector livetime, read by COSIpy.
ORIENTATION_PATH = PROJECT_ROOT / "data/DC4_orientation.ori"

# Files created by this workflow; they are not contained in the checkpoint.
TEST_OUTPUT_PATH = PROJECT_ROOT / "data/nf_background_test.hdf5"
FULL_OUTPUT_PATH = PROJECT_ROOT / "data/nf_background_3months.hdf5"
# Use separate reproducible random streams for the total count and event times.
EVENT_COUNT_SEED = 42
EVENT_TIME_SEED = 43

# These block sizes limit memory use; they do not shorten the observation.
RATE_BLOCK_SIZE = 100
EVENT_GENERATION_BLOCK_SIZE = 10_000
TEST_EVENT_COUNT = 100


def load_inputs():
    """Load Pascal's checkpoint and the DC4 observation with COSIpy.

    Passing ``MODEL_PATH`` to ``NFBackground`` is the point where the ``.pt``
    file is used. Internally, COSIpy opens the checkpoint, reconstructs the
    trained model architecture, and loads Pascal's learned weights.
    """
    # NFBackground is Pascal's COSIpy interface. After this call, ``background``
    # can evaluate the trained rate and sample the trained event density.
    background = NFBackground(
        MODEL_PATH,
        devices=["cpu"],
        # Compilation was disabled because it was unstable in this environment.
        density_compile_mode=None,
        show_progress=False,
    )
    print("NF model loaded successfully")

    # SpacecraftHistory is a COSIpy class, independent of Pascal's NF model.
    spacecraft_history = SpacecraftHistory.open(ORIENTATION_PATH)
    print(
        "Observation:",
        spacecraft_history.tstart,
        "to",
        spacecraft_history.tstop,
    )
    return background, spacecraft_history


def calculate_expected_events(
    background,
    spacecraft_history,
    block_size=RATE_BLOCK_SIZE,
):
    """Calculate the expected count contributed by every time interval.

    The rate is evaluated at both boundaries of every interval and integrated
    with the trapezoidal rule. The resulting mean rate is multiplied by that
    interval's livetime.
    ``block_size`` controls only how many intervals are evaluated at once; it
    does not select a duration or limit the three-month period.

    Small blocks are intentional here because large calls to ``evaluate_rate``
    caused segmentation faults in the current local PyTorch environment.
    """
    # COSIpy stores N interval livetimes between N + 1 time boundaries.
    time_boundaries = spacecraft_history.obstime
    number_intervals = len(time_boundaries) - 1
    print("Intervals:", number_intervals)

    rates_at_boundaries = np.empty(len(time_boundaries), dtype=np.float64)

    for start in range(0, len(time_boundaries), block_size):
        stop = min(start + block_size, len(time_boundaries))

        # Convert through a Python list to avoid unsafe Astropy/NumPy sharing.
        unix_times = time_boundaries[start:stop].to_value("unix").tolist()
        time_context = torch.tensor(
            unix_times,
            dtype=torch.float64,
        ).reshape(-1, 1)

        # Pascal/COSIpy call: evaluate the trained rate model from the .pt file.
        # The result is the predicted background rate at each supplied time.
        rates_at_boundaries[start:stop] = np.asarray(
            background.evaluate_rate(time_context).tolist(),
            dtype=np.float64,
        )

    livetimes = np.asarray(
        spacecraft_history.livetime.to_value("s"),
        dtype=np.float64,
    )
    if len(livetimes) != number_intervals:
        raise ValueError(
            "SpacecraftHistory must provide one livetime per time interval."
        )

    # Pascal's prescription uses trapezoidal rate integration. Multiplying the
    # mean endpoint rate by livetime accounts for the detector's live exposure.
    mean_rates = 0.5 * (
        rates_at_boundaries[:-1] + rates_at_boundaries[1:]
    )
    expected_events_per_interval = mean_rates * livetimes

    if not np.all(np.isfinite(expected_events_per_interval)):
        raise ValueError("Expected event counts must all be finite.")
    if np.any(expected_events_per_interval < 0):
        raise ValueError("Expected event counts must all be nonnegative.")

    return expected_events_per_interval


def build_time_cdf(expected_events_per_interval):
    """Convert interval expectations into a temporal sampling distribution.

    This implements the temporal-sampling prescription described by Pascal; it
    is not a method supplied by ``NFBackground``.

    The implicit CDF value at the first interval boundary is zero, and the
    returned cumulative values end at one. Intervals with more expected events
    occupy a larger part of the CDF and are therefore sampled more often. Kahan
    summation limits round-off error across the 531,997 intervals.
    """
    expected_events_per_interval = np.asarray(
        expected_events_per_interval,
        dtype=np.float64,
    )
    if expected_events_per_interval.ndim != 1:
        raise ValueError("Expected event counts must be a one-dimensional array.")
    if expected_events_per_interval.size == 0:
        raise ValueError("At least one time interval is required.")
    if not np.all(np.isfinite(expected_events_per_interval)):
        raise ValueError("Expected event counts must all be finite.")
    if np.any(expected_events_per_interval < 0):
        raise ValueError("Expected event counts must all be nonnegative.")

    cumulative_expected_events = np.empty_like(expected_events_per_interval)
    running_total = 0.0
    correction = 0.0

    for index, expected_count in enumerate(expected_events_per_interval):
        corrected_count = expected_count - correction
        updated_total = running_total + corrected_count
        correction = (updated_total - running_total) - corrected_count
        running_total = updated_total
        cumulative_expected_events[index] = running_total

    if not np.isfinite(running_total) or running_total <= 0:
        raise ValueError("The total expected event count must be positive and finite.")

    time_cdf = cumulative_expected_events / running_total
    # Fix the endpoint exactly at one so searchsorted cannot exceed the final
    # interval because of accumulated floating-point roundoff.
    time_cdf[-1] = 1.0
    total_expected_events = running_total
    return time_cdf, total_expected_events


def draw_event_count(expected_events, seed=EVENT_COUNT_SEED):
    """Draw one event count following Pascal's Poisson prescription.

    NumPy performs the random draw. The expected count came from Pascal's rate
    model, but this helper function belongs to our workflow.
    """
    if not np.isfinite(expected_events) or expected_events < 0:
        raise ValueError("The expected event count must be nonnegative and finite.")

    random_generator = np.random.default_rng(seed)
    return int(random_generator.poisson(expected_events))


def sample_event_times(
    time_cdf,
    interval_starts,
    interval_stops,
    number_events,
    random_generator,
):
    """Draw continuous times following the NF-predicted background rate.

    This is our implementation of Pascal's inverse-CDF sampling prescription.
    It selects times only; the NF event coordinates are generated later.
    """
    time_cdf = np.asarray(time_cdf, dtype=np.float64)
    interval_starts = np.asarray(interval_starts, dtype=np.float64)
    interval_stops = np.asarray(interval_stops, dtype=np.float64)

    if not (
        len(time_cdf) == len(interval_starts) == len(interval_stops)
    ):
        raise ValueError("CDF and interval arrays must have the same length.")
    if number_events < 0:
        raise ValueError("The number of events must be nonnegative.")
    if not np.all(np.diff(time_cdf) >= 0) or time_cdf[-1] != 1.0:
        raise ValueError("The time CDF must be monotonic and end at one.")
    if np.any(interval_stops <= interval_starts):
        raise ValueError("Every time interval must have positive duration.")

    # First choose a spacecraft interval according to its trapezoid-integrated
    # expected event count.
    interval_indices = np.searchsorted(
        time_cdf,
        random_generator.random(number_events),
        side="right",
    )

    # Then place each event uniformly inside its selected 15-second interval.
    interval_fractions = random_generator.random(number_events)

    return interval_starts[interval_indices] + interval_fractions * (
        interval_stops[interval_indices] - interval_starts[interval_indices]
    )


def prepare_event_datasets(event_times, event_samples):
    """Map Pascal's four NF outputs to explicit event dataset names.

    ``sample_density`` returns columns in this order: measured energy, Compton
    angle, local azimuth, and local polar angle. This workflow assigns those
    columns the names used in the output HDF5 file.
    """
    samples = np.asarray(event_samples.tolist(), dtype=np.float32)

    return {
        "TimeTags": np.asarray(event_times, dtype=np.float64),
        "Energies": samples[:, 0],
        "Phi": samples[:, 1],
        "Chi local": samples[:, 2],
        "Psi local": samples[:, 3],
    }


def write_event_file(output_path, event_times, event_samples):
    """Write a complete event sample to a new compressed HDF5 file.

    This is local file handling with h5py; neither COSIpy nor the checkpoint
    writes this file for us.
    """
    datasets = prepare_event_datasets(event_times, event_samples)

    with h5py.File(output_path, "w") as output_file:
        for name, values in datasets.items():
            output_file.create_dataset(name, data=values, compression="gzip")


def open_or_resume_output(output_path, target_events, random_generator):
    """Create a resumable HDF5 output or reopen an interrupted one.

    Resuming is an engineering feature of this workflow, not part of Pascal's
    trained model.
    """
    field_types = {
        "TimeTags": np.float64,
        "Energies": np.float32,
        "Phi": np.float32,
        "Chi local": np.float32,
        "Psi local": np.float32,
    }

    if output_path.exists():
        # Continue at the last block that was fully saved.
        output_file = h5py.File(output_path, "r+")
        if int(output_file.attrs["target_events"]) != target_events:
            output_file.close()
            raise ValueError("Existing output has a different target event count.")

        completed_events = int(output_file.attrs["completed_events"])
        for dataset in output_file.values():
            dataset.resize((completed_events,))

        random_generator.bit_generator.state = json.loads(
            output_file.attrs["time_rng_state"]
        )
        return output_file, completed_events

    # Create empty, expandable datasets for a new production run.
    output_file = h5py.File(output_path, "w")
    for name, dtype in field_types.items():
        output_file.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(EVENT_GENERATION_BLOCK_SIZE,),
            compression="gzip",
            dtype=dtype,
        )

    output_file.attrs["target_events"] = target_events
    output_file.attrs["completed_events"] = 0
    output_file.attrs["time_rng_state"] = json.dumps(
        random_generator.bit_generator.state
    )
    output_file.flush()
    return output_file, 0


def append_event_block(output_file, event_times, event_samples, completed_events):
    """Append one generated block to the local HDF5 output."""
    block_size = len(event_times)
    block_stop = completed_events + block_size
    values_by_name = prepare_event_datasets(event_times, event_samples)

    for name, values in values_by_name.items():
        dataset = output_file[name]
        dataset.resize((block_stop,))
        dataset[completed_events:block_stop] = values

    return block_stop


def generate_test_background(background, spacecraft_history, time_cdf):
    """Use Pascal's NF to generate a small interface-validation sample."""
    interval_starts = spacecraft_history.obstime[:-1].to_value("unix")
    interval_stops = spacecraft_history.obstime[1:].to_value("unix")
    random_generator = np.random.default_rng(EVENT_TIME_SEED)

    test_times = sample_event_times(
        time_cdf,
        interval_starts,
        interval_stops,
        number_events=TEST_EVENT_COUNT,
        random_generator=random_generator,
    )
    time_context = torch.tensor(
        test_times.tolist(),
        dtype=torch.float64,
    ).reshape(-1, 1)
    # Pascal/COSIpy call: use the density model loaded from the .pt checkpoint.
    # One row [Energy, Phi, Chi local, Psi local] is returned for each time.
    test_samples = background.sample_density(time_context)

    print(f"Test time range: {test_times.min():.3f} to {test_times.max():.3f}")
    print(f"NF test sample shape: {tuple(test_samples.shape)}")
    print("NF test sample minima:", test_samples.min(dim=0).values.tolist())
    print("NF test sample maxima:", test_samples.max(dim=0).values.tolist())

    write_event_file(TEST_OUTPUT_PATH, test_times, test_samples)
    print(f"Test events written to: {TEST_OUTPUT_PATH}")


def generate_full_background(
    background,
    spacecraft_history,
    time_cdf,
    target_events,
):
    """Generate and save the full background without holding it in memory.

    Time generation and HDF5 batching belong to this workflow. The physical
    event coordinates in each block come from Pascal's trained density model.
    """
    interval_starts = spacecraft_history.obstime[:-1].to_value("unix")
    interval_stops = spacecraft_history.obstime[1:].to_value("unix")
    time_generator = np.random.default_rng(EVENT_TIME_SEED)
    output_file, completed_events = open_or_resume_output(
        FULL_OUTPUT_PATH,
        target_events,
        time_generator,
    )

    if completed_events == target_events:
        output_file.close()
        print(f"Full NF background already complete: {FULL_OUTPUT_PATH}")
        return

    # COSIpy prepares the worker used by Pascal's NF density sampler.
    background.init_compute_pool(["cpu"])
    try:
        with output_file, tqdm(
            total=target_events,
            initial=completed_events,
            unit="events",
            desc="Generating NF background",
        ) as progress:
            while completed_events < target_events:
                number_events = min(
                    EVENT_GENERATION_BLOCK_SIZE,
                    target_events - completed_events,
                )
                event_times = sample_event_times(
                    time_cdf,
                    interval_starts,
                    interval_stops,
                    number_events,
                    time_generator,
                )
                time_context = torch.tensor(
                    event_times.tolist(),
                    dtype=torch.float64,
                ).reshape(-1, 1)
                # Pascal/COSIpy call: generate the four event coordinates
                # conditioned on the sampled event times.
                event_samples = background.sample_density(time_context)

                completed_events = append_event_block(
                    output_file,
                    event_times,
                    event_samples,
                    completed_events,
                )
                output_file.attrs["completed_events"] = completed_events
                output_file.attrs["time_rng_state"] = json.dumps(
                    time_generator.bit_generator.state
                )
                output_file.flush()
                progress.update(number_events)
    finally:
        # COSIpy cleanup for the NF worker process.
        background.shutdown_compute_pool()

    print(f"Full NF background written to: {FULL_OUTPUT_PATH}")


def main():
    """Run the complete three-month NF background generation workflow."""
    # Step 1: Load Pascal's trained model and the DC4 observation timeline.
    background, spacecraft_history = load_inputs()

    # Step 2: Integrate rate * livetime over every spacecraft interval.
    expected_events_per_interval = calculate_expected_events(
        background,
        spacecraft_history,
    )

    # Step 3: Build the time distribution and draw one Poisson realization.
    time_cdf, total_expected_events = build_time_cdf(
        expected_events_per_interval
    )
    target_event_count = draw_event_count(total_expected_events)

    print(
        f"Expected NF events in three months: {total_expected_events:,.0f}"
    )
    print(
        f"Poisson-drawn NF events (seed={EVENT_COUNT_SEED}): "
        f"{target_event_count:,}"
    )
    print(f"Time CDF endpoint: {time_cdf[-1]:.6f}")

    # Step 4: Validate the interface with a small sample before full generation.
    generate_test_background(background, spacecraft_history, time_cdf)

    # Step 5: Generate or resume the complete three-month event file.
    generate_full_background(
        background,
        spacecraft_history,
        time_cdf,
        target_event_count,
    )


if __name__ == "__main__":
    main()
