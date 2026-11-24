"""Full-sky convergence check of the sky integral: sub-pixel level k = 2 vs k = 3.

Run from the project root, for example:

    .venv/bin/python scripts/check_subpixel_convergence.py                 # whole sky, one process
    .venv/bin/python scripts/check_subpixel_convergence.py --max-pixels 4  # quick test run

What it computes
----------------
For every pixel whose sub-pixel level depends on k, the integral P_i of the NF
density of component 01 over each (Em, Phi) bin is computed twice with
``NFBinnedTemplate.compute``: once at k = K_LOW and once at k = K_HIGH. All
fixes are on (template defaults, listed in TEMPLATE_SETTINGS below).

Pixels whose level does not depend on k are skipped, because both runs would
give the same numbers: the polar pixels (fixed level polar_subpixel_level,
refined near the pole) and the pixels near a pole (at least
near_pole_subpixel_level = 3). At NSIDE 16 this leaves 3024 of 3072 pixels.

The Em and Phi integrals use the cheap settings QUAD (the same as in notebook
09), so the check isolates the sky integral. Estimated cost at ~7,500
evaluations/s: 3024 pixels x (3,840 + 15,360) evaluations ~ 2.2 h.

Inputs and settings
-------------------
All inputs are the constants in the "Settings" block below; they are also
written into the output file. Command-line options only choose which pixels
this process does and where it writes, not what is computed.

Output
------
One HDF5 file per block of pixels (default: data/NF_background/convergence/),
with datasets
    pixels          (n,)            pixel indices (RING scheme, NSIDE 16)
    P_k2, P_k3      (1, 60, n)      P_i at k = K_LOW and K_HIGH (dimensionless; NaN = not done)
    done            (n,)            True once both levels are stored for that pixel
and attributes with every setting above plus model_counts and model_livetime
(expected counts in the model's livetime: N_i = model_counts * P_i).

Results are written every --pixels-per-step pixels. If the file exists, the run
resumes from the pixels not done yet, provided all settings in the file match;
otherwise it stops without changing the file.

Parallel runs: split the pixels into --n-blocks blocks and start one process
per block (--block 0, 1, ...), each with its own output file. The machine has
4 performance cores; whether e.g. 2 processes x 2 threads beat 1 x 4 is not
measured.
"""

import argparse
import sys
import time
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import torch
from histpy import Axes, Axis, HealpixAxis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # so that "src.nf_template" is found
from src.nf_template import NFBinnedTemplate   # noqa: E402

# ---------------------------------------------------------------------------
# Settings (the inputs of the check)
# ---------------------------------------------------------------------------

# NF model: component 01, version 3.00
MODEL_PATH = PROJECT_ROOT / "model/nfbackground_part-01_v3-00.fits"

# DC4 511 keV imaging binning (dataio_511.yaml; checked against the notebook 08 histograms in notebook 09)
EM_EDGES_KEV = [509.0, 513.0]              # 1 energy bin
PHI_EDGES_DEG = np.arange(0, 181, 3)       # 60 bins of 3 deg
NSIDE, SCHEME = 16, "ring"                 # PsiChi: 3072 pixels, spacecraft frame

# The two sky levels compared (4**k sub-pixels per ordinary pixel)
K_LOW, K_HIGH = 2, 3

# Em and Phi quadrature: cheap settings of notebook 09 (2 GL points on each side of 511 keV, 1 per Phi bin)
QUAD = dict(n_em_per_bin=2, n_phi_per_bin=1, em_breaks_kev=[511.0], phi_breaks_deg=[])

# Template settings: the defaults of NFBinnedTemplate, written out so they are visible and saved
TEMPLATE_SETTINGS = dict(
    batch_size=4096,               # points per forward pass (>= 2048 keeps results bit-identical)
    device="cpu",
    seam_delta_deg=0.2,            # chi = 0 strip
    polar_cap_deg=4.0,             # polar pixels: centre < 4 deg from a pole ...
    polar_subpixel_level=5,        # ... at level 5 ...
    pole_colat_deg=1.0,            # ... and sub-pixels within 1 deg of the pole ...
    pole_subpixel_level=7,         # ... at level 7
    near_pole_cap_deg=10.0,        # pixels with centre < 10 deg from a pole ...
    near_pole_subpixel_level=3,    # ... at least level 3
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/NF_background/convergence"


def settings_as_attrs():
    """All settings of the check as HDF5 attributes (to save, and to compare when resuming)."""
    attrs = {
        "model_file": MODEL_PATH.name,
        "em_edges_kev": np.asarray(EM_EDGES_KEV, dtype=float),
        "phi_edges_deg": np.asarray(PHI_EDGES_DEG, dtype=float),
        "nside": NSIDE,
        "scheme": SCHEME,
        "k_low": K_LOW,
        "k_high": K_HIGH,
        "n_em_per_bin": QUAD["n_em_per_bin"],
        "n_phi_per_bin": QUAD["n_phi_per_bin"],
        "em_breaks_kev": np.asarray(QUAD["em_breaks_kev"], dtype=float),
        "phi_breaks_deg": np.asarray(QUAD["phi_breaks_deg"], dtype=float),
    }
    attrs.update({key: value for key, value in TEMPLATE_SETTINGS.items() if key not in ("batch_size", "device")})
    return attrs


def same_attrs(saved, wanted):
    """True if every wanted setting is in the file with the same value."""
    for key, value in wanted.items():
        if key not in saved:
            return False
        if isinstance(value, str):
            if str(saved[key]) != value:
                return False
        elif not np.array_equal(np.asarray(saved[key], dtype=float), np.asarray(value, dtype=float)):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", type=int, default=0, help="Block done by this process (0 ... n-blocks - 1).")
    parser.add_argument("--n-blocks", type=int, default=1, help="Number of blocks the pixels are split into.")
    parser.add_argument("--threads", type=int, default=4, help="Torch threads (4 measured fastest for 1 process).")
    parser.add_argument("--pixels-per-step", type=int, default=48, help="Pixels computed between two saves.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Only the first n pixels of the block (test).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--em-edges", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Other energy window [keV] (default: the 511 keV bin, 509 513).")
    parser.add_argument("--em-break", type=float, default=None,
                        help="Split point [keV] of that window (default 511; use the window centre elsewhere, "
                             "so the Em points are as dense as at 511 keV).")
    args = parser.parse_args()

    # Another energy window: same settings otherwise; the window goes into the file name and the attributes
    global EM_EDGES_KEV
    window = ""
    if args.em_edges is not None:
        if args.em_break is None or not args.em_edges[0] < args.em_break < args.em_edges[1]:
            raise ValueError("--em-edges needs --em-break strictly inside the window.")
        EM_EDGES_KEV = list(args.em_edges)
        QUAD["em_breaks_kev"] = [args.em_break]
        window = f"_{args.em_edges[0]:g}-{args.em_edges[1]:g}keV"

    if not 0 <= args.block < args.n_blocks:
        raise ValueError(f"--block must be in [0, {args.n_blocks - 1}], got {args.block}.")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    torch.set_num_threads(args.threads)

    # Binning as histpy Axes with the labels NFBinnedTemplate expects
    axes = Axes([Axis(np.asarray(EM_EDGES_KEV, dtype=float) * u.keV, label="Em"),
                 Axis(np.asarray(PHI_EDGES_DEG, dtype=float) * u.deg, label="Phi"),
                 HealpixAxis(nside=NSIDE, scheme=SCHEME, label="PsiChi")])

    template = NFBinnedTemplate(MODEL_PATH, **TEMPLATE_SETTINGS)
    print(f"model {MODEL_PATH.name}, version {template.model_version}, model_counts {template.model_counts:,.0f}")

    # Pixels whose level depends on k: the others give identical numbers at K_LOW and K_HIGH
    all_pixels = np.arange(12 * NSIDE**2)
    nest = SCHEME == "nested"
    level_low = template.subpixel_levels(NSIDE, all_pixels, nest, K_LOW)
    level_high = template.subpixel_levels(NSIDE, all_pixels, nest, K_HIGH)
    changing = all_pixels[level_low != level_high]
    print(f"{len(changing)} of {len(all_pixels)} pixels change level between k = {K_LOW} and k = {K_HIGH}")

    # This process's block (contiguous, nearly equal sizes)
    pixels = np.array_split(changing, args.n_blocks)[args.block]
    if args.max_pixels is not None:
        pixels = pixels[:args.max_pixels]
    n = len(pixels)

    name = f"part-01{window}_k{K_LOW}_vs_k{K_HIGH}_block{args.block}of{args.n_blocks}"
    if args.max_pixels is not None:
        name += f"_first{args.max_pixels}"
    output_path = args.output_dir / f"{name}.hdf5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wanted = settings_as_attrs()
    n_em, n_phi = len(EM_EDGES_KEV) - 1, len(PHI_EDGES_DEG) - 1

    # Create the file, or check it before resuming
    if output_path.exists():
        with h5py.File(output_path, "r") as f:
            if not same_attrs(dict(f.attrs), wanted) or not np.array_equal(f["pixels"][:], pixels):
                raise RuntimeError(f"{output_path} exists with different settings or pixels; not changed.")
            done = f["done"][:]
        print(f"Resuming {output_path}: {done.sum()} of {n} pixels already done")
    else:
        with h5py.File(output_path, "x") as f:
            f.attrs.update(wanted)
            f.attrs["model_version"] = template.model_version
            f.attrs["model_counts"] = template.model_counts
            f.attrs["model_livetime"] = template.model_livetime
            f.attrs["description"] = ("NF density integrated per CDS bin (P_i, dimensionless) at two sky "
                                      "sub-pixel levels, all fixes on; spacecraft frame")
            f.create_dataset("pixels", data=pixels)
            for key in (f"P_k{K_LOW}", f"P_k{K_HIGH}"):
                f.create_dataset(key, data=np.full((n_em, n_phi, n), np.nan))
            f.create_dataset("done", data=np.zeros(n, dtype=bool))
        done = np.zeros(n, dtype=bool)
        print(f"Writing {output_path}")

    todo = np.flatnonzero(~done)
    print(f"{len(todo)} pixels to do, "
          f"{template.n_evaluations(axes, pixels[todo], K_LOW, **QUAD) + template.n_evaluations(axes, pixels[todo], K_HIGH, **QUAD):,} evaluations")

    start_time = time.perf_counter()
    for step_start in range(0, len(todo), args.pixels_per_step):
        index = todo[step_start:step_start + args.pixels_per_step]   # positions in the output arrays
        step_pixels = pixels[index]

        # The integral at both levels (compute always recomputes the pixels it is given)
        P_low = template.compute(axes, step_pixels, subpixel_level=K_LOW, verbose=False, **QUAD)
        P_high = template.compute(axes, step_pixels, subpixel_level=K_HIGH, verbose=False, **QUAD)

        # Save this step; "done" last, so a crash mid-write leaves those pixels to redo
        with h5py.File(output_path, "r+") as f:
            f[f"P_k{K_LOW}"][:, :, index] = P_low
            f[f"P_k{K_HIGH}"][:, :, index] = P_high
            done_all = f["done"][:]
            done_all[index] = True
            f["done"][:] = done_all

        n_done = step_start + len(index)
        elapsed = time.perf_counter() - start_time
        print(f"{n_done}/{len(todo)} pixels, {elapsed / 60:.1f} min, "
              f"~{elapsed / n_done * (len(todo) - n_done) / 60:.0f} min left", flush=True)

    print("Finished.")


if __name__ == "__main__":
    main()
