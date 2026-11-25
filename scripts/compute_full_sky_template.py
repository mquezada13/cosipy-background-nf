"""Full-sky binned background template of one NF component (--part): FINAL settings, all fixes on.

Run from the project root (takes ~6.5 h per part; resumes if interrupted):

    .venv/bin/python scripts/compute_full_sky_template.py --part 2
    .venv/bin/python scripts/compute_full_sky_template.py --part 2 --max-pixels 2   # quick test run

This is the full-sky version of the last test of notebook 09 (Section 11.2):
``NFBinnedTemplate`` with its default settings (chi = 0 strip and pole
corrections on), the FINAL Em / Phi quadrature, sub-pixel level k = 2, the DC4
511 keV binning, and the DC4 time window.

Inputs and settings
-------------------
All inputs are the constants in the "Settings" block below; they are also
written into the output files. --part chooses the model file
(model/nfbackground_part-0X_v3-00.fits); the other command-line options only
choose how the run is done, not what is computed.

The part 01 template in data/NF_background/template/ was computed with the
earlier Phi breaks [15.5, 16, 16.5, 17] (notebook 09). Parts 02-04 use the
breaks below, validated in notebook 13; with them part 01 changes by <= 0.13%
in the bin 90-93 deg (notebook 13, Section 6).

Output (default: data/NF_background/template/), X = part number
------
1. part-0X_FINAL_k2_P.hdf5 -- written while running, one save per step:
       pixels               (3072,)        pixel indices (RING, NSIDE 16), in computing order
       P                    (1, 60, 3072)  P_i, NF density integrated per bin (dimensionless; NaN = not done)
       subpixel_level_used  (3072,)        level each pixel really got (2; 3 near the poles; 5 + refinement at the poles)
       done                 (3072,)        True once the pixel is stored
   plus all settings and model_counts, model_livetime as attributes.
   Indexed by pixel number: P[:, :, p] is pixel p.
2. When every pixel is done, the expected counts in [TMIN, TMAX] as histpy
   Histograms (axes Em, Phi, PsiChi), in the same way as
   ``NFBinnedTemplate.get_binned_background``:
       part-0X_FINAL_k2_511keV_DC4binning_local.hdf5      spacecraft frame
       part-0X_FINAL_k2_511keV_DC4binning_galactic.hdf5   Galactic frame
   N_i = R * T * P_i, R = model_counts / model_livetime, T = live time from the .ori file.

If file 1 exists, the run resumes with the pixels not done yet, provided all
settings in the file match; otherwise it stops without changing the file.
Existing count histograms are never overwritten.

Cost: ~2,208 evaluations per sky node (8 Em x 276 Phi nodes). Ordinary pixels
16 sky nodes, near-pole pixels 64, polar pixels ~1,850 (~13 min each, done last,
one per step). About 6.5 h per part at ~6,500 evaluations/s.
"""

import argparse
import sys
import time
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import torch
from astropy.time import Time
from histpy import Axes, Axis, HealpixAxis, Histogram
from scoords import SpacecraftFrame
from cosipy.spacecraftfile import SpacecraftHistory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))            # so that "src.nf_template" is found
from src.nf_template import NFBinnedTemplate, galactic_conversion_matrix, near_pole   # noqa: E402

# ---------------------------------------------------------------------------
# Settings (the inputs of the template)
# ---------------------------------------------------------------------------

# NF model of one component (version 3.00; the part is chosen with --part) and orientation file
MODEL_PATH_PATTERN = "model/nfbackground_part-{part:02d}_v3-00.fits"
ORI_PATH = PROJECT_ROOT / "data/DC4_orientation.ori"

# Time window of the analysis [unix s] (as in notebook 09)
TMIN, TMAX = 1835487300.036206, 1843467254.887687

# DC4 511 keV imaging binning (dataio_511.yaml; checked against the notebook 08 histograms in notebook 09)
EM_EDGES_KEV = [509.0, 513.0]              # 1 energy bin
PHI_EDGES_DEG = np.arange(0, 181, 3)       # 60 bins of 3 deg
NSIDE, SCHEME = 16, "ring"                 # PsiChi: 3072 pixels

# Sky: sub-pixel level k (4**k sky nodes per ordinary pixel; more near the poles, see TEMPLATE_SETTINGS)
SUBPIXEL_LEVEL = 2

# Em and Phi quadrature: FINAL of notebook 09 (Sections 5-6), Phi breaks extended in notebook 13 (Section 6.2):
# 4 GL points on each side of 511 keV, 4 per Phi bin; bin 15-18 deg split at the edge near 16 deg,
# bin 87-90 deg split at 89.5 deg and bin 90-93 deg split at the peak just above 90 deg (parts 03, 04)
FINAL = dict(n_em_per_bin=4, n_phi_per_bin=4, em_breaks_kev=[511.0],
             phi_breaks_deg=[15.5, 16.0, 16.5, 17.0, 89.5, 90.5, 91.0, 91.5, 92.0])

# Template settings: the defaults of NFBinnedTemplate (all fixes on), written out so they are visible and saved
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

# Attitude grouping for the Galactic rotation (default of get_binned_background)
SCATT_NSIDE = 32

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/NF_background/template"


def settings_as_attrs(model_path):
    """All settings as HDF5 attributes (to save, and to compare when resuming)."""
    attrs = {
        "model_file": model_path.name,
        "ori_file": ORI_PATH.name,
        "tmin": TMIN,
        "tmax": TMAX,
        "em_edges_kev": np.asarray(EM_EDGES_KEV, dtype=float),
        "phi_edges_deg": np.asarray(PHI_EDGES_DEG, dtype=float),
        "nside": NSIDE,
        "scheme": SCHEME,
        "subpixel_level": SUBPIXEL_LEVEL,
        "n_em_per_bin": FINAL["n_em_per_bin"],
        "n_phi_per_bin": FINAL["n_phi_per_bin"],
        "em_breaks_kev": np.asarray(FINAL["em_breaks_kev"], dtype=float),
        "phi_breaks_deg": np.asarray(FINAL["phi_breaks_deg"], dtype=float),
        "scatt_nside": SCATT_NSIDE,
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
    parser.add_argument("--part", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="NF component: model/nfbackground_part-0X_v3-00.fits.")
    parser.add_argument("--threads", type=int, default=4, help="Torch threads (4 measured fastest).")
    parser.add_argument("--pixels-per-step", type=int, default=48, help="Ordinary pixels computed between two saves.")
    parser.add_argument("--max-pixels", type=int, default=None, help="Only the first n pixels (test; no count histograms).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    model_path = PROJECT_ROOT / MODEL_PATH_PATTERN.format(part=args.part)
    for path in (model_path, ORI_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(args.threads)

    # Binning as histpy Axes with the labels NFBinnedTemplate expects
    axes = Axes([Axis(np.asarray(EM_EDGES_KEV, dtype=float) * u.keV, label="Em"),
                 Axis(np.asarray(PHI_EDGES_DEG, dtype=float) * u.deg, label="Phi"),
                 HealpixAxis(nside=NSIDE, scheme=SCHEME, label="PsiChi")])

    template = NFBinnedTemplate(model_path, **TEMPLATE_SETTINGS)
    print(f"model {model_path.name}, version {template.model_version}, model_counts {template.model_counts:,.0f}")

    # Computing order: ordinary and near-pole pixels first, the 8 slow polar pixels last
    npix = 12 * NSIDE**2
    nest = SCHEME == "nested"
    all_pixels = np.arange(npix)
    is_polar = near_pole(NSIDE, all_pixels, nest, template.polar_cap_rad)
    order = np.concatenate([all_pixels[~is_polar], all_pixels[is_polar]])
    if args.max_pixels is not None:
        order = order[:args.max_pixels]

    name = f"part-{args.part:02d}_FINAL_k2" + (f"_test{args.max_pixels}" if args.max_pixels is not None else "")
    p_path = args.output_dir / f"{name}_P.hdf5"
    p_path.parent.mkdir(parents=True, exist_ok=True)
    wanted = settings_as_attrs(model_path)
    n_em, n_phi = len(EM_EDGES_KEV) - 1, len(PHI_EDGES_DEG) - 1

    # Create the file, or check it before resuming
    if p_path.exists():
        with h5py.File(p_path, "r") as f:
            if not same_attrs(dict(f.attrs), wanted) or not np.array_equal(f["pixels"][:], order):
                raise RuntimeError(f"{p_path} exists with different settings or pixels; not changed.")
            done = f["done"][:]
        print(f"Resuming {p_path}: {done[order].sum()} of {len(order)} pixels already done")
    else:
        with h5py.File(p_path, "x") as f:
            f.attrs.update(wanted)
            f.attrs["model_version"] = template.model_version
            f.attrs["model_counts"] = template.model_counts
            f.attrs["model_livetime"] = template.model_livetime
            f.attrs["description"] = ("P_i: NF density integrated per CDS bin (dimensionless), spacecraft frame, "
                                      "indexed by pixel; expected counts in the model livetime = model_counts * P")
            f.create_dataset("pixels", data=order)
            f.create_dataset("P", data=np.full((n_em, n_phi, npix), np.nan))
            f.create_dataset("subpixel_level_used", data=np.full(npix, -1, dtype=np.int64))
            f.create_dataset("done", data=np.zeros(npix, dtype=bool))
        done = np.zeros(npix, dtype=bool)
        print(f"Writing {p_path}")

    # Steps: ordinary pixels in groups of --pixels-per-step, polar pixels one per step (~12 min each)
    todo = order[~done[order]]
    todo_polar = is_polar[todo]
    steps = [todo[~todo_polar][i:i + args.pixels_per_step]
             for i in range(0, (~todo_polar).sum(), args.pixels_per_step)]
    steps += [np.array([p]) for p in todo[todo_polar]]

    cost = [template.n_evaluations(axes, step, SUBPIXEL_LEVEL, **FINAL) for step in steps]
    print(f"{len(todo)} pixels to do in {len(steps)} steps, {sum(cost):,} evaluations")

    start_time = time.perf_counter()
    for i, step_pixels in enumerate(steps):
        # The integral of these pixels
        P = template.compute(axes, step_pixels, subpixel_level=SUBPIXEL_LEVEL, verbose=False, **FINAL)

        # Save this step; "done" last, so a crash mid-write leaves those pixels to redo
        with h5py.File(p_path, "r+") as f:
            for j, p in enumerate(step_pixels):          # one pixel at a time (h5py needs sorted indices otherwise)
                f["P"][:, :, p] = P[:, :, j]
                f["subpixel_level_used"][p] = template.subpixel_level_used[p]
            done_all = f["done"][:]
            done_all[step_pixels] = True
            f["done"][:] = done_all

        elapsed = time.perf_counter() - start_time
        evaluations_done = sum(cost[:i + 1])
        left = elapsed / evaluations_done * (sum(cost) - evaluations_done)
        print(f"step {i + 1}/{len(steps)}: {done_all[order].sum()}/{len(order)} pixels, {elapsed / 60:.1f} min, "
              f"{evaluations_done / elapsed:,.0f} evaluations/s, ~{left / 60:.0f} min left", flush=True)

    if args.max_pixels is not None:
        print("Test run finished (no count histograms).")
        return

    # --- All pixels done: expected counts in [TMIN, TMAX], as in get_binned_background ---
    with h5py.File(p_path, "r") as f:
        if not f["done"][:].all():
            raise RuntimeError("Not all pixels are done.")
        P_all = f["P"][:]

    history = SpacecraftHistory.open(ORI_PATH).select_interval(Time(TMIN, format="unix"), Time(TMAX, format="unix"))
    rate = template.model_counts / template.model_livetime                  # events / s
    livetime = float(history.livetime.to_value(u.s).sum())                  # live time in the window [s]

    def counts_axes(coordsys):
        return Axes([Axis(np.asarray(EM_EDGES_KEV, dtype=float) * u.keV, label="Em"),
                     Axis(np.asarray(PHI_EDGES_DEG, dtype=float) * u.deg, label="Phi"),
                     HealpixAxis(nside=NSIDE, scheme=SCHEME, coordsys=coordsys, label="PsiChi")])

    # Local frame: N = R * T * P
    local = Histogram(counts_axes(SpacecraftFrame()), contents=rate * livetime * P_all)
    local_path = args.output_dir / f"{name}_511keV_DC4binning_local.hdf5"
    local.write(local_path, overwrite=False)
    print(f"Wrote {local_path}: total {float(local.contents.sum()):,.1f} counts, live time {livetime:,.1f} s")

    # Galactic frame: rotate with W (live time each Galactic pixel spends in each local pixel)
    W = galactic_conversion_matrix(history, NSIDE, nest, SCATT_NSIDE)
    galactic = Histogram(counts_axes("galactic"), contents=rate * (P_all @ W.T))
    galactic_path = args.output_dir / f"{name}_511keV_DC4binning_galactic.hdf5"
    galactic.write(galactic_path, overwrite=False)
    print(f"Wrote {galactic_path}: total {float(galactic.contents.sum()):,.1f} counts")
    print("Finished.")


if __name__ == "__main__":
    main()
