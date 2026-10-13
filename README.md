# cosipy-background-nf
Development repo for binned COSI background models using Pascal Janowski’s normalizing-flow background PDF

## Current validation status

The standalone environment uses `cosipy` from Pascal's
`scipascal_area-renormalization` branch (commit `2cc83ad5`). The optional
machine-learning dependencies were installed with `cosipy[ml]`.

Verified so far:

- `NFBackground` imports from `cosipy.background_estimation.ml`.
- `model/nfbackground_v2-00.pt` has the required checkpoint metadata:
  `version`, `density_input`, and `rate_input`.
- The checkpoint loads successfully with `NFBackground` on CPU and reports
  model version `2.00`.
- The public model methods include `evaluate_density`, `evaluate_rate`, and
  `sample_density`.

Minimal loading example:

```python
from cosipy.background_estimation.ml import NFBackground

background = NFBackground("model/nfbackground_v2-00.pt", devices=["cpu"])
```

This only confirms that the trained model can be loaded. Density evaluation,
sampling, comparison with the simulated DC4 background, and conversion to a
binned COSI background model have not yet been validated.

Python 3.14 currently produces a PyTorch `FutureWarning` about
`torch.jit.script`; it did not prevent the model from loading.
