# Phase space anatomy of dynamical quantum phase transitions
### Zakaria Mzaouali
#### Institut für Theoretische Physik, Eberhard Karls Universität Tübingen, Auf der Morgenstelle 14, 72076 Tübingen, Germany.
#### Jülich Supercomputing Centre, Forschungszentrum Jülich GmbH, 52425 Jülich, Germany

Code and numerical data for independently reproducing the paper and its supplementary results. The calculations compare local dynamical order with global return competition in qutrit Potts and collective Ising systems, and resolve return rates into unsigned Wigner support and interference costs.

## Install and verify

Use Python 3.10 or newer and run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
make verify-data
make test
make figures
```

`make figures` rebuilds the three main figures, three supplementary figures, and three supplementary tables from the supplied data. Figures are saved as PNG and PDF under `figures/main/` and `figures/supplement/`; tables and analysis outputs go to `build/reproduction/tables/`. These commands require no LaTeX installation. Plot manifests identify their data inputs and selected font.

To match the recorded Python 3.12.7 environment, install [requirements-lock.txt](requirements-lock.txt) before installing the package with `python -m pip install --no-deps -e .`.

## Code and data

| Path | Contents |
| --- | --- |
| `src/dqpt_wigner/` | Hamiltonians, exact and MPS evolution, Wigner transforms, support contractions, and crossing analysis |
| `configs/` | Physical parameters, numerical precision settings, and convergence studies |
| `tests/` | Exact small-system comparisons, conservation laws, support identities, and GHZ marginal checks |
| `results/potts/finite_chain/` | Global finite-size scaling, sampled central-block trajectories, and TEBD convergence |
| `results/potts/blocks/` | Direct grouped block crossings through length 16, with step and bond refinements |
| `results/potts/events/` | Independently contracted block events and common-time primitive quantities |
| `results/potts/infinite_chain/` | Infinite-MPS signed crossings, checkpoints, and convergence studies |
| `results/ising/` | Quantum return branches at N = 400, h/J = 0.7, and the classical field scan |
| `results/supplement/` | Compact numerical inputs for supplementary figures and tables |
| `reproducibility/data_manifest.sha256` | SHA-256 checksums for the supplied data and configurations |

The figure commands can also be run individually:

```bash
python scripts/plot_main.py
python scripts/plot_supplement.py
python scripts/summarize_blocks.py
```

## Recompute the numerical results

The following commands generate fresh results under `build/reproduction/`. The finite-chain configuration includes the reported time-step and bond checks. The infinite-chain validation configuration varies the step, bond cap, and cutoff independently.

```bash
python scripts/run_potts_finite.py --config configs/potts_finite.yaml
python scripts/run_ising.py --config configs/ising.yaml
python scripts/run_product_control.py

python scripts/run_potts_blocks.py --out build/reproduction/potts/blocks/production
python scripts/run_potts_blocks.py --sizes 50 --lengths 16 --dt .01 \
  --workspace-mib 512 --jobs 1 --out build/reproduction/potts/blocks/refinement_step
python scripts/run_potts_blocks.py --sizes 50 --lengths 16 --max-bond 48 --cutoff 1e-10 \
  --workspace-mib 768 --jobs 1 --out build/reproduction/potts/blocks/refinement_bond

python scripts/run_block_events.py \
  --input build/reproduction/potts/finite_chain/potts_block_wigner_long.csv
python scripts/run_block_events.py --sizes 50 --dt .01 --skip-unsigned \
  --input build/reproduction/potts/finite_chain/potts_block_wigner_long.csv \
  --output-dir build/reproduction/potts/events_dt001

python scripts/run_potts_infinite.py --config configs/potts_infinite_production.json
python scripts/run_potts_infinite.py --config configs/potts_infinite_validation.json
```

Derive the supplementary inputs and rebuild all figures and tables from these fresh calculations:

```bash
python scripts/derive_supplement.py --results-root build/reproduction
python scripts/plot_main.py --results-root build/reproduction
python scripts/plot_supplement.py --results-root build/reproduction
python scripts/summarize_blocks.py --results-root build/reproduction
```

A full run can be computationally demanding. Each length-16 return support contains `3^16 = 43,046,721` Wigner cells; split contractions bound working memory while retaining every cell. `--workspace-mib` sets the block-contraction workspace and `--jobs` controls concurrent finite-chain work. For an installation check, the finite-Potts and Ising runners support `--smoke`. Infinite-chain campaigns support `--run-id`, `--max-seconds`, and `--resume` for individual or interrupted calculations.

## Mathematical checks and interpretation

For a support of length `ell`, the code independently contracts its return probability `P`, unsigned Wigner weight `A`, and negative mass `nu`. Where the logarithms are defined,

```text
A - P = 2 nu
r = -(1/ell) log P
s = -(1/ell) log A
q =  (1/ell) log(A/P)
r = s + q,     P/A = exp(-ell q).
```

For the grouped competitor, probabilities and unsigned weights are summed over sectors 1 and 2 before taking logarithms. Direct event contractions evaluate all quantities at the same crossing time. Tests compare exact state evolution with finite and infinite MPS calculations, verify independent projector and Wigner contractions, and check that GHZ states can have identical proper marginals but different global returns. The product-rotation control verifies a return cusp with a Wigner-positive critical state.

The measured Potts blocks establish a displacement between signed and unsigned branch exchanges. Infinite-MPS calculations test the signed thermodynamic crossing. Finite-block extrapolations do not establish an unsigned thermodynamic crossing or a thermodynamic displacement. In the Ising comparison, the field-driven late-time order diagnostic and the finite-system temporal return exchange probe different observables.

## License and citation

Code is distributed under the [MIT license](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff). To export the distributable repository files, use `make export-public OUTPUT=/tmp/dqpt-wigner-public` with an empty destination.

## Acknowledgments
Z.M. acknowledges funding from the Ministry of Economic Affairs, Labour and Tourism Baden-Württemberg in the frame of the Competence Center Quantum Computing Baden-Württemberg (project ``KQCBW25'').
