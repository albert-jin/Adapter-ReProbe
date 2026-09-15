# Adapter-ReProbe

The compact, public-facing implementation snapshot is in
[`Ada-Reprobe-Code/`](Ada-Reprobe-Code/). It contains the core claim-level
alignment and calibration changes, lightweight experiment utilities, configs,
and regression tests, with model weights, datasets, caches, and outputs
excluded. The paper is currently under review at ICASSP 2027.

# Ada-ReProbe

Compact public code snapshot for **Adapter-ReProbe: Label-Efficient
Maintenance of Internal-State Process Verifiers after Language-Model
Fine-Tuning**. **Paper status: under review at ICASSP 2027.**

This release contains the implementation needed to inspect and reproduce the
method at source level, while intentionally excluding model weights, adapters,
datasets, hidden-state caches, generated samples, and experiment outputs.

## What is included

- `src/luh/alignment.py`: robust saved-reply/token alignment and claim-position
  mapping;
- `src/luh/feature_extractors/basic_hidden_states.py`: explicit current-token
  and previous-token alignment;
- `src/luh/heads/uncertainty_head_claim.py`: corrected causal masking, scalar
  calibration, and the diagnostic low-rank bridge;
- `src/train_luh/run_train_luh.py`: PEFT adapter loading, trainable-scope
  control, support subsampling, deterministic seeds, and AP/Brier/ECE export;
- `scripts/`: the lightweight calibration, transfer, natural Best-of-N,
  auditing, aggregation, and plotting utilities used for the paper;
- `configs/`: the two resolved training/evaluation configurations; and
- `tests/`: alignment, feature semantics, and research-metric regression tests.

## Provenance and scope

The snapshot is taken from the `research/adapter-reprobe` branch of the
modified ReProbe repository at commit
`d30c693928f16dbea9cfd6209cc5bd19218dd913`, with upstream reference
`fac0c16da74d55f71cc9fa469cc08e88c9604ee4`. It is a focused source release,
not a replacement for the complete upstream repository.

Although a separate project in the workspace extends TIME under the name
DeltaAlign, this repository is the Adapter-ReProbe implementation and is not
that TIME/DeltaAlign codebase. Keeping the provenance explicit avoids mixing
the two ICASSP projects.

## Setup and smoke tests

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests
```

The full training and evaluation scripts require a compatible backbone,
tokenizer, PEFT adapter/checkpoint, and task datasets supplied separately. Set
`PYTHONPATH=src:src/train_luh` when invoking the training entry point, for
example:

```bash
PYTHONPATH=src:src/train_luh python src/train_luh/run_train_luh.py \
  --help
```

Use the shell scripts in `scripts/` only after replacing their model, dataset,
cache, and output paths with local paths. They never assume that the excluded
server artifacts are part of this repository.

## Citation and redistribution

Please cite the Adapter-ReProbe paper once the author-approved citation is
available. The upstream ReProbe source did not contain an explicit license in
the snapshot used for this experiment; add the appropriate author-approved
license and attribution before treating this repository as a final public
software release.
