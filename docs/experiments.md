# Experiments and CLI

## Common commands
Run the default experiment:
```bash
python run.py --config configs/default.yaml
```

Run the small smoke test:
```bash
python run.py --config configs/quick_smoke.yaml
```
Note: `configs/quick_smoke.yaml` defaults to `device: cuda`. Switch to `cpu` if you do not have a GPU.

Run the DP config on GPU:
```bash
python run.py --config configs/dp_gpu.yaml
```

Run the option A OT config:
```bash
python run.py --config configs/ot_option_a.yaml
```

Run the stage-level MIA demo:
```bash
python run.py --config configs/stage_mia_demo.yaml
```

Run the stage shadow MIA demo:
```bash
python run.py --config configs/stage_shadow_mia_demo.yaml
```

## CellOT lupuspatients (Kang)
Install dataset I/O deps:
```bash
python -m pip install anndata h5py
```

Download the preprocessed dataset ZIP and extract the lupuspatients subset:
```bash
python scripts/fetch_cellot_datasets.py --dataset lupuspatients
```

Run an end-to-end NoisyFlow experiment (ctrl→stim transport; stimulated cell-type classification on OOD holdout patient `101`):
```bash
python run.py --config configs/cellot_lupus_kang_smoke.yaml
python run.py --config configs/cellot_lupus_kang_rectifiedflow_ref50.yaml
```

Tuned config (more clients, longer training, more synthesis) with an ``acceptable'' labeled target budget:
```bash
python run.py --config configs/cellot_lupus_kang_rectifiedflow_ref50_tuned.yaml
```

Privacy-utility curve (plots `acc_ref_plus_synth` vs ε for a stage-1-only DP pipeline; OT is post-processing):
```bash
python run.py --config configs/cellot_lupus_kang_privacy_curve_ref50_stage1only.yaml
```

## Toy demo
A compact demo exists in `noisyflow/demo.py` and is exposed by `noisyflow_sketch.py`:
```bash
python noisyflow_sketch.py
```

## Dependencies
Required:
- `torch`
- `numpy`
- `pyyaml` (required for config loading)

Optional:
- `opacus` (required for DP-SGD)
- `matplotlib` (required for privacy curve plots)

## Tests
```bash
python -m unittest discover -s tests
```
