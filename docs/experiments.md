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
