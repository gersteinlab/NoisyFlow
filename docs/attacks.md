# Membership Inference Attacks

All attacks are implemented in `noisyflow/attacks/membership_inference.py` and are configured in the YAML files.

## Loss-threshold attack (`membership_inference`)
- Runs on the final classifier trained in stage 3.
- Uses per-sample cross entropy losses for synthetic (member) and target test (nonmember) data.
- Learns a threshold that maximizes accuracy on the combined set.

Outputs:
- `attack_acc`
- `attack_auc`
- `attack_threshold`
- `attack_advantage`

## Shadow-model attack (`shadow_mia`)
- Trains shadow classifiers on data from the same builder.
- Extracts features (`loss`, `stats`, `probs`, or `logits`) from the shadow models.
- Trains an attack MLP and evaluates it on the target classifier.

Outputs:
- `shadow_attack_acc`
- `shadow_attack_auc`
- `shadow_attack_advantage`
- `shadow_feature_set`

Notes:
- Use `shadow_mia.data_overrides` to tweak the data parameters for shadow models.

## Stage MIA (`stage_mia`)
- Attacks the stage 1/2 models directly using per-example features.
- Uses flow matching loss, and optionally OT potential and transport norm.
- Requires a per-client holdout split (`holdout_fraction > 0`).

Outputs:
- `stage_mia_attack_acc`
- `stage_mia_attack_auc`
- `stage_mia_attack_advantage`
- `stage_mia_train_frac`

Notes:
- If `stage2.option` is `B`, OT features are not included.
- Use `include_ot_transport_norm` to add the transport norm feature when OT is used.

## Stage shadow MIA (`stage_shadow_mia`)
- Trains shadow stage models and an attack model on stage features.
- Uses a similar feature set to `stage_mia` but with shadow models.

Outputs:
- `stage_shadow_mia_acc`
- `stage_shadow_mia_auc`
- `stage_shadow_mia_advantage`

Notes:
- Requires `holdout_fraction > 0`.
- Use `stage_shadow_mia.data_overrides` to adjust shadow data parameters.
