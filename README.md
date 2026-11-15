# Text State Fidelity in Interactive World Models

**Honglei Xin, Beijing Institute of Technology**

This release accompanies *Text State Fidelity in Interactive World Models:
A Controlled Benchmark and Diagnostic Study*. It separates visual-text
observation, action-conditioned numerical state updates and decoder readout.

## Download

- [Full dataset: 24,030 PNG images and annotations](https://github.com/xhl123XHL352/TSFIWM/releases/download/v1.0.0/TSFIWM-dataset-v1.0.0.zip)
- [Code, 62 numerical-reference checkpoints and per-example results](https://github.com/xhl123XHL352/TSFIWM/releases/download/v1.0.1/TSFIWM-code-results-v1.0.1.zip)
- [SHA-256 checksums](https://github.com/xhl123XHL352/TSFIWM/releases/download/v1.0.1/SHA256SUMS.txt)

Extract the code/results archive into a new directory. For training or visual
inference, extract the dataset archive into that same directory. Saved-output
scoring uses the annotations already included in the code/results archive.

## Benchmark

The forecast corpus contains 1,080 independent roots and 21,600 wide-view
images. Each root includes four calibration observations and four matched
four-step action branches. The goal corpus adds 90 independent roots. Native
initial panel views and matched wide-image crops provide specified observation
controls. Random five-character identifiers bind each instrument to its reading.
The three instrument families are a counter, an alternating update and a gated
accumulator. Numerical updates depend on recorded physical button execution.

Forecast splits contain 600/120/180/180 roots for train/dev/test/test_shift.
Goal splits contain 30 roots each for control_dev/control_test/control_shift.
All branches stay within their root; identifiers and root seeds are disjoint
across the specified splits. Ordinary initial readings are in [15,70), shifted
initial readings in [200,900). The two rendering fonts are DejaVu Sans and
DejaVu Sans Mono. Camera, panel geometry, robot and press primitive are fixed.

The primary numerical metric is change-balanced geometric exactness (CGE).
Identifier exactness, strict numerical root success, goal success and character
error rate are reported separately. Confidence intervals resample independent
roots or Doom episodes with the recorded family/map strata and pairing.

## Reproduce saved-output scores

Use Python 3.10 and the versions in `environment-versions.json`. Numerical
scoring requires NumPy. Figure generation requires Matplotlib and Pillow;
Times New Roman is used when installed, with DejaVu Serif as the fallback.
Run these commands from the extracted archive root:

```bash
python source/analyze_results.py --split test --round reproduced --groups panel-annotation panel-parseq posthoc-rule-table --summary-only
python source/analyze_results.py --split test_shift --round reproduced --groups panel-annotation panel-parseq posthoc-rule-table --summary-only
python source/analyze_control.py --split control_test
python source/analyze_control.py --split control_shift
python source/analyze_doom.py --subset test_seen
python source/analyze_doom.py --subset test_unseen
python source/analyze_character_readout.py
python source/build_result_tables.py
python source/make_figures.py
```

`CODE-MANIFEST.json` hashes every included file. The character analysis contains
14,112 field records, 56 conditions and paired decoder comparisons. Generator
and model configuration files are in `source/`; protocol freezes and per-example
predictions are in `results/`. The included checkpoints are the study's 62 small
numerical references. Native DoomShift inference uses the pinned full U-Net or
PixArt checkpoint and the corresponding stock/tuned decoder.

## Training and visual inference

Use a project-local environment and cache. Forecast generation used PyBullet
3.2.7, NumPy 2.2.6 and Pillow 12.1.0; goal rendering and physical control used
NumPy 1.26.4 and Pillow 10.2.0. Numerical references used PyTorch 2.7.1+cu128.
The recorded environment maps and upstream revisions are supplied with the
release. Set `TEXTSTATE_GPU_DEVICES` to the permitted device identifier and
pass the same identifier through `--gpu-uuid`. The exported resource wrapper
tracks allocated process time and retains its recorded per-workspace defaults.

PARSeq code belongs in `cache/repos/parseq`, DoomShift in `cache/repos/doomshift`.
Qwen and InternVL model directories are `models/qwen3.5-9b` and
`models/internvl3.5-8b`. Retrieve upstream weights and data from their publishers
at the pinned revisions in the configurations and notices.

## License and attribution

New code: [MIT](LICENSE). New Panel data, annotations and study-produced
numerical results: [CC BY 4.0](LICENSE-DATA.txt). Upstream assets retain their
own terms; see [third-party notices](THIRD-PARTY-NOTICES.txt). Cite this
repository and version when using the benchmark or retained results.
