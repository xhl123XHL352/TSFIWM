# TSFIWM controlled interactive text-state dataset

Honglei Xin, Beijing Institute of Technology, 2026.
Repository: https://github.com/xhl123XHL352/TSFIWM; version: v1.0.0.

This archive contains 24,030 PNG images, split annotations, simulator generation
code and asset notices. The forecast corpus contains 1,080 independent roots
and 21,600 wide-view images; the goal corpus contains 90 fresh roots. Additional
native panel views and matched initial-image crops provide specified
observation conditions. The exact file inventory and corpus counts are in
DATA-MANIFEST.json.

| Corpus | Split | Independent roots | Seed start |
|---|---|---:|---:|
| panel-v3 | train | 600 | 100000 |
| panel-v3 | dev | 120 | 200000 |
| panel-v3 | test | 180 | 300000 |
| panel-v3 | test_shift | 180 | 400000 |
| panel-control-v1 | control_dev | 30 | 900000 |
| panel-control-v1 | control_test | 30 | 1000000 |
| panel-control-v1 | control_shift | 30 | 1100000 |

## Observations, transitions and labels

Each forecast root contains four calibration images and four four-step action
branches from the same saved physical state. A PyBullet Panda presses buttons
on three instruments: a counter, an alternating update and a gated accumulator.
Numerical updates depend on contact and button depression. Requested commands,
executed outcomes and hidden audit parameters are recorded separately.

Every root contains inputs.json, labels.json and complete.json. Public inputs
provide observed filenames, instrument family, calibration requested commands
and future requested commands or goal entries. Labels contain the states,
digital modes, contacts and audit-only gains. complete.json stores generator
signatures and input/label hashes. Use the specified observations at prediction
time; future frames, future contacts, gain/mode labels and reference goal plans
serve evaluation and audit. All recorded forecast calibration presses succeed,
so gains can also be inferred from the public equations and observations.

All branches of a root share one split. Independent resampling uses whole
roots with family strata. The five-character identifier sets are disjoint
across forecast splits; goal roots use fresh seeds. Goal success compares the
three final readings with the true identifier-matched desired readings.

Native initial views in panel-initial-views-v1 are additional 384x384 camera
observations. Matched crops in panel-initial-crops-v1 reuse the first wide
image and are selected by panel geometry. File-level counts state the available
split coverage. The fixed camera, panel, robot and press primitive implement a
controlled diagnostic environment. The fonts are DejaVu Sans and DejaVu Sans
Mono. Ordinary initial readings lie in [15,70), shifted readings in [200,900)
with a different background scheme. Both successful and failed future presses
are retained. Images depict synthetic scenes and random instrument identifiers.

## Metrics and reproduction

CGE computes the geometric mean of changed-field and unchanged-field numerical
exactness within each family, then averages families. Numerical exactness uses
ties-to-even rounding followed by clipping to [0,999]. MAE uses unrounded
predictions. Strict root success requires every numerical forecast in all four
branches to be correct. Identifier exactness is a separate text-recognition
outcome. Legacy result keys named balanced_accuracy contain CGE.

The companion code/results archive provides the scorers, checkpoints and
per-example predictions. Generator dependencies are in requirements-forecast.txt
and requirements-goals.txt. Use Python 3.10 and PyBullet 3.2.7 in a project-local
environment. Forecast generation used NumPy 2.2.6/Pillow 12.1.0; goal rendering
used NumPy 1.26.4/Pillow 10.2.0. Physics uses a 1/240-second step, 100 solver
iterations and TinyRenderer. Starting from an empty output directory:

    python source/panel_data.py --splits train dev test test_shift
    python source/prepare_control_goals.py

Native forecast views use source/panel_initial_views.py with the selected
split. Goal views are produced by the goal generator. The matched-crop helper
operates on saved initial images with known geometry.

New Panel images and annotations: CC BY 4.0 (LICENSE-DATA.txt).
New generator code: MIT (LICENSE-CODE.txt).
Panda/DejaVu asset licenses are retained in third_party_notices/.
The manuscript's DoomShift audit uses its separately obtained upstream corpus.
