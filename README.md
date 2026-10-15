# CE-MoE

## Quick start

Python 3.11 was tested. Run these commands from this directory:

```bash
python -m pip install -r requirements.txt
python test.py
```

The supplied `checkpoint/best.pt` is already trained and can be used immediately.

## Files

- `data/dataset.xlsx`: complete dataset(22,181 rows).
- `cemoe/`: data processing, three experts, CAV fusion, SCF/DRM/CWC and training helpers.
- `train.py`: train CE-MoE; select epochs and seeds using validation RMSE only.
- `test.py`: evaluate saved checkpoint, without retraining.
- `checkpoint/best.pt`: weights, preprocessing, configuration, seed and training history.
- `checkpoint/best.json`, `selection.csv`: readable training and selection information.

## Train a new checkpoint

```bash
python train.py --out checkpoint/my_run
python train.py --quick --seeds 11 --out checkpoint/quick_run
python train.py --full --out checkpoint/full_run
python test.py --checkpoint checkpoint/full_run/best.pt --all-test
```

Use a new output directory to avoid overwriting the provided checkpoint. `--epochs N`
and `--seeds 11 23 ...` set explicit budgets. No test metric is used for selection.
The offline validation-best state and evolved candidate are compared on validation;
the better state is saved.
