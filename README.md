# [Implementation] Unraveling Arithmetic in Large Language Models: The Role of Algebraic Structures

[https://openreview.net/pdf?id=aNmbQ4kGSQ](https://openreview.net/pdf?id=aNmbQ4kGSQ)

## Install Requirements

```commandline
python3 -m venv venv_llm_math
source venv_llm_math/bin/activate
pip install -r requirements.txt
```


## Quick Run
Quickly go through everything to check whether the installation is correct

```sh
bash run_quick.sh 
```


## Dataset generation

```sh
python3 dataset_generator.py
```

Datasets are written to `data/all_64_<n>_<K>/` (and small ones to `data/quick_64_<n>_<K>/`), where
`n` is the modulus and `K` the number of training samples. Each dataset type has a
`train_<type>.csv` and a `test_<type>.csv`:

- com: commutativity of operator $+$ (train on one ordering, test on other orderings)
- ide: identity of operator $+$ (test on inputs with an extra $0$)
- comx: commutativity of the random operator $\oplus$
- idex: identity of the random operator $\oplus$
- lh: operator $\triangleleft$ (returns the leftmost operand)
- rh: operator $\triangleright$ (returns the rightmost operand)
- z0: operator $\ominus$ (counts descents; depends on the order)


### Run Training

```sh
bash run_trainer.sh
```

Each run writes `results/<dataset>_seqnew-<time>/trainer.log` with the accuracy of every
dataset type at every logging step. To evaluate a checkpoint afterwards:

```sh
python3 evaluator.py --ckpt_path=results/<run>/checkpoints --dataset_path=data/all_64_7_1000 --dataset_type=com+ide
```


### Plot results

```sh
python3 vis_plot_test_acc.py
python3 vis_plot_convergence.py
```

### Visualize hidden states
```sh
python3 vis_com_std.py
python3 vis_ide_std.py
python3 vis_std_to_latex.py
```

## Inverse and Distributive

```sh
python3 dataset_generator_dist.py
python3 dataset_generator_inv.py
```

```sh
bash run_trainer_dist.sh
bash run_trainer_inv.sh
```

Distributivity inputs are up to 24 tokens long, so `run_trainer_dist.sh` passes `--max_length=32`.

## Code layout

| File | Purpose |
|---|---|
| `dataset_generator*.py`, `datagen_common.py` | Generate the datasets |
| `mydataset.py` | PyTorch datasets that read the CSV files |
| `trainer.py`, `evaluator.py`, `utils.py` | Training and evaluation |
| `vis_plot_*.py`, `results_log.py` | Accuracy plots (TikZ) from the training logs |
| `vis_com_std.py`, `vis_ide_std.py`, `vis_hidden_common.py`, `vis_std_to_latex.py` | Hidden-state analysis |


## Integrated sub-projects

This repository also contains two related projects (merged with `git subtree`, so their full commit history is kept):

| Folder | Original repository | Content |
|---|---|---|
| [`llm_algebra/`](llm_algebra) | `d09942015ntu/llm_algebra` | Earlier version of the algebraic-structure experiments, with a custom toy transformer (`mymodels/`, unfinished). |
| [`llm_math_boundary/`](llm_math_boundary) | `d09942015ntu/llm_math_boundary` | Experiments on the reasoning boundary of LLMs for math word problems (BigGSM, MAWPS), calling LLM APIs and evaluating the results. |

Each sub-project keeps its own README and requirements. Run its scripts from inside its folder, for example:

```sh
cd llm_math_boundary
pip install -r requirements.txt
bash run_1_my_api.sh
```
