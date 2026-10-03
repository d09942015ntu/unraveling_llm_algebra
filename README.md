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

### dataset types:
- com: commutativity for operator $+$
- ide: commutativity for operator $+$
- comx: commutativity for operator $\oplus$
- idex: commutativity for operator $\oplus$
- lh: commutativity for operator $\triangleleft$
- rh: commutativity for operator $\triangleright$
- z0: commutativity for operator $\ominus$
 
```sh
python3 dataset_generator.py 
```


### Run Training

```sh
bash run_trainer.sh 
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


## Integrated sub-projects

This repository also contains two related projects (merged with `git subtree`, so their full commit history is kept):

| Folder | Original repository | Content |
|---|---|---|
| [`llm_algebra/`](llm_algebra) | `d09942015ntu/llm_algebra` | Earlier version of the algebraic-structure experiments, with a custom toy transformer (`mymodels/`) and training-dynamics analysis (`dynamics.py`). |
| [`llm_math_boundary/`](llm_math_boundary) | `d09942015ntu/llm_math_boundary` | Experiments on the reasoning boundary of LLMs for math word problems (BigGSM, MAWPS), calling LLM APIs and evaluating the results. |

Each sub-project keeps its own README and requirements. Run its scripts from inside its folder, for example:

```sh
cd llm_math_boundary
pip install -r requirements.txt
bash run_1_my_api.sh
```
