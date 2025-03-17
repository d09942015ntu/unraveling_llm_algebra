# Usage

## Install Requirements

```commandline
python3 -m venv venv_llm_math
source venv_llm_math/bin/activate
pip install -r requirement.txt
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
