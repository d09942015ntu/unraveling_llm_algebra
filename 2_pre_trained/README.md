# Scenario of Existing Pre-Trained Model 


## Dataset 

Dataset is modified from
1. BIGGSM: https://huggingface.co/datasets/LightChen2333/BigGSM
2. MAWPS: https://huggingface.co/datasets/garrethlee/MAWPS

## Knowledge Types:

### Commutativity & Identity
- 1_full: Original Equation
- 2_com: Commutativity
- 3_ide: Identity 
- 11_fullN: Original Equation & Noise
- 21_comN: Commutativity & Noise
- 31_ideN: Identity & Noise

### Associativity
- 1_fw: Forward Merging
- 2_bw: Backward Merging
- 3_full: One-Step Equation 
- 11_fwN: Forward Merging & Noise
- 21_bwN: Backward Merging & Noise
- 31_fullN: One-Step Equation & Noise


## Run Inference of Pre-trained LLMs


### Commutativity & Identity

```sh
bash run_exp1_api.sh 
```

### Associativity

```sh
bash run_exp2_api.sh 
```


## Run Evaluation of the Inference Result


### Commutativity & Identity

```sh
bash run_exp1_eval.sh 
```

### Associativity

```sh
bash run_exp2_eval.sh 
```
