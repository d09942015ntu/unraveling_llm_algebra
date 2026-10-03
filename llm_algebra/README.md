## Usage

Earlier version of the experiments in the main folder. Datasets use the `ide_41_<m>_<pn>` format
(operators `+` and `-`, comma-separated CSV).

```sh
python3 dataset_generator.py  # generate the datasets in ./data
python3 trainer.py            # pre-train the model (expects GPT-2 files in ./models/gpt2)
python3 evaluator.py          # evaluate the latest checkpoint
```

Other files:

- `mymodels/`: a toy transformer (`toytrans`) written as a Hugging Face custom model; unfinished.
- `mymodel.py`: a single attention block, run directly for a toy example.
- `sort_outputs.py`, `dynamics.py`: small helpers used to format results for the paper.
