# Escaping Optimization Stagnation: Taking Steps Beyond Task Arithmetic via Difference Vectors


> AAAI 2026 (Oral)


## Method Overview

```
Pretrained θ₀  ──┐
                  ├──► Δ = θ_opt − θ₀   (difference vector)
      optimum ───┘
        │
        ▼
  DV-BASI iteration:  optimize α over  θ_opt + α·Δ·τᵢ  (task vectors τᵢ)
        │
        ▼
  new_opt → repeat until convergence
```

Each outer iteration recomputes the difference vector from the latest optimum and re-runs the inner coefficient optimisation (aTLAS-style), enabling the model to move beyond flat/stagnant loss landscapes.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate DVBASI
```

---

## Data Preparation

See [DATASETS.md](DATASETS.md) for full setup instructions. Default data root is `~/data/` (override with `--data-location`).

---

## Checkpoint Directory Layout

All scripts expect checkpoints under a root directory (pass via `--save`). After fine-tuning each task, the layout should be:

```
checkpoints/{model}/
├── zeroshot_accuracies.json
├── ft_accuracies.json
├── CarsVal/
│   ├── zeroshot.pt
│   └── finetuned.pt
├── DTDVal/
│   ├── zeroshot.pt
│   └── finetuned.pt
├── ...
└── combined_8/
    └── addition_baseline.pt      # output of task_addition_atlas.py (aTLAS stage)
```

For linearized fine-tuning, replace `zeroshot.pt`/`finetuned.pt` with `linear_zeroshot.pt`/`linear_finetuned.pt`, and `ft_accuracies.json` with `linear_ft_accuracies.json`.

---

## Usage

### Step 1 — DV-BASI: iterative difference-vector optimisation

`task_addition_DVBASI.py` runs the outer DV-BASI loop, starting from the aTLAS baseline. Each iteration constructs a new difference vector and re-optimises the coefficients.


**Auto-run aTLAS before DV-BASI (recommended):**

```bash
python src/task_addition_DVBASI.py \
    --model ViT-B-32 \
    --finetuning-mode standard \
    --dv-auto-atlas-force \
    --dv-max-iters 5 \
    --dv-auto-atlas \
    --dv-atlas-epochs 50 \
    --dv-patience 10 \
    --lr 0.01
```

**Linearized fine-tuning:

    --save checkpoints/ViT-B-32 \
    --finetuning-mode linear \
    --dv-auto-atlas-force \
    --dv-max-iters 5 \
    --dv-auto-atlas \
    --dv-atlas-epochs 50 \
    --dv-patience 10 \
    --lr 0.01
```

**Unsupervised mode** (uses softmax entropy instead of cross-entropy for both aTLAS and DV-BASI):

```bash
python src/task_addition_DVBASI.py \
    --model ViT-B-32 \
    --save checkpoints/ViT-B-32 \
    --finetuning-mode standard \
    --dv-max-iters 5 \
    --dv-auto-atlas \
    --dv-atlas-epochs 50 \
    --dv-patience 10 \
    --lr 0.01
    --unsupervised
```


## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wang2026escaping,
  title={Escaping Optimization Stagnation: Taking Steps Beyond Task Arithmetic via Difference Vectors},
  author={Wang, Jinping and Gao, Zhiqiang and Dinggen, Zhang and Xie, Zhiwu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={31},
  pages={26310--26318},
  year={2026}
}
```

---

## Acknowledgements

This codebase builds on [Task Vectors](https://github.com/mlfoundations/task_vectors) and [aTLAS](https://github.com/fredzzhang/atlas). Dataset setup scripts are adapted from those repositories.


