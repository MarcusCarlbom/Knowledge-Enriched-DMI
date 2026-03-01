# Knowledge-Enriched Distributional Model Inversion Attacks — Extended with Defensive Features

This repository is based on the ICCV 2021 paper **Knowledge-Enriched Distributional Model Inversion Attacks** by Chen et al. It implements a GAN-based model inversion attack against face recognition classifiers, and has been extended with three inference-time privacy-preserving defensive mechanisms.

Original paper: [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Knowledge-Enriched_Distributional_Model_Inversion_Attacks_ICCV_2021_paper.pdf)  
Original code: [SCccc21/Knowledge-Enriched-DMI](https://github.com/SCccc21/Knowledge-Enriched-DMI)

---

## Setup

Tested on home gaming computer unning Python 3.12, PyTorch 2.10.0+cu128, CUDA 12.8, on an NVIDIA RTX 5070 Ti, running WSL2 on Windows 11 (Build 26100).
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy scikit-learn Pillow opencv-python pandas matplotlib
```

### Dataset

Download the CelebA aligned face images from the [official source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and extract into:
```
data/img_align_celeba_raw/img_align_celeba/
```

### Pretrained Models

Download target model checkpoints from the [original Google Drive](https://drive.google.com/drive/folders/1U4gekn72UX_n1pHdm9GQUQwwYVDvpTfN) and place them in `target_model/target_ckp/`:
```
target_model/target_ckp/VGG16_88.26.tar
target_model/target_ckp/IR152_91.16.tar
target_model/target_ckp/FaceNet64_88.50.tar
target_model/target_ckp/FaceNet_95.88.tar
```

Download pretrained GAN checkpoints from the [second Drive folder](https://drive.google.com/drive/folders/1L3frX-CE4j36pe5vVWuy3SgKGS9kkA70) and place them in `improvedGAN/`:
```
improvedGAN/improved_celeba_G.tar
improvedGAN/improved_celeba_D.tar
improvedGAN/celeba_G.tar
improvedGAN/celeba_D.tar
```

### Single GPU

If running on a single GPU, change the default device in `recovery.py` from `'4,5,6,7'` to `'0'`, and in `classify.json` change the GPU list from `"0,1,2,3,4,5,6,7"` to `"0"`.

---

## Running

### Baseline attack
```bash
python3 recovery.py --model VGG16 --improved_flag --dist_flag
```

### Running with defenses
```bash
# Gaussian noise only
python3 recovery.py --model VGG16 --improved_flag --dist_flag --noise_std 0.02

# Top-k masking only
python3 recovery.py --model VGG16 --improved_flag --dist_flag --top_k 10

# Confidence truncation only
python3 recovery.py --model VGG16 --improved_flag --dist_flag --truncate_decimals 2

# All three combined
python3 recovery.py --model VGG16 --improved_flag --dist_flag --noise_std 0.02 --top_k 10 --truncate_decimals 2
```

### Running all experiments
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This runs all 25 experiment configurations automatically, skipping any already completed, and saves each result to `results/`. To summarize results into a table:
```bash
python3 summarise_results.py
```

---

## Changes Made

Three files were added or modified:

- **`defense.py`** (new): Contains `defend_output()`, the single composable function implementing all three defenses.
- **`classify.py`** (modified): `VGG16.forward()` now calls `defend_output()` on the raw logits before returning them, reading defense parameters via `getattr(self, 'noise_std', 0.0)`, `getattr(self, 'top_k', 0)`, and `getattr(self, 'truncate_decimals', 0)`. When all parameters are at their defaults the model behaves identically to the original.
- **`recovery.py`** (modified): Three new command line arguments added — `--noise_std`, `--top_k`, `--truncate_decimals`. After the target model checkpoint is loaded, parameters are written to the model via `T.module.noise_std`, `T.module.top_k`, and `T.module.truncate_decimals`.

Two utility files were also added:

- **`run_experiments.sh`**: Runs all 25 defense configurations automatically.
- **`summarise_results.py`**: Parses all result files and prints a comparison table.

---

## Security Extensions

All three defenses operate at inference time inside `VGG16.forward()` in `classify.py`. The call sequence is: `recovery.py` sets defense parameters on the model → `attack.py` calls `T(fake)[-1]` → `VGG16.forward()` calls `defend_output(res, ...)` → the attacker receives only the defended logits. All defenses add zero meaningful computational overhead as they are single tensor operations on already-computed logits.

### Defense 1 — Gaussian Output Perturbation

Adds Gaussian noise with standard deviation `noise_std` to all 1000 output logits via `torch.randn_like(logits) * noise_std`. The motivation follows Fredrikson et al., who showed that confidence score perturbation can reduce model inversion effectiveness — see [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://rist.tech.cornell.edu/papers/mi-ccs.pdf) (CCS 2015). Legitimate users are unaffected since the correct class scores far higher than any other, so small noise never flips the top-1 ranking.

### Defense 2 — Top-k Masking

Retains only the top-k highest logits and sets all others to −10⁹ using `torch.topk`, a threshold mask, and `(~mask) * (-1e9)`. The attack computes `Iden_Loss = CrossEntropyLoss(out, iden)` using the full 1000-class distribution — removing 99%+ of it forces the optimizer to work with almost no gradient signal. Limiting API responses to top-k predictions is a widely recommended inference-time defense, as discussed in [Model Inversion Attacks: A Survey of Approaches and Countermeasures](https://arxiv.org/abs/2411.10023) (Zhou et al., 2024).

### Defense 3 — Confidence Truncation

Rounds all logits to a fixed number of decimal places via `torch.round(logits * scale) / scale`. The attack relies on fine-grained floating point differences across 2400 optimization iterations — truncation destroys this precision entirely. Originally proposed by Fredrikson et al. as a basic countermeasure in [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://rist.tech.cornell.edu/papers/mi-ccs.pdf) (CCS 2015), this remains underexplored against modern GAN-based attacks.
---

## Results

| Run | Attack Acc | Top-5 Acc | Acc Var | Acc5 Var |
|-----|-----------|-----------|---------|----------|
| noise_0.01_truncate_2 | 0.00 | 0.00 | 0.0000 | 0.0001 |
| noise_0.01_truncate_3 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| noise_0.02_topk_10_truncate_2 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| noise_0.03_topk_5_truncate_3 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| noise_0.03_truncate_2 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| noise_0.03_truncate_3 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| topk_1 | 0.00 | 0.00 | 0.0000 | 0.0001 |
| topk_10_truncate_2 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| topk_10_truncate_3 | 0.00 | 0.00 | 0.0000 | 0.0000 |
| topk_5_truncate_2 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| topk_5_truncate_3 | 0.00 | 0.00 | 0.0000 | 0.0000 |
| truncate_2 | 0.00 | 0.01 | 0.0000 | 0.0001 |
| truncate_3 | 0.00 | 0.00 | 0.0000 | 0.0001 |
| noise_0.01_topk_5 | 0.03 | 0.06 | 0.0002 | 0.0001 |
| topk_5 | 0.03 | 0.05 | 0.0002 | 0.0002 |
| noise_0.03_topk_5 | 0.04 | 0.07 | 0.0002 | 0.0003 |
| topk_10 | 0.09 | 0.16 | 0.0007 | 0.0004 |
| noise_0.01_topk_10 | 0.10 | 0.16 | 0.0005 | 0.0004 |
| noise_0.03_topk_10 | 0.10 | 0.16 | 0.0004 | 0.0005 |
| topk_50 | 0.44 | 0.62 | 0.0010 | 0.0004 |
| **baseline** | **0.68** | **0.90** | **0.0042** | **0.0007** |
| noise_0.03 | 0.68 | 0.89 | 0.0033 | 0.0016 |
| noise_0.02 | 0.69 | 0.91 | 0.0046 | 0.0013 |
| noise_0.05 | 0.69 | 0.91 | 0.0018 | 0.0008 |
| noise_0.01 | 0.70 | 0.90 | 0.0013 | 0.0005 |

Gaussian noise had no measurable effect across all tested values, with attack accuracy remaining at 0.68–0.70 — indistinguishable from the baseline. This is because KEDMI's distributional recovery optimizes over 2400 iterations, each sampling a fresh latent vector, so per-query noise averages out and the optimizer remains unaffected. This suggests noise-based defenses, while effective against single-query attacks as shown in Fredrikson et al. (CCS 2015), do not transfer to iterative distributional optimization.

Top-k masking shows a clear relationship between k and defense strength — k=50 reduces attack accuracy to 0.44, k=10 to 0.09, k=5 to 0.03, and k=1 to 0.00. Confidence truncation to just 2 decimal places alone reduces attack accuracy from 0.68 to 0.00, making it the strongest single defense found. Any combination involving truncation also achieves 0.00, confirming truncation dominates when combined. Notably, adding noise on top of top-k provides no meaningful improvement over top-k alone. This means that adding noise is not a safety measure for this type of attack but the other two formats does.Both effective defenses introduce a utility trade-off. Top-k masking reduces the information returned to the caller, and truncation reduces its precision. This is an inherent privacy-utility trade-off where stronger protection requires exposing less information to the end user.
---

## Citation
```
@inproceedings{chen2021knowledge,
  title={Knowledge-Enriched Distributional Model Inversion Attacks},
  author={Chen, Si and Kahla, Mostafa and Jia, Ruoxi and Qi, Guo-Jun},
  booktitle={ICCV},
  year={2021}
}
```