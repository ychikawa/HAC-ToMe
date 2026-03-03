# [WACV2026] Efficient Vision Transformers via Token Merging with Head-Wise Attention Correction

The Vision Transformers (ViTs) in this repository incorporate Token Merging (ToMe) and the proposed Head-wise Attention Correction (HAC) modules. This repo contains files required to train, fine-tune, and benchmark DeiT/LV-ViT/MAE-style models with ToMe, DiffRate, and related variants.

---

## 1. Setup
1. Clone the repository.
2. Create the Conda environment:
   ```bash
   conda env create --file env.yml
   conda activate hac_env
   ```

---

## 2. Directory Overview
- `main.py`: entry point aggregating all train/eval logic.
- `engine.py`: per-epoch training, evaluation, calibration routines.
- `utils.py`: distributed helpers plus FLOPs/throughput utilities.
- `models/`: DeiT/LV-ViT/MAE implementations and `models/algo` housing ToMe/S-HAC/L-HAC/etc.
- `data/`: ImageNet dataset loader and preprocessing code.
- `run/`: ready-to-use shell scripts for common training/eval jobs.
- `compression_rate.json`: DiffRate layer-wise token-keeping schedules.
- `output/` (generated): checkpoint directory; inference logs append to `result/result.txt`.

---

## 3. Dataset Preparation
- Expect ImageNet layout (`{DATA_PATH}/train`, `{DATA_PATH}/val`).
- Use `--sampling_ratio` / `--sampling_ratio_test` for subsampling.
- Update the `DATA_PATH` variable (line 3) inside every script under `run/` before executing.

---

## 4. Typical Usage

### 4.1 Common CLI Flags
- `--model`: choose from `deit_tiny`, `deit_small`, `deit_base`, `augreg_*`, `sam_base`, `lvvit_*`, `vit_*_mae`, etc.
- `--algo`: `tome`, `tome_SHAC`, `tome_LHAC`, `diffrate`, `diffrate_LHAC`, `mctf`, `mctf_LHAC`, `default`.
- `--r`: tokens removed per layer for ToMe-style algorithms.
- `--benchmark`: store FLOPs and throughput inside `log_stats`.
- `--modular`: update only HAC architecture parameters (used when training L-HAC).
- `--resume --resume-file ./output/checkpoint.pth`: resume training or run evaluation.

### 4.2 Training Scripts
- **L-HAC (distributed across 4 GPUs via torchrun)**

```./run/LHAC.sh
torchrun --nproc_per_node=4 -- main.py \
    --model $MODEL \
    --data_path $DATA_PATH \
    --batch-size 256 --epochs 1 \
    --output_dir ./output \
    --task_type [1,0,0] \
    --task_weight [1,0,0] \
    --algo $ALGO \
    --r $R \
    --modular
```

- **S-HAC (single-GPU benchmark example)**

```./run/SHAC.sh
python main.py \
    --model $MODEL \
    --data_path  $DATA_PATH \
    --task_type [1,0,0] \
    --algo tome_SHAC \
    --r $R \
    --benchmark
```

Each script stores stdout under `RESULT_DIR` and writes the latest weights to `output/checkpoint.pth`.

### 4.3 Evaluation Scripts
- `eval_LHAC.sh`: evaluate ToMe+LHAC (`--resume` reads `output/checkpoint.pth`).
- `eval_tome.sh`: evaluate plain ToMe without HAC.
- `eval_diffrate.sh`: DiffRate compression using `compression_rate.json` (`--tgt_flops` selects the schedule).

Adjust `MODEL`, `ALGO`, `R`, and `TGT_FLOPS` as needed.

---

## 5. Compression Schedules
- ToMe-style: specifying `--r` triggers `main.py`’s `r2schedule`, which produces layer-wise token counts consumed by `models/algo/tome_LHAC.py::CustomBlock`.
- DiffRate: `--load_schedule --tgt_flops <value>` pulls `merge_kept_num` / `prune_kept_num` from `compression_rate.json`.

---

## 6. Benchmarking and Logging
- `--benchmark` compiles the model with `torch.compile(mode="reduce-overhead")`, times 200 runs, and appends FLOPs/throughput/Acc@1 to `result/result.txt` (see `main.py`).
- `run/*.sh` scripts tee stdout to `results_*/{MODEL}_R{R}.txt`.
- Checkpoints in `--output_dir` are saved as `checkpoint.pth` and can be reloaded with `--resume`.

---

## 7. Citation
If you use this work in your research, please cite:

```bibtex
@InProceedings{Ichikawa2026WACV,
    author    = {Ichikawa, Yuki and Motomura, Masato and Van Chu, Thiem and Fujiki, Daichi},
    title     = {Efficient Vision Transformers via Token Merging with Head-wise Attention Correction},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {3908-3917}
}
