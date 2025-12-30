import argparse
import copy
import csv
import json
import logging
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from data_loaders import get_dataloaders
from model import MaxMViT_MLP, get_optimizer
from utils import load_config, seed_everything


@dataclass(frozen=True)
class Variant:
    name: str
    fusion: str
    head: str
    fusion_kwargs: dict
    head_kwargs: dict


def _default_variants() -> dict[str, Variant]:
    return {
        # Baseline (paper / original repo behavior)
        "baseline_concat_bn": Variant(
            name="baseline_concat_bn",
            fusion="concat",
            head="mlp_bn",
            fusion_kwargs={},
            head_kwargs={},
        ),
        # Head-only improvements
        "concat_ln": Variant(
            name="concat_ln",
            fusion="concat",
            head="mlp_ln",
            fusion_kwargs={},
            head_kwargs={},
        ),
        "concat_residual_ln": Variant(
            name="concat_residual_ln",
            fusion="concat",
            head="mlp_residual_ln",
            fusion_kwargs={},
            head_kwargs={"n_blocks": 2},
        ),
        # Fusion improvements (use LN head by default)
        "gated_scalar_concat_ln": Variant(
            name="gated_scalar_concat_ln",
            fusion="gated_scalar",
            head="mlp_ln",
            fusion_kwargs={"mode": "concat"},
            head_kwargs={},
        ),
        "gated_scalar_sum_ln": Variant(
            name="gated_scalar_sum_ln",
            fusion="gated_scalar",
            head="mlp_ln",
            fusion_kwargs={"mode": "sum"},
            head_kwargs={},
        ),
        "gated_channel_complement_ln": Variant(
            name="gated_channel_complement_ln",
            fusion="gated_channel",
            head="mlp_ln",
            fusion_kwargs={"complementary": True},
            head_kwargs={},
        ),
        "gated_channel_indep_ln": Variant(
            name="gated_channel_indep_ln",
            fusion="gated_channel",
            head="mlp_ln",
            fusion_kwargs={"complementary": False},
            head_kwargs={},
        ),
        "attn2token_ln": Variant(
            name="attn2token_ln",
            fusion="attn2token",
            head="mlp_ln",
            fusion_kwargs={"d_model": 768, "n_layers": 1, "n_heads": 8, "dropout": 0.1},
            head_kwargs={},
        ),
        # Bilinear-style fusions
        "hadamard_concat_ln": Variant(
            name="hadamard_concat_ln",
            fusion="hadamard",
            head="mlp_ln",
            fusion_kwargs={"d_proj": 768, "mode": "concat", "dropout": 0.0},
            head_kwargs={},
        ),
        "hadamard_prod_ln": Variant(
            name="hadamard_prod_ln",
            fusion="hadamard",
            head="mlp_ln",
            fusion_kwargs={"d_proj": 768, "mode": "prod", "dropout": 0.0},
            head_kwargs={},
        ),
        "lowrank_bilinear_ln": Variant(
            name="lowrank_bilinear_ln",
            fusion="lowrank_bilinear",
            head="mlp_ln",
            fusion_kwargs={"rank": 256, "d_out": 768, "dropout": 0.0},
            head_kwargs={},
        ),
    }


def _configure_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,  # important for multiple runs in one process
    )


def _make_run_dirs(base_config: dict, experiment_name: str) -> tuple[str, str]:
    log_root = base_config.get("paths", {}).get("log_dir", "logs")
    ckpt_root = base_config.get("paths", {}).get("checkpoint_dir", "checkpoints")
    run_log_dir = os.path.join(log_root, experiment_name)
    run_ckpt_dir = os.path.join(ckpt_root, experiment_name)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    return run_log_dir, run_ckpt_dir


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _train_one_variant(base_config: dict, variant: Variant, args) -> dict:
    run_config = copy.deepcopy(base_config)
    base_name = base_config.get("experiment_name", "experiment")
    experiment_name = f"{base_name}__abl__{variant.name}"
    run_config["experiment_name"] = experiment_name

    train_cfg = run_config.get("training", {})
    model_cfg = run_config.get("model", {})

    device_name = args.device or train_cfg.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 50))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 0.0002))
    patience = int(args.patience if args.patience is not None else train_cfg.get("patience", 5))

    run_log_dir, run_ckpt_dir = _make_run_dirs(run_config, experiment_name)
    log_file = os.path.join(run_log_dir, "ablation.log")
    _configure_logging(log_file)

    seed = int(train_cfg.get("seed", 42))
    seed_everything(seed)

    logging.info(f"Variant: {variant.name}")
    logging.info(f"Fusion: {variant.fusion} kwargs={variant.fusion_kwargs}")
    logging.info(f"Head: {variant.head} kwargs={variant.head_kwargs}")
    logging.info(f"Device: {device} | epochs={epochs} | lr={lr} | patience={patience}")

    train_loader, val_loader = get_dataloaders(run_config)
    if not train_loader:
        raise RuntimeError("Failed to load data (train_loader is empty).")

    num_classes = int(model_cfg.get("num_classes", 4))
    hidden_size = int(args.hidden_size if args.hidden_size is not None else model_cfg.get("hidden_size", 512))
    dropout_rate = float(args.dropout_rate if args.dropout_rate is not None else model_cfg.get("dropout_rate", 0.2))

    model = MaxMViT_MLP(
        num_classes=num_classes,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        fusion=variant.fusion,
        head=variant.head,
        fusion_kwargs=variant.fusion_kwargs,
        head_kwargs=variant.head_kwargs,
        pretrained=not args.no_pretrained,
        verbose=args.verbose_model,
    ).to(device)

    total_params = _count_params(model)
    logging.info(f"Total params: {total_params:,}")

    optimizers = get_optimizer(model, lr=lr)
    sched_cfg = train_cfg.get("scheduler", {})
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.1)),
            patience=int(sched_cfg.get("patience", 2)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
        for opt in optimizers
    ]

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    top_k_checkpoints: list[dict] = []
    top_k = int(args.top_k)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (cqt, mel, label) in enumerate(train_loader):
            cqt, mel, label = cqt.to(device), mel.to(device), label.to(device)
            for opt in optimizers:
                opt.zero_grad()

            outputs = model(cqt, mel)
            loss = criterion(outputs, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for opt in optimizers:
                opt.step()

            total_loss += float(loss.item())
            _, predicted = outputs.max(1)
            total += int(label.size(0))
            correct += int(predicted.eq(label).sum().item())

            if args.debug_batches and (batch_idx % int(args.debug_batches) == 0):
                logging.info(f"[{variant.name}] epoch={epoch+1} batch={batch_idx} loss={loss.item():.4f}")

        train_loss = total_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        if val_loader:
            with torch.no_grad():
                for cqt, mel, label in val_loader:
                    cqt, mel, label = cqt.to(device), mel.to(device), label.to(device)
                    outputs = model(cqt, mel)
                    loss = criterion(outputs, label)
                    val_loss += float(loss.item())
                    _, predicted = outputs.max(1)
                    val_total += int(label.size(0))
                    val_correct += int(predicted.eq(label).sum().item())
            val_loss = val_loss / max(1, len(val_loader))
            val_acc = 100.0 * val_correct / max(1, val_total)
        else:
            val_loss = train_loss
            val_acc = train_acc

        for sch in schedulers:
            sch.step(val_loss)

        logging.info(
            f"[{variant.name}] epoch={epoch+1:02d} "
            f"train(L={train_loss:.4f},A={train_acc:.1f}%) "
            f"val(L={val_loss:.4f},A={val_acc:.1f}%)"
        )

        # Save checkpoint each epoch (optional)
        ckpt_path = None
        if not args.no_checkpoints:
            ckpt_path = os.path.join(run_ckpt_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

            top_k_checkpoints.append(
                {"acc": float(val_acc), "loss": float(val_loss), "epoch": int(epoch + 1), "path": ckpt_path}
            )
            top_k_checkpoints.sort(key=lambda x: x["acc"], reverse=True)
            while len(top_k_checkpoints) > top_k:
                to_remove = top_k_checkpoints.pop()
                if os.path.exists(to_remove["path"]):
                    os.remove(to_remove["path"])

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            best_epoch = int(epoch + 1)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"[{variant.name}] early stopping triggered (patience={patience}).")
                break

    best_ckpt = None
    if top_k_checkpoints:
        best_ckpt = top_k_checkpoints[0]["path"]

    elapsed = time.time() - start_time
    logging.info(f"[{variant.name}] done. best_val_acc={best_val_acc:.2f}% @ epoch={best_epoch}")

    # Free memory between runs
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "variant": variant.name,
        "fusion": variant.fusion,
        "head": variant.head,
        "fusion_kwargs": variant.fusion_kwargs,
        "head_kwargs": variant.head_kwargs,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "params": total_params,
        "device": str(device),
        "epochs_ran": epoch + 1,
        "seconds": elapsed,
        "log_file": log_file,
        "best_checkpoint": best_ckpt,
    }


def _write_outputs(results: list[dict], out_dir: str) -> tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)

    results_sorted = sorted(results, key=lambda r: (r["best_val_acc"], -r["best_val_loss"]), reverse=True)

    csv_path = os.path.join(out_dir, "ablation_results.csv")
    json_path = os.path.join(out_dir, "ablation_results.json")
    md_path = os.path.join(out_dir, "ablation_results.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(results_sorted)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Results (Fusion + Head)\n\n")
        f.write("| Rank | Variant | Fusion | Head | Best Val Acc (%) | Best Epoch | Params |\n")
        f.write("|---:|---|---|---|---:|---:|---:|\n")
        for i, r in enumerate(results_sorted, start=1):
            f.write(
                f"| {i} | {r['variant']} | {r['fusion']} | {r['head']} | "
                f"{r['best_val_acc']:.2f} | {r['best_epoch']} | {r['params']} |\n"
            )
        f.write("\n")

    return csv_path, json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Train ablation variants (fusion + head) and compare results.")
    parser.add_argument("--config", type=str, required=True, help="Path to base config yaml")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant names, or 'all'. See train_ablation.py for keys.",
    )
    parser.add_argument("--out-dir", type=str, default="ablation_out", help="Directory for comparison outputs")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda, cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all variants")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--hidden-size", type=int, default=None, help="Override MLP hidden size")
    parser.add_argument("--dropout-rate", type=float, default=None, help="Override MLP dropout rate")
    parser.add_argument("--top-k", type=int, default=3, help="Keep top-k checkpoints by val acc")
    parser.add_argument("--no-checkpoints", action="store_true", help="Do not save checkpoints")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet pretrained backbones")
    parser.add_argument("--verbose-model", action="store_true", help="Print backbone init messages")
    parser.add_argument("--debug-batches", type=int, default=0, help="Log every N batches (0 disables)")

    args = parser.parse_args()

    base_config = load_config(args.config)
    variants_map = _default_variants()

    if args.variants.strip().lower() == "all":
        selected = list(variants_map.values())
    else:
        names = [v.strip() for v in args.variants.split(",") if v.strip()]
        missing = [n for n in names if n not in variants_map]
        if missing:
            raise SystemExit(f"Unknown variants: {missing}. Available: {sorted(variants_map.keys())}")
        selected = [variants_map[n] for n in names]

    results: list[dict] = []
    for variant in selected:
        results.append(_train_one_variant(base_config, variant, args))

    csv_path, json_path, md_path = _write_outputs(results, args.out_dir)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
