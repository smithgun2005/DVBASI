import json
import os
import subprocess
import sys
import time
import warnings

import torch
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/openclip-cachedir/open_clip"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.args import parse_arguments
from src.composition import LinearizedDVBASI, NonlinearDVBASI
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp
from src.eval import eval_single_dataset
from src.heads import get_classification_head
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, MultiHeadImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector


def avg(x):
    return sum(x) / len(x)


def lp_reg(x, p=None, gamma=0.5):
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()


def clone_tensor_list(params):
    return [p.detach().cpu().clone() for p in params]


def ensure_compatible_params(params_a, params_b, name_a, name_b):
    if len(params_a) != len(params_b):
        raise ValueError(
            f"Parameter list length mismatch: {name_a}={len(params_a)} vs {name_b}={len(params_b)}"
        )

    for idx, (pa, pb) in enumerate(zip(params_a, params_b)):
        if tuple(pa.shape) != tuple(pb.shape):
            raise ValueError(
                f"Parameter shape mismatch at index {idx}: "
                f"{name_a}={tuple(pa.shape)} vs {name_b}={tuple(pb.shape)}"
            )


def load_param_list_from_checkpoint(path, reference_names=None):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, (list, tuple)):
        return clone_tensor_list(obj)

    if isinstance(obj, dict):
        if "params" in obj and isinstance(obj["params"], (list, tuple)):
            return clone_tensor_list(obj["params"])

        tensor_items = [(k, v) for k, v in obj.items() if torch.is_tensor(v)]
        if tensor_items and len(tensor_items) == len(obj):
            tensor_dict = dict(tensor_items)
            if reference_names is not None and all(k in tensor_dict for k in reference_names):
                return clone_tensor_list([tensor_dict[k] for k in reference_names])
            return clone_tensor_list(list(tensor_dict.values()))

        raise TypeError(f"Unsupported dict checkpoint format at {path}")

    if hasattr(obj, "named_parameters"):
        return clone_tensor_list([p for _, p in obj.named_parameters()])

    raise TypeError(f"Unsupported checkpoint type at {path}: {type(obj)}")


def resolve_default_save_dir(model_name):
    candidates = [
        f"/data/atlas/checkpoints/{model_name}",
        f"/root/autodl-tmp/atlas-main/checkpoints/{model_name}",
        os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "checkpoints",
            model_name,
        ),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def resolve_existing_path(candidates, description):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    pretty = "\n".join(f"  - {p}" for p in candidates if p)
    raise FileNotFoundError(f"Cannot find {description}. Tried:\n{pretty}")


def resolve_pretrained_path(args, datasets_val, linearized):
    if args.dv_pretrained_path is not None:
        return os.path.expanduser(args.dv_pretrained_path)

    ckpt_name = "linear_zeroshot.pt" if linearized else "zeroshot.pt"
    candidates = [
        os.path.join(args.save, datasets_val[0], ckpt_name),
        os.path.join(args.save, datasets_val[0].replace("Val", ""), ckpt_name),
    ]
    return resolve_existing_path(candidates, "DV-BASI pretrained checkpoint")


def atlas_optima_candidates(save_dir, n_datasets):
    ckpdir = os.path.join(save_dir, f"combined_{n_datasets}")
    return [
        os.path.join(ckpdir, "addition_baseline"),
        os.path.join(ckpdir, "addition_baseline.pt"),
        os.path.join(save_dir, "addition_baseline"),
        os.path.join(save_dir, "addition_baseline.pt"),
    ]


def resolve_initial_optima_path(args, n_datasets):
    if args.dv_initial_optima_path is not None:
        path = os.path.expanduser(args.dv_initial_optima_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"dv-initial-optima-path not found: {path}")
        return path

    ckpdir = os.path.join(args.save, f"combined_{n_datasets}")
    candidates = atlas_optima_candidates(args.save, n_datasets) + [
        os.path.join(ckpdir, "DV_BASI_addition_baseline.pt"),
        os.path.join(args.save, "DV_BASI_addition_baseline.pt"),
    ]
    return resolve_existing_path(candidates, "DV-BASI initial optima (aTLAS output)")


def resolve_output_dir(args, n_datasets):
    if args.dv_output_dir is not None:
        outdir = os.path.expanduser(args.dv_output_dir)
    else:
        outdir = os.path.join(args.save, f"combined_{n_datasets}", "DV_BASI_auto")

    os.makedirs(outdir, exist_ok=True)
    return outdir


def build_task_vectors(args, datasets_val, linearized):
    task_vectors = []
    for dataset in datasets_val:
        if linearized:
            pretrained_checkpoint = os.path.join(args.save, dataset, "linear_zeroshot.pt")
            finetuned_checkpoint = os.path.join(args.save, dataset, "linear_finetuned.pt")
            task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))
        else:
            pretrained_checkpoint = os.path.join(args.save, dataset, "zeroshot.pt")
            finetuned_checkpoint = os.path.join(args.save, dataset, "finetuned.pt")
            task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    return task_vectors


def build_dataloaders(args, datasets_val, preprocess_fn):
    n_datasets = len(datasets_val)
    dataloaders = [
        get_dataloader(
            get_dataset(
                dataset,
                preprocess_fn,
                location=args.data_location,
                batch_size=max(1, int(args.batch_size / n_datasets)),
                num_workers=2,
            ),
            is_train=False,
            args=args,
            image_encoder=None,
        )
        for dataset in datasets_val
    ]

    num_batches = [len(dataloader) for dataloader in dataloaders]
    prim = min(enumerate(num_batches), key=lambda x: x[1])[0]
    prim_loader = dataloaders.pop(prim)
    num_batches = num_batches[prim]

    return prim_loader, dataloaders, prim, num_batches


def build_dvbasi_image_encoder(args, task_vectors, linearized, optima_params, delta_params):
    if linearized:
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = LinearizedDVBASI(
            image_encoder.model,
            task_vectors,
            blockwise=args.blockwise_coef,
            init_weights=optima_params,
            dparams=[delta_params],
        )
        return image_encoder

    image_encoder = ImageEncoder(args)
    image_encoder = NonlinearDVBASI(
        image_encoder,
        task_vectors,
        blockwise=args.blockwise_coef,
        init_weights=optima_params,
        dparams=[delta_params],
    )
    return image_encoder


def extract_coef(ddp_model, linearized):
    if linearized:
        return ddp_model.module.image_encoder.model.coef
    return ddp_model.module.image_encoder.coef


def get_optimized_params(image_encoder, linearized, final_coef):
    if linearized:
        image_encoder.model.coef = torch.nn.Parameter(final_coef)
        return image_encoder.model.get_optimized_params(coef=final_coef)

    image_encoder.coef = torch.nn.Parameter(final_coef)
    return image_encoder.get_optimized_params(coef=final_coef)


def _append_arg(cmd, flag, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _append_flag(cmd, flag, enabled):
    if enabled:
        cmd.append(flag)


def _safe_remove(path):
    if path and os.path.exists(path):
        os.remove(path)


def run_atlas_if_needed(args, n_datasets):
    if not args.dv_auto_atlas:
        return

    atlas_candidates = atlas_optima_candidates(args.save, n_datasets)
    atlas_exists = any(os.path.exists(p) for p in atlas_candidates)

    # force 时先删旧文件，避免“看起来重跑了，实际上读到旧 baseline”
    if args.dv_auto_atlas_force:
        for p in atlas_candidates:
            _safe_remove(p)
        atlas_exists = False

    if atlas_exists:
        print("[DV-BASI] aTLAS baseline exists, skip auto atlas run.")
    else:
        atlas_epochs = args.dv_atlas_epochs if args.dv_atlas_epochs is not None else args.epochs
        atlas_script = os.path.join(PROJECT_ROOT, "src", "task_addition_atlas.py")

        cmd = [
            sys.executable,
            atlas_script,
            "--save", args.save,
            "--data-location", args.data_location,
            "--model", args.model,
            "--epochs", str(atlas_epochs),
            "--world-size", str(args.world_size),
            "--port", str(args.port),
            "--finetuning-mode", args.finetuning_mode,
        ]

        # ===== 关键：把会影响训练结果的参数继续透传给 aTLAS =====
        _append_arg(cmd, "--lr", getattr(args, "lr", None))
        _append_arg(cmd, "--wd", getattr(args, "wd", None))
        _append_arg(cmd, "--lp-reg", getattr(args, "lp_reg", None))
        _append_arg(cmd, "--subsample", getattr(args, "subsample", None))
        _append_arg(cmd, "--seed", getattr(args, "seed", None))

        # 如果你的 args 里还有这些，就一起透传；没有就自动跳过
        _append_arg(cmd, "--batch-size", getattr(args, "batch_size", None))
        _append_arg(cmd, "--num-grad-accumulation", getattr(args, "num_grad_accumulation", None))

        _append_flag(cmd, "--blockwise-coef", getattr(args, "blockwise_coef", False))
        _append_flag(cmd, "--unsupervised", getattr(args, "unsupervised", False))
        _append_arg(cmd, "--dv-init-coef", getattr(args, "dv_init_coef", None) or None)

        print("[DV-BASI] Running aTLAS automatically before DV iterations...")
        print("[DV-BASI] aTLAS cmd:")
        print(" ".join(cmd), flush=True)

        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=os.environ.copy())
        print("[DV-BASI] aTLAS auto-run finished.")

    atlas_path = resolve_existing_path(atlas_candidates, "aTLAS addition baseline")
    if args.dv_initial_optima_path is None:
        args.dv_initial_optima_path = atlas_path
        print(f"[DV-BASI] Use aTLAS baseline as first optima: {atlas_path}")


def train_one_iteration(rank, args, linearized, datasets_val, prim_loader, rmng_loaders, prim, num_batches,
                        task_vectors, optima_params, pretrained_params):
    delta_params = [opt - pre for pre, opt in zip(pretrained_params, optima_params)]
    image_encoder = build_dvbasi_image_encoder(
        args,
        task_vectors,
        linearized,
        optima_params=optima_params,
        delta_params=delta_params,
    )

    classification_heads = [get_classification_head(args, dataset) for dataset in datasets_val]
    prim_head = classification_heads.pop(prim)
    classification_heads = [prim_head] + classification_heads

    model = MultiHeadImageClassifier(image_encoder, classification_heads)
    model.freeze_head()
    model = model.cuda().float()

    ddp_prim_loader = distribute_loader(prim_loader)
    ddp_rmng_loader = [distribute_loader(dataloader) for dataloader in rmng_loaders]

    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=True,
        output_device=rank,
    )

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler()
    loss_fn = softmax_entropy if args.unsupervised else torch.nn.CrossEntropyLoss()

    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    coef = extract_coef(ddp_model, linearized)

    # Apply initial perturbation: set α₀ to dv_init_coef along the difference-vector direction.
    # α=0 (default) starts exactly at the current optimum; a nonzero value (e.g. 0.1) gives
    # an initial step along Δ, helping escape flat loss landscapes (see paper §3.2).
    if args.dv_init_coef != 0.0:
        with torch.no_grad():
            coef.data.fill_(args.dv_init_coef)

    best_epoch_loss = float("inf")
    no_improve_epochs = 0
    epoch_avg_losses = []
    inner_early_stopped = False

    for epoch in range(args.epoch):
        ddp_prim_loader.sampler.set_epoch(epoch)
        for loader in ddp_rmng_loader:
            loader.sampler.set_epoch(epoch)

        rmng_iter = [iter(loader) for loader in ddp_rmng_loader]
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for i, batch in enumerate(ddp_prim_loader):
            rmng_batch = [next(r_iter) for r_iter in rmng_iter]
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            rmng_batch = [maybe_dictionarize(r_batch) for r_batch in rmng_batch]

            inputs = torch.cat(
                [batch["images"].cuda()] + [r_batch["images"].cuda() for r_batch in rmng_batch]
            )
            data_time = time.time() - start_time

            split = [len(batch["images"])] + [len(r_batch["images"]) for r_batch in rmng_batch]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = ddp_model(inputs, split)
                if args.unsupervised:
                    all_losses = [loss_fn(x) for x in logits]
                    loss_unscaled = sum(all_losses) + lp_reg(coef, args.lp_reg)
                else:
                    labels = [batch["labels"].cuda()] + [r_batch["labels"].cuda() for r_batch in rmng_batch]
                    all_losses = [loss_fn(x, y) for x, y in zip(logits, labels)]
                    loss_unscaled = sum(all_losses) + lp_reg(coef, args.lp_reg)
                loss = loss_unscaled / args.num_grad_accumulation

            scaler.scale(loss).backward()
            epoch_loss_sum += float(loss_unscaled.detach().item())
            epoch_loss_count += 1

            if i % args.num_grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and (i % args.num_grad_accumulation == 0)
                and is_main_process()
            ):
                percent_complete = 100 * i / len(ddp_prim_loader)
                print_losses = str([round(x.item(), 4) for x in all_losses])
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(ddp_prim_loader)}]\t"
                    f"Losses: {print_losses:<72}\t"
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}",
                    flush=True,
                )

        epoch_stats = torch.tensor([epoch_loss_sum, epoch_loss_count], dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(epoch_stats, op=torch.distributed.ReduceOp.SUM)
        global_avg_loss = (epoch_stats[0] / max(epoch_stats[1], 1.0)).item()
        epoch_avg_losses.append(global_avg_loss)

        improved = global_avg_loss < (best_epoch_loss - args.dv_min_improve)
        if improved:
            best_epoch_loss = global_avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        stop_flag = torch.tensor(0, device="cuda")
        if is_main_process():
            print(
                f"[Inner-ES] epoch={epoch} avg_loss={global_avg_loss:.6f} "
                f"improved={improved} no_improve={no_improve_epochs}/{args.dv_patience}"
            )
            if args.dv_patience > 0 and no_improve_epochs >= args.dv_patience:
                print("[Inner-ES] Stop current DV iteration due to epoch-loss patience.")
                stop_flag = torch.tensor(1, device="cuda")

        torch.distributed.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            inner_early_stopped = True
            break

    image_encoder = ddp_model.module.image_encoder
    final_coef = coef.data.clone()
    new_params = get_optimized_params(image_encoder, linearized, final_coef)
    new_params_cpu = clone_tensor_list(new_params)
    train_info = {
        "epoch_avg_losses": epoch_avg_losses,
        "inner_early_stopped": inner_early_stopped,
        "epochs_ran": len(epoch_avg_losses),
        "best_epoch_avg_loss": best_epoch_loss,
    }

    return image_encoder, new_params_cpu, train_info


def main(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    datasets_val = [f"{dataset}Val" for dataset in args.datasets]
    n_datasets = len(datasets_val)
    linearized = args.finetuning_mode == "linear"

    ft_candidates = [
        os.path.join(args.save, "linear_ft_accuracies.json") if linearized else None,
        os.path.join(args.save, "ft_accuracies.json"),
    ]
    ft_path = resolve_existing_path([p for p in ft_candidates if p is not None], "ft_accuracies.json")
    with open(ft_path, "r") as f:
        args.ft_acc = json.load(f)

    base_encoder = LinearizedImageEncoder(args, keep_lang=False) if linearized else ImageEncoder(args)
    preprocess_fn = base_encoder.train_preprocess

    prim_loader, rmng_loaders, prim, num_batches = build_dataloaders(args, datasets_val, preprocess_fn)

    task_vectors = build_task_vectors(args, datasets_val, linearized)

    pretrained_path = resolve_pretrained_path(args, datasets_val, linearized)
    initial_optima_path = resolve_initial_optima_path(args, n_datasets)
    outdir = resolve_output_dir(args, n_datasets)

    pretrained_params = load_param_list_from_checkpoint(pretrained_path)
    optima_params = load_param_list_from_checkpoint(initial_optima_path)
    ensure_compatible_params(pretrained_params, optima_params, "pretrained", "initial_optima")

    if is_main_process():
        print("=" * 100)
        print(f"DV-BASI auto-run on {args.model} with {len(datasets_val)} datasets")
        print(f"Pretrained params : {pretrained_path}")
        print(f"Initial optima    : {initial_optima_path}")
        print(f"Output dir        : {outdir}")
        print("=" * 100)

    best_score = float("-inf")
    best_iter = 0
    iter_logs = []

    for dv_iter in range(1, args.dv_max_iters + 1):
        if is_main_process():
            print(f"\n[DV-BASI] Iteration {dv_iter}/{args.dv_max_iters}")

        image_encoder, new_params_cpu, train_info = train_one_iteration(
            rank,
            args,
            linearized,
            datasets_val,
            prim_loader,
            rmng_loaders,
            prim,
            num_batches,
            task_vectors,
            optima_params,
            pretrained_params,
        )

        ensure_compatible_params(pretrained_params, new_params_cpu, "pretrained", "iter_optima")

        if is_main_process():
            iter_path = os.path.join(outdir, f"DV_BASI_addition_iter_{dv_iter}.pt")
            torch.save(new_params_cpu, iter_path)

            datasets_test = [dataset.replace("Val", "") for dataset in datasets_val]
            test_acc = {
                f"{dataset}:top1": eval_single_dataset(image_encoder, dataset, args)["top1"]
                for dataset in datasets_test
            }
            top1_values = [test_acc[f"{dataset}:top1"] for dataset in datasets_test]
            test_acc["avg_top1"] = avg(top1_values)

            test_acc_norm = {
                f"{dataset}:normalised_top1": test_acc[f"{dataset}:top1"] / args.ft_acc[f"{dataset}Val"]
                for dataset in datasets_test
            }
            test_acc.update(test_acc_norm)
            test_acc["avg_normalised_top1"] = avg(test_acc_norm.values())

            score = test_acc["avg_normalised_top1"]
            improved = score > (best_score + args.dv_min_improve)

            if improved:
                best_score = score
                best_iter = dv_iter
                torch.save(new_params_cpu, os.path.join(outdir, "DV_BASI_addition_best.pt"))

            iter_record = {
                "iter": dv_iter,
                "iter_checkpoint": iter_path,
                "improved": improved,
                "score": score,
                "metrics": test_acc,
                "train_info": train_info,
            }
            iter_logs.append(iter_record)

            summary = {
                "pretrained_path": pretrained_path,
                "initial_optima_path": initial_optima_path,
                "best_iter": best_iter,
                "best_score": best_score,
                "records": iter_logs,
            }
            with open(os.path.join(outdir, "DV_BASI_auto_log.json"), "w") as f:
                json.dump(summary, f, indent=4)

            print(
                f"[DV-BASI] iter={dv_iter} score={score:.6f} "
                f"improved={improved} inner_early_stopped={train_info['inner_early_stopped']}"
            )

        # Use current iteration optima as the next iteration start point.
        optima_params = new_params_cpu

        torch.cuda.empty_cache()

    cleanup_ddp()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    default_datasets = [
        "Cars", "DTD", "EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397", "SVHN",
    ]

    args = parse_arguments()
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.datasets = default_datasets
    args.epoch = getattr(args, "epoch", args.epochs)
    # 无监督熵值开关：传入 --unsupervised 则使用 softmax_entropy，否则使用 CrossEntropyLoss
    if not hasattr(args, "unsupervised") or args.unsupervised is None:
        args.unsupervised = False

    if args.save is None:
        args.save = resolve_default_save_dir(args.model)
    args.save = os.path.expanduser(args.save)
    if not os.path.isdir(args.save):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {args.save}. "
            "Pass --save to specify the checkpoint root."
        )

    zs_path = os.path.join(args.save, "zeroshot_accuracies.json")
    if not os.path.isfile(zs_path):
        raise FileNotFoundError(f"Missing zeroshot accuracy file: {zs_path}")
    with open(zs_path, "r") as f:
        args.zs_acc = json.load(f)

    print("=" * 100)
    print(f"Learn task vector coefficients of {args.model} on {len(args.datasets)} datasets:")
    for i, x in enumerate(args.datasets):
        print(f"{i + 1}. {x}")
    print("=" * 100)

    run_atlas_if_needed(args, n_datasets=len(args.datasets))

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
