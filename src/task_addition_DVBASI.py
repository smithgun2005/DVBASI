import sys
import os
import time
import json
import torch
from torch import initial_seed
from torch.cuda.amp import GradScaler
from src.linearize import LinearizedImageEncoder
from src.modeling import ImageEncoder, MultiHeadImageClassifier
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.composition import WeightedImageEncoder, WeightedLinearizedModel, LinearizedDVBASI, NonlinearDVBASI
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.datasets.registry import get_dataset
from src.heads import get_classification_head
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.distributed import cleanup_ddp, distribute_loader, is_main_process, setup_ddp


def softmax_entropy(x):
    entropy = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return entropy.mean()

def avg(x):
    return sum(x) / len(x)

def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()

def main(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    datasets = [f"{dataset}Val" for dataset in args.datasets]
    n_datasets = len(datasets)
    ckpdir = os.path.join(args.save, f"combined_{n_datasets}")
    os.makedirs(ckpdir, exist_ok=True)

    assert args.finetuning_mode in [
        "linear", "standard",
    ], "Only linear and standard fine-tuning are supported."

    linearized_finetuning = args.finetuning_mode == "linear"
    if linearized_finetuning:
        print("Using linearized fine-tuning.")

    task_vectors = []
    for dataset in datasets:
        if args.finetuning_mode == "linear":
            pretrained_checkpoint = f"{args.save}/{dataset}/linearzeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/linearfinetuned.pt"
            task_vectors.append(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint))
        else:
            pretrained_checkpoint = f"{args.save}/{dataset}/zeroshot.pt"
            finetuned_checkpoint = f"{args.save}/{dataset}/finetuned.pt"
            task_vectors.append(NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint))

    if args.finetuning_mode == "linear":
        with open(os.path.join(args.save, "ft_accuracies.json"), 'r') as f:
            args.ft_acc = json.load(f)
            pretrained_path = f' '
            pretrained_params = torch.load(pretrained_path)
            optima_path = f' ' 
            optima_params = torch.load(optima_path, map_location=lambda storage, loc: storage)
            delta1 = []
            for first_p, second_p in zip(pretrained_params, optima_params):
                first_p = first_p.to("cuda:0")
                second_p = second_p.to("cuda:0")
                delta1.append(second_p - first_p)
            delta = []
            delta.append(delta1)
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = LinearizedDVBASI(
            image_encoder.model, task_vectors, blockwise=args.blockwise_coef,init_weights=optima_params, dparams=delta
        )

    else:
        with open(os.path.join(args.save, "ft_accuracies.json"), 'r') as f:
            args.ft_acc = json.load(f)

        with open(os.path.join(args.save, "ft_accuracies.json"), 'r') as f:
            args.ft_acc = json.load(f)
            pretrained_path = f' ' 
            pretrained_params = torch.load(pretrained_path)
            optima_path = f' '  
            optima_params = torch.load(optima_path, map_location=lambda storage, loc: storage)
            delta1 = []
            for first_p, second_p in zip(pretrained_params, optima_params):
                first_p = first_p.to("cuda:0")
                second_p = second_p.to("cuda:0")
                delta1.append(second_p - first_p)
            delta = []
            delta.append(delta1)
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = NonlinearDVBASI(
            image_encoder.model, task_vectors, blockwise=args.blockwise_coef, init_weights=optima_params, dparams=delta
        )
    if hasattr(image_encoder, "coef"):
        with torch.no_grad():
            image_encoder.coef.data.fill_(0.1)
    elif hasattr(image_encoder, "model") and hasattr(image_encoder.model, "coef"):
        with torch.no_grad():
            image_encoder.model.coef.data.fill_(0.1)
    preprocess_fn = image_encoder.train_preprocess


    dataloaders = [get_dataloader(
        get_dataset(
            dataset, preprocess_fn,
            location=args.data_location,
            batch_size=int(args.batch_size / n_datasets),
            num_workers=2),

        is_train=False, args=args, image_encoder=None
    ) for dataset in datasets]
    num_batches = [len(dataloader) for dataloader in dataloaders]
    prim = min(enumerate(num_batches), key=lambda x: x[1])[0]
    num_batches = num_batches[prim]
    prim_loader = dataloaders.pop(prim)

    classification_heads = [get_classification_head(args, dataset) for dataset in datasets]
    prim_head = classification_heads.pop(prim)
    classification_heads = [prim_head, ] + classification_heads
    model = MultiHeadImageClassifier(image_encoder, classification_heads)

    model.freeze_head()
    model = model.cuda()

    if args.print_every * 10 < num_batches:
        print_every = int(num_batches / 10)
    elif args.print_every * 4 > num_batches:
        print_every = max(int(num_batches / 4), 1)
    else:
        print_every = args.print_every

    ddp_prim_loader = distribute_loader(prim_loader)
    ddp_rmng_loader = [distribute_loader(dataloader) for dataloader in dataloaders]
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        output_device=rank,
    )

    loss_fn = softmax_entropy

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=args.wd)

    scaler = GradScaler()

    zs_acc = args.zs_acc
    best_acc = avg(zs_acc.values())
    patience = 20
    patience_counter = 0
    best_coef = None
    val_acc = []

    for epoch in range(args.epoch):
        ddp_prim_loader.sampler.set_epoch(epoch)
        for loader in ddp_rmng_loader:
            loader.sampler.set_epoch(epoch)
        rmng_iter = [iter(loader) for loader in ddp_rmng_loader]
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
                [batch["images"].cuda(), ] +
                [r_batch["images"].cuda() for r_batch in rmng_batch]
            )
            data_time = time.time() - start_time

            split = [len(batch["images"]), ] + [len(r_batch["images"]) for r_batch in rmng_batch]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(inputs, split)
                all_losses = [loss_fn(x) for x in logits]
                loss = sum(all_losses)
                if linearized_finetuning:
                    coef = ddp_model.module.image_encoder.model.coef
                else:
                    coef = ddp_model.module.image_encoder.coef
                loss = loss / args.num_grad_accumulation

            scaler.scale(loss).backward()

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

        if loss > prev_loss:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}")
        else:
            patience_counter = 0
            print(f"Early stopping counter: {patience_counter}")
            prev_loss = loss
            if linearized_finetuning:
                best_coef= ddp_model.module.image_encoder.model.coef.data.clone()
            else:
                best_coef = ddp_model.module.image_encoder.coef.data.clone()
        if patience_counter >= patience:
            print("Early stopping triggered due to loss increase.")
            break

    datasets = [dataset.replace("Val", "") for dataset in datasets]
    if is_main_process():
        image_encoder = ddp_model.module.image_encoder

        if linearized_finetuning:
            if args.epoch !=0:
                image_encoder.model.coef = torch.nn.Parameter(best_coef)
            new_params = image_encoder.model.get_optimized_params(coef=best_coef)
        else:
            image_encoder.coef = torch.nn.Parameter(best_coef)
            new_params = image_encoder.get_optimized_params(coef=best_coef)

        torch.save(new_params, os.path.join(ckpdir, ""))

        test_acc = {f"{dataset}:top1":
                        eval_single_dataset(image_encoder, dataset, args)["top1"]
                    for dataset in datasets
                    }
        test_acc["avg_top1"] = avg(test_acc.values())

        test_acc_norm = {
            f"{dataset}:normalised_top1": acc / args.ft_acc[dataset]
            for dataset, acc in zip(datasets, test_acc.values())
        }

        test_acc.update(test_acc_norm)
        test_acc["avg_normalised_top1"] = avg(test_acc_norm.values())

        with open(os.path.join(args.save, ""), 'w') as f:
            json.dump(test_acc, f, indent=4)

    cleanup_ddp()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    datasets = [
         "SVHN","DTD","Cars","EuroSAT", "GTSRB",
        "MNIST", "RESISC45", "SUN397",
    ]

    args = parse_arguments()
    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 10
    args.datasets = datasets
    args.epoch = 20
    args.save = f"/data/atlas/checkpoints/{args.model}"
    with open(os.path.join(args.save, "zeroshot_accuracies.json"), 'r') as f:
        args.zs_acc = json.load(f)

    print("=" * 100)
    print(f"Learn task vector coefficients of {args.model} on {len(datasets)} datasets:")
    for i, x in enumerate(datasets):
        print(f"{i + 1}. {x}")
    print("=" * 100)
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
