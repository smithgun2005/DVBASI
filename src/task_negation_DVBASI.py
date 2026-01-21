
import os
import time
import json
import torch

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


def lp_reg(x, p=None, gamma=0.5) -> torch.Tensor:
    return 0 if p is None else gamma * torch.norm(x, p=p, dim=0).mean()


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:

    logp = x.log_softmax(dim=1)
    p = x.softmax(dim=1)
    entropy = -(p * logp).sum(dim=1)
    return entropy.mean()


def main(rank, args):
    setup_ddp(rank, args.world_size, port=args.port)

    patience = getattr(args, "patience", 20)
    control_weight = getattr(args, "control_weight", 2.0)

    tgt_dataset = args.tgt_dataset
    ctr_dataset = args.ctr_dataset

    ckpdir = os.path.join("data/atlas", args.save, tgt_dataset)
    os.makedirs(ckpdir, exist_ok=True)

    linearized = (args.finetuning_mode == "linear")
    if linearized:
        task_vectors = [LinearizedTaskVector(
            os.path.join(args.save, tgt_dataset, "linear_zeroshot.pt"),
            os.path.join(args.save, tgt_dataset, "linear_finetuned.pt")
        )]
        checkpoint_path = f''
        checkpoint = torch.load(checkpoint_path)
        checkpoint_path_pre = f''
        checkpoint_pre = torch.load(checkpoint_path_pre, map_location=lambda storage, loc: storage)
        theta_pre = checkpoint_pre
        delta1 = []
        for first_p, second_p in zip(checkpoint, theta_pre):
            first_p = first_p.to("cuda:0")
            second_p = second_p.to("cuda:0")
            delta1.append(second_p - first_p)
        delta = []
        delta.append(delta1)
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
        image_encoder.model = LinearizedDVBASI(
            image_encoder.model, task_vectors, blockwise=args.blockwise_coef, init_weights=optima_params, dparams=delta
        )

    else:
        task_vectors = [NonLinearTaskVector(
            os.path.join(args.save, tgt_dataset, "zeroshot.pt"),
            os.path.join(args.save, tgt_dataset, "finetuned.pt")
        )]
        checkpoint_path = f''
        checkpoint = torch.load(checkpoint_path)
        checkpoint_path_pre = f''
        checkpoint_pre = torch.load(checkpoint_path_pre, map_location=lambda storage, loc: storage)
        theta_pre = checkpoint_pre
        delta1 = []
        for first_p, second_p in zip(checkpoint, theta_pre):
            first_p = first_p.to("cuda:0")
            second_p = second_p.to("cuda:0")
            delta1.append(second_p - first_p)
        delta = []
        delta.append(delta1)
        image_encoder = ImageEncoder(args)
        image_encoder.model = NonlinearDVBASI(
            image_encoder.model, task_vectors, blockwise=args.blockwise_coef, init_weights=optima_params, dparams=delta
        )

    tgt_head = get_classification_head(args, tgt_dataset)
    ctr_head = get_classification_head(args, ctr_dataset)
    model = MultiHeadImageClassifier(image_encoder, [tgt_head, ctr_head])
    model.freeze_head()
    model = model.cuda()

    preprocess = model.train_preprocess
    bs2 = int(args.batch_size / 2)
    tgt_loader = get_dataloader(
        get_dataset(tgt_dataset, preprocess, location=args.data_location,
                    batch_size=bs2, num_workers=2),
        is_train=False, args=args, image_encoder=None
    )
    ctr_loader = get_dataloader(
        get_dataset(ctr_dataset, preprocess, location=args.data_location,
                    batch_size=bs2, num_workers=2),
        is_train=False, args=args, image_encoder=None
    )
    num_batches = len(tgt_loader)

    if args.print_every * 10 < num_batches:
        print_every = num_batches // 10
    elif args.print_every * 4 > num_batches:
        print_every = max(num_batches // 4, 1)
    else:
        print_every = args.print_every

    ddp_tgt = distribute_loader(tgt_loader)
    ddp_ctr = distribute_loader(ctr_loader)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=False
    )

    params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr * args.lr_multiplier, weight_decay=args.wd)
    scaler = GradScaler()


    if linearized:
        head_path = os.path.join(ckpdir, "")
        log_path = os.path.join(args.save, "")
        coef = ddp_model.module.image_encoder.model.coef
    else:
        head_path = os.path.join(ckpdir, "")
        log_path = os.path.join(args.save, "")
        coef = ddp_model.module.image_encoder.coef


    best_unsup_loss = float('inf')
    bad_epochs = 0
    best_coef = None
    history = []

    if is_main_process():
        print(f"=> Starting unsupervised on {tgt_dataset} vs {ctr_dataset}")
        if os.path.exists(log_path):
            history = json.load(open(log_path))

    for epoch in range(args.epoch):
        ddp_tgt.sampler.set_epoch(epoch)
        ddp_ctr.sampler.set_epoch(epoch)
        ctr_iter = iter(ddp_ctr)

        epoch_loss = 0.0
        batches = 0

        for i, batch in enumerate(ddp_tgt):
            ctr_batch = next(ctr_iter)
            batch = maybe_dictionarize(batch)
            ctr_batch = maybe_dictionarize(ctr_batch)
            imgs = torch.cat([batch["images"], ctr_batch["images"]], dim=0).cuda()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = ddp_model(imgs, [len(batch["images"]), len(ctr_batch["images"])])
                logits_tgt, logits_ctr = logits
                ent_tgt = softmax_entropy(logits_tgt)
                ent_ctr = softmax_entropy(logits_ctr)
                loss = -ent_tgt + control_weight * ent_ctr
                loss = loss + lp_reg(coef, args.lp_reg)
                loss = loss / args.num_grad_accumulation


            scaler.scale(loss).backward()
            if (i + 1) % args.num_grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            batches += 1

            if batches % print_every == 0 and is_main_process():
                print(f"Epoch {epoch}, batch {batches}/{num_batches}, "
                      f"ent_tgt={ent_tgt.item():.4f}, ent_ctr={ent_ctr.item():.4f}")

        avg_loss = epoch_loss / batches
        history.append({"epoch": epoch, "unsup_loss": avg_loss})

        if is_main_process():
            print(f"Epoch {epoch} completed. avg_unsup_loss={avg_loss:.4f}")

        if avg_loss < best_unsup_loss:
            best_unsup_loss = avg_loss
            best_coef = coef.data.clone()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if is_main_process():
                    print(f"Early stopping at epoch {epoch} "
                          f"(no improvement for {bad_epochs} epochs)")
                break

    if is_main_process():
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        if linearized:
            ddp_model.module.image_encoder.model.coef = torch.nn.Parameter(best_coef)
        else:
            ddp_model.module.image_encoder.coef = torch.nn.Parameter(best_coef)
        new_params = ddp_model.module.image_encoder.get_optimized_params(coef=best_coef)
        torch.save(new_params, head_path)
        print(f"Saved learned task-vector head to {head_path}")


        encoder = ddp_model.module.image_encoder
        tgt_test_set = tgt_dataset.replace("Val", "")
        ctr_test_set = ctr_dataset.replace("Val", "")
        tgt_acc = eval_single_dataset(encoder, tgt_test_set, args)["top1"]
        ctr_acc = eval_single_dataset(encoder, ctr_test_set, args)["top1"]
        print(f"Post-unsupervised accuracy on {tgt_test_set}: {100*tgt_acc:.2f}% | "
              f"{ctr_test_set}: {100*ctr_acc:.2f}%")

    cleanup_ddp()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_arguments()

    if not hasattr(args, "patience"):
        args.patience = 20
    if not hasattr(args, "control_weight"):
        args.control_weight = 2

    args.batch_size = 64 if args.model == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.model == "ViT-L-14" else 1
    args.print_every = 100
    args.ctr_dataset = "ImageNetVal"
    args.save = f"/data/atlas/checkpoints/{args.model}"

    with open(f"/data/atlas/checkpoints/ViT-B-32/zeroshot_accuracies.json") as f:
        args.zs_acc = json.load(f)

    datasets = {
        "Cars": [20, 5],
        "DTD": [20, 10],
        "EuroSAT": [7, 5],
        "GTSRB": [10, 5],
        "MNIST": [15, 3],
        "RESISC45": [20, 2],
        "SUN397": [20, 3],
        "SVHN": [5, 5],
    }

    for dataset, (epochs, lr_mul) in datasets.items():
        args.tgt_dataset = dataset + "Val"
        args.epoch = epochs
        args.lr_multiplier = lr_mul
        print("=" * 80)
        print(f"Unsupervised learning + eval on {args.model} / {dataset}")
        print("=" * 80)
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size)
