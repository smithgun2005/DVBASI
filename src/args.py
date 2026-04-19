
import argparse
import os

import torch


def int_or_float(value):
    if "." in value:
        return float(value)
    return int(value)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("/root/autodl-tmp/atlas-main/data/"),
        help="The root directory for the datasets.",
    )

    parser.add_argument(
        "--target-dataset",
        default=None,
        type=str,
        help="Single target dataset used for single-task aTLAS / SVD experiments.",
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=8,
        help="Number of SVD atoms extracted from a task vector.",
    )

    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. ",
    )
    parser.add_argument(
        "--eval-on-full",
        default=False,
        action="store_true",
        help="Evaluate on the full dataset, when the model is trained on one class.",
    )
    parser.add_argument(
        "--loss-fn",
        default="entropy",
        type=str,
        help="Loss function to use.",
        choices=["entropy", "cross_entropy"],
    )
    parser.add_argument(
        "--lp-reg",
        default=None,
        type=int,
        choices=[1, 2],
        help="Regularisation applied to the learned coefficients.",
    )
    parser.add_argument(
        "--blockwise-coef",
        default=False,
        action="store_true",
        help="Use different coefficients on different parameter blocks.",
    )
    parser.add_argument(
        "--unsupervised",
        default=False,
        action="store_true",
        help="Use unsupervised softmax entropy loss instead of cross-entropy. "
             "When enabled, both aTLAS and DVBASI optimization use softmax_entropy.",
    )
    parser.add_argument(
        "--subsample",
        default=1.0,
        type=int_or_float,
        help="Subsample the datasets with a float or specify the number of shots with an integer.",
    )
    parser.add_argument(
        "--control-threshold",
        default=0.95,
        type=float,
        help="Percentage of accuracy on the control dataset to maintain.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.",
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument("--batch-size", type=int, default=128)


    # ---------------- training ----------------
    parser.add_argument(
        "--num-grad-accumulation",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.5, help="Weight decay")
    parser.add_argument("--ls", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument("--warmup_length", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Where to load zero-shot weights and task vectors",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="/root/autodl-tmp/atlas-main/src/results/",
        help="Where to save results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of processes for distributed training.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=-1,
        help="How often to checkpoint the model.",
    )
    parser.add_argument("--port", type=int, default=12355, help="Port for distributed training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Adapter trained with aTLAS",
        choices=["tip", "lpp", "tip_cot"],
    )
    parser.add_argument(
        "--finetuning-mode",
        default="standard",
        choices=["standard", "linear", "posthoc", "none"],
        help="Whether to use linearized models or not.",
    )


    parser.add_argument(
        "--op-n",
        type=int,
        default=1,
        help="Number of chained opLift factors on BOTH left/right sides.",
    )

    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=21,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=None,
        help="Run atlas x K where the task vectors are randomly partitioned n times (few-shot only)",
    )
    parser.add_argument(
        "--dv-max-iters",
        type=int,
        default=1,
        help="Maximum number of DV-BASI outer iterations.",
    )
    parser.add_argument(
        "--dv-patience",
        type=int,
        default=3,
        help="Stop DV-BASI outer iterations after this many non-improving iterations.",
    )
    parser.add_argument(
        "--dv-min-improve",
        type=float,
        default=0.0,
        help="Minimum improvement threshold for resetting DV-BASI patience.",
    )
    parser.add_argument(
        "--dv-pretrained-path",
        type=str,
        default=None,
        help="Checkpoint path used to extract pretrained parameter list for difference-vector construction.",
    )
    parser.add_argument(
        "--dv-initial-optima-path",
        type=str,
        default=None,
        help="Checkpoint path of initial optima (aTLAS result). If omitted, auto-detect under --save.",
    )
    parser.add_argument(
        "--dv-output-dir",
        type=str,
        default=None,
        help="Directory to store per-iteration DV-BASI checkpoints and logs.",
    )
    parser.add_argument(
        "--dv-auto-atlas",
        action="store_true",
        help="Automatically run task_addition_atlas before DV-BASI iterations.",
    )
    parser.add_argument(
        "--dv-auto-atlas-force",
        action="store_true",
        help="Force rerunning task_addition_atlas even when atlas baseline checkpoint already exists.",
    )
    parser.add_argument(
        "--dv-atlas-epochs",
        type=int,
        default=50,
        help="Epochs used when auto-running task_addition_atlas (default: use --epochs).",
    )
    parser.add_argument(
        "--dv-init-coef",
        type=float,
        default=0.1,
        help=(
            "Initial value for the difference-vector coefficient α at the start of each "
            "DV-BASI inner iteration. α=0 means the search starts exactly at the current "
            "optimum; a small positive value (e.g. 0.1) gives an initial perturbation along "
            "the difference-vector direction, which can help escape flat loss landscapes. "
            "Corresponds to the initial perturbation ε described in the paper."
        ),
    )

    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]

    return parsed_args
