
from __future__ import annotations

import argparse
import os
import random
import shutil
import time
from typing import Any

import numpy as np
import torch

import util
from engine import trainer1


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {device_name} to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def first_adjacency(supports) -> torch.Tensor:
    adjacency = supports[0] if isinstance(supports, (list, tuple)) else supports
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"The adjacency must be square [N,N], received {tuple(adjacency.shape)}"
        )
    return adjacency


def prepare_batch(x, y, device: torch.device):
    model_input = torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
    target = torch.as_tensor(y, dtype=torch.float32, device=device).transpose(1, 3)
    target = target[:, 0, :, :]  # [B,N,H]
    return model_input, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/ChengDu_City")
    parser.add_argument(
        "--adjdata", type=str, default="data/ChengDu_City/adj_mat.pkl"
    )
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--seq_length", type=int, default=12)
    parser.add_argument("--nhid", type=int, default=32)
    parser.add_argument("--in_dim", type=int, default=1)
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=0,
        help="0 means infer the node count from the adjacency matrix",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--force", type=str2bool, default=False)
    parser.add_argument("--save", type=str, default="./garage/ChengDu_City")
    parser.add_argument("--expid", type=int, default=120)
    parser.add_argument("--decay", type=float, default=0.92)
    parser.add_argument("--CL", type=str2bool, default=True)
    parser.add_argument("--l", type=int, default=3)
    parser.add_argument("--sample_ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--lambda_cl", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--smoke_test", type=str2bool, default=True)
    return parser.parse_args()


def run_smoke_test(engine, dataloader, args, device):
    try:
        x, y = next(dataloader["train_loader"].get_iterator())
    except StopIteration as exc:
        raise RuntimeError("Training data loader is empty") from exc

    model_input, target = prepare_batch(x, y, device)
    model_input = model_input[: min(2, model_input.shape[0])]
    target = target[: model_input.shape[0]]

    was_training = engine.model.training
    engine.model.eval()
    with torch.no_grad():
        prediction, contrastive_loss, dynamic_adjacency = engine.model(model_input)
    engine.model.train(was_training)

    expected_prediction = (
        model_input.shape[0],
        args.seq_length,
        args.num_nodes,
        1,
    )
    if tuple(prediction.shape) != expected_prediction:
        raise RuntimeError(
            f"Prediction shape {tuple(prediction.shape)} != {expected_prediction}"
        )
    if tuple(target.shape) != (
        model_input.shape[0],
        args.num_nodes,
        args.seq_length,
    ):
        raise RuntimeError(f"Unexpected target shape: {tuple(target.shape)}")
    if tuple(dynamic_adjacency.shape) != (args.num_nodes, args.num_nodes):
        raise RuntimeError(
            f"Unexpected dynamic adjacency: {tuple(dynamic_adjacency.shape)}"
        )
    if not torch.isfinite(prediction).all() or not torch.isfinite(contrastive_loss):
        raise FloatingPointError("Smoke test produced NaN or Inf")

    print(
        "Smoke test passed | "
        f"input={tuple(model_input.shape)}, "
        f"prediction={tuple(prediction.shape)}, "
        f"target={tuple(target.shape)}, "
        f"A_dynamic={tuple(dynamic_adjacency.shape)}, "
        f"L_cl={float(contrastive_loss):.6f}"
    )


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)
    device = resolve_device(args.device)

    supports = util.load_adj(args.adjdata, args.adjtype)
    adjacency = first_adjacency(supports)
    inferred_nodes = int(adjacency.shape[0])

    if args.num_nodes == 0:
        args.num_nodes = inferred_nodes
    elif args.num_nodes != inferred_nodes:
        raise ValueError(
            f"--num_nodes={args.num_nodes}, but adjacency contains {inferred_nodes} nodes"
        )

    dataloader = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size,
    )

    print(args)
    print(f"device: {device}")
    print(f"number of nodes: {args.num_nodes}")

    engine = trainer1(
        args.in_dim,
        args.seq_length,
        args.num_nodes,
        args.nhid,
        args.CL,
        args.l,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        supports,
        args.decay,
        sample_ratio=args.sample_ratio,
        lambda_cl=args.lambda_cl,
    )

    params_path = os.path.join(args.save, "DASTGCN")
    if os.path.exists(params_path):
        if not args.force:
            raise SystemExit(
                f"Params folder exists: {params_path}. Use --force true to replace it."
            )
        shutil.rmtree(params_path)
    os.makedirs(params_path, exist_ok=True)

    if args.smoke_test:
        run_smoke_test(engine, dataloader, args, device)

    print("Start training...", flush=True)

    train_history = []
    valid_history = []
    train_times = []
    valid_times = []

    best_valid_loss = float("inf")
    best_epoch = 0
    no_improvement = 0
    best_path = os.path.join(params_path, "DASTGCN_best.pth")

    for epoch in range(1, args.epochs + 1):
        dataloader["train_loader"].shuffle()
        train_metrics = []
        train_start = time.time()

        for iteration, (x, y) in enumerate(
            dataloader["train_loader"].get_iterator()
        ):
            train_x, train_y = prepare_batch(x, y, device)
            metrics = engine.train(train_x, train_y)
            train_metrics.append(metrics)

            if args.print_every > 0 and iteration % args.print_every == 0:
                print(
                    f"Iter {iteration:03d} | "
                    f"Loss {metrics[0]:.4f} | "
                    f"MAE {metrics[1]:.4f} | "
                    f"MAPE {metrics[2]:.4f} | "
                    f"RMSE {metrics[3]:.4f}",
                    flush=True,
                )

        train_times.append(time.time() - train_start)

        valid_metrics = []
        valid_start = time.time()
        for x, y in dataloader["val_loader"].get_iterator():
            valid_x, valid_y = prepare_batch(x, y, device)
            valid_metrics.append(engine.eval(valid_x, valid_y))
        valid_times.append(time.time() - valid_start)

        train_mean = np.asarray(train_metrics, dtype=np.float64).mean(axis=0)
        valid_mean = np.asarray(valid_metrics, dtype=np.float64).mean(axis=0)
        train_history.append(float(train_mean[0]))
        valid_history.append(float(valid_mean[0]))

        gumbel_tau = engine.model.anneal_gumbel_temperature()
        engine.scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_mean[0]:.4f}, MAE {train_mean[1]:.4f}, "
            f"MAPE {train_mean[2]:.4f}, RMSE {train_mean[3]:.4f} | "
            f"Valid Loss {valid_mean[0]:.4f}, MAE {valid_mean[1]:.4f}, "
            f"MAPE {valid_mean[2]:.4f}, RMSE {valid_mean[3]:.4f} | "
            f"Tau {gumbel_tau:.4f} | "
            f"LR {engine.optimizer.param_groups[0]['lr']:.8f}",
            flush=True,
        )

        if valid_mean[0] < best_valid_loss:
            best_valid_loss = float(valid_mean[0])
            best_epoch = epoch
            no_improvement = 0
            torch.save(engine.model.state_dict(), best_path)
        else:
            no_improvement += 1

        if args.patience > 0 and no_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch: {best_epoch}")
            break

    np.savetxt(
        os.path.join(params_path, "DASTGCN_his_train_loss.csv"),
        train_history,
        delimiter=",",
    )
    np.savetxt(
        os.path.join(params_path, "DASTGCN_his_loss.csv"),
        valid_history,
        delimiter=",",
    )

    print(f"Average training time: {np.mean(train_times):.4f} sec/epoch")
    print(f"Average validation time: {np.mean(valid_times):.4f} sec")
    print(f"Best validation loss: {best_valid_loss:.4f} at epoch {best_epoch}")

    engine.model.load_state_dict(torch.load(best_path, map_location=device))
    engine.model.eval()

    outputs = []
    parameter_adj = None
    with torch.no_grad():
        for x, _ in dataloader["test_loader"].get_iterator():
            test_x = torch.as_tensor(
                x, dtype=torch.float32, device=device
            ).transpose(1, 3)
            prediction_4d, _, parameter_adj = engine.model(test_x)
            outputs.append(prediction_4d[..., 0].transpose(1, 2))  # [B,N,H]

    prediction = torch.cat(outputs, dim=0)
    real_y = torch.as_tensor(
        dataloader["y_test"], dtype=torch.float32, device=device
    ).transpose(1, 3)[:, 0, :, :]
    prediction = prediction[: real_y.shape[0]]

    horizon = min(prediction.shape[-1], real_y.shape[-1])
    test_metrics = []
    for horizon_index in range(horizon):
        values = util.metric(
            prediction[:, :, horizon_index],
            real_y[:, :, horizon_index],
        )
        test_metrics.append(values)
        print(
            f"Horizon {horizon_index + 1:02d} | "
            f"MAE {values[0]:.4f} | MAPE {values[1]:.4f} | "
            f"RMSE {values[2]:.4f} | SVAR {values[3]:.4f}"
        )

    test_mean = np.asarray(test_metrics, dtype=np.float64).mean(axis=0)
    print(
        f"Test mean | MAE {test_mean[0]:.4f} | MAPE {test_mean[1]:.4f} | "
        f"RMSE {test_mean[2]:.4f} | SVAR {test_mean[3]:.4f}"
    )

    final_path = os.path.join(
        params_path, f"DASTGCN_exp{args.expid}_best_{best_epoch}.pth"
    )
    torch.save(engine.model.state_dict(), final_path)

    np.savez_compressed(
        os.path.join(params_path, "DASTGCN_prediction_results.npz"),
        prediction=prediction.cpu().numpy(),
        parameter_adj=parameter_adj.cpu().numpy(),
        ground_truth=real_y.cpu().numpy(),
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total time spent: {time.time() - start_time:.4f} sec")
