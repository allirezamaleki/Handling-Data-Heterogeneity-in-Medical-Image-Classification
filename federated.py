from tqdm.auto import tqdm
from model import vit_base, cait_base, swin_base
from train import train_local, evaluate
from typing import List, Dict
import torch


def _zero_like_control_variates(model) -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(param) for name, param in model.named_parameters()}


def _get_param_state(model) -> Dict[str, torch.Tensor]:
    # Return a detached clone of parameter tensors only (exclude buffers)
    return {name: p.detach().clone() for name, p in model.named_parameters()}

def fedavg(updates: List[Dict], weights: List[float]) -> Dict:
    if abs(sum(weights) - 1.0) > 1e-6:
        weights = [w / sum(weights) for w in weights]
    agg: Dict = {}
    for k in updates[0]:
        agg[k] = sum(w * u[k] for u, w in zip(updates, weights))
    return agg


def run_federated(
    train_loaders,
    test_loaders,
    global_test_loader,
    num_clients: int,
    num_classes: int,
    rounds: int,
    local_epochs: int,
    lr: float
):
    global_model = vit_base(num_classes)
    # Initialize SCAFFOLD control variates: global c and per-client c_i
    global_c = _zero_like_control_variates(global_model)
    client_c_list: List[Dict[str, torch.Tensor]] = [
        {name: tensor.clone() for name, tensor in global_c.items()} for _ in range(num_clients)
    ]

    for rnd in tqdm(range(1, rounds + 1), desc="Round"):
        print(f"\nRound {rnd}/{rounds}")

        local_states  = []
        client_weights = []
        delta_c_list: List[Dict[str, torch.Tensor]] = []

        # iterate over clients
        for cid in range(num_clients):
            print(f"\n>> Client {cid}")
            # start from the current global
            local_model = vit_base(num_classes)
            local_model.load_state_dict(global_model.state_dict())

            # cache starting params and client control variate
            start_params = _get_param_state(local_model)
            client_c = client_c_list[cid]
            total_local_steps = 0

            # train + eval one epoch at a time
            for epoch in range(1, local_epochs + 1):
                # train for exactly 1 epoch
                state_dict, [train_loss], [train_acc], steps = train_local(
                    local_model,
                    train_loaders[cid],
                    epochs=1,   # one epoch per call
                    lr=lr,
                    c_global=global_c,
                    c_client=client_c,
                )
                total_local_steps += steps

                # immediately evaluate on that client's test split
                test_metrics = evaluate(local_model, test_loaders[cid], num_classes)

                print(
                    f"  Epoch {epoch:>2} → "
                    f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:>5.2f}%  |  "
                    f"Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:>5.2f}%"
                )
                print(
                    f"  Test Metrics → "
                    f"F1: {test_metrics['f1_score']:.4f}, "
                    f"Precision: {test_metrics['precision']:.4f}, "
                    f"Recall: {test_metrics['recall']:.4f}, "
                    f"AUC: {test_metrics['auc']:.4f}"
                )

            # after all epochs, collect for aggregation
            local_states.append(local_model.state_dict())
            client_weights.append(len(train_loaders[cid].dataset))

            # compute SCAFFOLD client control variate update
            end_params = _get_param_state(local_model)
            if total_local_steps > 0:
                inv_lr_T = 1.0 / (lr * float(total_local_steps))
                new_client_c: Dict[str, torch.Tensor] = {}
                delta_c: Dict[str, torch.Tensor] = {}
                for name in client_c.keys():
                    # c_i' = c_i - c + (w_start - w_end)/(eta * T)
                    ci_prime = client_c[name] - global_c[name] + (start_params[name] - end_params[name]) * inv_lr_T
                    new_client_c[name] = ci_prime
                    delta_c[name] = ci_prime - client_c[name]
                # store delta for global c update and commit client's new c_i
                delta_c_list.append(delta_c)
                client_c_list[cid] = new_client_c
            else:
                # no steps -> no update to c_i
                delta_c_list.append({name: torch.zeros_like(t) for name, t in client_c.items()})

        # aggregate into new global
        # normalize client_weights
        total_samples = sum(client_weights)
        norm_weights  = [w / total_samples for w in client_weights]
        new_state = fedavg(local_states, norm_weights)
        global_model.load_state_dict(new_state)

        # Update global control variate: c <- c + sum_i p_i * (c_i' - c_i)
        # Use the same normalization weights as FedAvg
        for name in global_c.keys():
            update = sum(w * dc[name] for dc, w in zip(delta_c_list, norm_weights))
            global_c[name] = global_c[name] + update

        # final global evaluation
        global_metrics = evaluate(global_model, global_test_loader, num_classes)
        print(f"\n→ Global Results →")
        print(f"  Loss: {global_metrics['loss']:.4f}, Accuracy: {global_metrics['accuracy']:.2f}%")
        print(f"  F1: {global_metrics['f1_score']:.4f}, Precision: {global_metrics['precision']:.4f}")
        print(f"  Recall: {global_metrics['recall']:.4f}, AUC: {global_metrics['auc']:.4f}")
