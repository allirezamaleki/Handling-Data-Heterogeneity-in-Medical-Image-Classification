from data_loader import count_centers, prepare_federated_loaders
from federated import run_federated
from datasets import load_dataset
import torch, os, random, numpy as np

LOCAL_EPOCHS = 5
ROUNDS       = 15
LR           = 0.01
SEED = 42

def set_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # required by torch.use_deterministic_algorithms on CUDA
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass



def main():
    set_seed()
    num_clients = count_centers()
    print(f"Found {num_clients} centers")
    train_ldrs, test_ldrs, glob_test = prepare_federated_loaders(num_clients)

    ds_train = load_dataset("flwrlabs/fed-isic2019", split="train")
    num_classes = len(ds_train.features["label"].names)

    run_federated(
        train_ldrs, test_ldrs, glob_test,
        num_clients, num_classes,
        rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LR
    )

if __name__ == "__main__":
    main()
