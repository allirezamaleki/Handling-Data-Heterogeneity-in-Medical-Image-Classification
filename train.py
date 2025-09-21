import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torchmetrics import F1Score, Precision, Recall, AUROC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_local(model, loader, epochs, lr, c_global=None, c_client=None):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    losses, accs = [], []

    use_scaffold = c_global is not None and c_client is not None

    num_steps = 0
    for _ in range(epochs):
        tot_loss, corr, tot = 0.0, 0, 0
        for batch in loader:
            imgs  = batch["pixel_values"].to(DEVICE)
            lbls  = batch["labels"].to(DEVICE)
            opt.zero_grad()
            out   = model(imgs)
            loss  = crit(out, lbls)
            loss.backward()

            if use_scaffold:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        param.grad.add_(c_global[name] - c_client[name])

            opt.step()

            tot_loss += loss.item() * lbls.size(0)
            preds    = out.argmax(dim=1)
            corr    += (preds == lbls).sum().item()
            tot     += lbls.size(0)
            num_steps += 1

        losses.append(tot_loss / tot)
        accs.append(100 * corr / tot)
    return model.state_dict(), losses, accs, num_steps

def evaluate(model, loader, num_classes):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss, corr, tot = 0.0, 0, 0
    
    # Initialize metrics
    f1 = F1Score(task='multiclass', num_classes=num_classes).to(DEVICE)
    precision = Precision(task='multiclass', num_classes=num_classes).to(DEVICE)
    recall = Recall(task='multiclass', num_classes=num_classes).to(DEVICE)
    auroc = AUROC(task='multiclass', num_classes=num_classes).to(DEVICE)
    
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for batch in loader:
            imgs  = batch["pixel_values"].to(DEVICE)
            lbls  = batch["labels"].to(DEVICE)
            out   = model(imgs)
            probs = torch.softmax(out, dim=1)
            
            tot_loss += crit(out, lbls).item() * lbls.size(0)
            preds    = out.argmax(dim=1)
            corr    += (preds == lbls).sum().item()
            tot     += lbls.size(0)
            
            all_preds.append(preds)
            all_labels.append(lbls)
            all_probs.append(probs)
    
    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Calculate metrics
    f1_score = f1(all_preds, all_labels).item()
    prec = precision(all_preds, all_labels).item()
    rec = recall(all_preds, all_labels).item()
    auc = auroc(all_probs, all_labels).item()
    
    return {
        'loss': tot_loss / tot,
        'accuracy': 100 * corr / tot,
        'f1_score': f1_score,
        'precision': prec,
        'recall': rec,
        'auc': auc
    }
