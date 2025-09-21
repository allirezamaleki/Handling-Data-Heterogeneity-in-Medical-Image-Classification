import torch
import torch.nn as nn
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vit_base(num_classes: int) -> nn.Module:

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model.to(DEVICE)

def cait_base(num_classes: int) -> nn.Module:
    model = timm.create_model('cait_s24_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model.to(DEVICE)

def swin_base(num_classes: int) -> nn.Module:
    model = timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=True,
        num_classes=num_classes
    )
    return model.to(DEVICE)
