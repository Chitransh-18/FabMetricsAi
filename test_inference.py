import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np

# 1. Recreate the identical architecture skeleton
def build_model(num_classes: int = 9) -> nn.Module:
    weights = ResNet34_Weights.DEFAULT  # Uses local or downloaded weights
    model = resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_local_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = build_model(num_classes=9).to(device)
    
    # 1. Load the checkpoint dictionary
    checkpoint = torch.load(weights_path, map_location=device)
    
    # 2. Extract the nested model weights safely
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint unpacked successfully! (Loaded best F1: {checkpoint.get('best_val_f1', 'N/A')})")
    else:
        # Fallback if it was just raw weights
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")
        
    model.eval()
    return model, device
    
if __name__ == "__main__":
    # Update this path to where your downloaded file is stored locally
    WEIGHTS_FILE = "baseline_resnet34.pth" 
    
    try:
        model, device = load_local_model(WEIGHTS_FILE)
        
        # Create a fake wafer map tensor to test local execution pipeline
        # Shape: [Batch_size, Channels, Height, Width] -> ResNet expects 3 channels, 224x224
        dummy_wafer = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_wafer)
            predicted_class = torch.argmax(output, dim=1).item()
            
        class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']
        print(f"Inference Test Successful! Predicted dummy class: {class_names[predicted_class]}")
        
    except FileNotFoundError:
        print(f"Error: Please place your downloaded '{WEIGHTS_FILE}' file into this directory.")