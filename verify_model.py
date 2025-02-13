import os
import torch
import numpy as np
from tqdm import tqdm
from models.av_net import AVNet
from config import get_config
from datamodule.data_module import DataModule

def test_model_stability(model, dataloader, max_batches=10):
    """Test model stability with different input configurations"""
    print("\nTesting model stability...")
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Test different modalities
        for modality in ["AV", "AO", "VO"]:
            print(f"\nTesting {modality} modality:")
            model.modal = modality
            
            for i, batch in enumerate(dataloader):
                if i >= max_batches:
                    break
                    
                try:
                    # Move batch to device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    input_data = (
                        batch["audio"],
                        batch["audio_mask"],
                        batch["video"],
                        batch["video_lengths"]
                    )
                    output = model(input_data)
                    
                    # Check for NaN/Inf values
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        print(f"Warning: Found NaN/Inf in output for batch {i}")
                        return False
                        
                    print(f"Batch {i+1}: OK")
                    
                except Exception as e:
                    print(f"Error processing batch {i}: {str(e)}")
                    return False
                    
            print(f"{modality} modality test passed")
    
    return True

def test_memory_usage(model, dataloader, max_batches=10):
    """Test memory usage during forward and backward passes"""
    print("\nTesting memory usage...")
    model.train()
    device = next(model.parameters()).device
    
    # Track peak memory usage
    peak_memory = 0
    
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
            
        try:
            # Clear cache
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            input_data = (
                batch["audio"],
                batch["audio_mask"],
                batch["video"],
                batch["video_lengths"]
            )
            output = model(input_data, batch["targets"], batch["target_lengths"])
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.size(-1)),
                batch["targets"].view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Track memory
            current_memory = torch.cuda.memory_allocated()
            peak_memory = max(peak_memory, current_memory)
            
            memory_mb = current_memory / 1024 / 1024
            print(f"Batch {i+1}: Memory usage = {memory_mb:.2f} MB")
            
        except Exception as e:
            print(f"Error in memory test batch {i}: {str(e)}")
            return False
            
        # Clear gradients
        model.zero_grad()
    
    print(f"Peak memory usage: {peak_memory/1024/1024:.2f} MB")
    return True

def test_input_shapes(model):
    """Test model with different input shapes"""
    print("\nTesting input shape handling...")
    device = next(model.parameters()).device
    model.eval()
    
    test_shapes = [
        # (batch_size, audio_len, video_len)
        (1, 500, 50),
        (2, 1000, 100),
        (4, 750, 75)
    ]
    
    with torch.no_grad():
        for batch_size, audio_len, video_len in test_shapes:
            try:
                # Create dummy inputs
                audio = torch.randn(batch_size, audio_len, 80).to(device)
                audio_mask = torch.ones(batch_size, audio_len, dtype=torch.bool).to(device)
                video = torch.randn(batch_size, video_len, 3, 224, 224).to(device)
                video_len = torch.full((batch_size,), video_len, dtype=torch.long).to(device)
                
                # Forward pass
                output = model((audio, audio_mask, video, video_len))
                
                print(f"Shape test passed: batch_size={batch_size}, "
                      f"audio_len={audio_len}, video_len={video_len}")
                
            except Exception as e:
                print(f"Error with shape: {(batch_size, audio_len, video_len)}")
                print(f"Error message: {str(e)}")
                return False
    
    return True

def main():
    # Load configuration
    config = get_config()
    
    # Initialize model
    model_args = (
        config["model"]["d_model"],
        config["model"]["n_heads"],
        config["model"]["n_layers"],
        config["model"]["pe_max_len"],
        config["model"]["fc_hidden_size"],
        config["model"]["dropout"],
        config["model"]["num_classes"]
    )
    
    model = AVNet(
        modal="AV",
        MoCofile=os.path.join(os.getcwd(), config["data"]["moco_file"]),
        reqInpLen=config["model"]["required_input_length"],
        modelargs=model_args
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Initialize data module
    data_module = DataModule(config)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    
    # Run tests
    print("Starting model verification...")
    
    # Test model stability
    if not test_model_stability(model, train_loader):
        print("Model stability test failed!")
        return
    
    # Test memory usage
    if not test_memory_usage(model, train_loader):
        print("Memory usage test failed!")
        return
    
    # Test input shapes
    if not test_input_shapes(model):
        print("Input shape test failed!")
        return
    
    print("\nAll verification tests passed successfully!")
    print("Model is ready for training.")

if __name__ == "__main__":
    main()
