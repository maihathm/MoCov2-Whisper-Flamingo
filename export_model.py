import os
import argparse
import torch
import onnx
import numpy as np
from models.av_net import AVNet
from config import get_config

def validate_onnx_model(onnx_path):
    """Validate exported ONNX model"""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("ONNX model validation successful!")

def export_to_onnx(model, save_path, input_shapes):
    """Export PyTorch model to ONNX format"""
    # Prepare dummy inputs
    dummy_audio = torch.randn(*input_shapes["audio"])
    dummy_audio_mask = torch.ones(*input_shapes["audio_mask"], dtype=torch.bool)
    dummy_video = torch.randn(*input_shapes["video"])
    dummy_video_len = torch.tensor([input_shapes["video"][1]], dtype=torch.long)
    
    # Set model to evaluation mode
    model.eval()
    
    # Export model
    torch.onnx.export(
        model,
        (
            (dummy_audio, dummy_audio_mask, dummy_video, dummy_video_len),
            None,  # targetinBatch
            None   # targetLenBatch
        ),
        save_path,
        export_params=True,
        opset_version=14,  # Use recent opset for better compatibility
        do_constant_folding=True,  # Optimize constant operations
        input_names=['audio', 'audio_mask', 'video', 'video_length'],
        output_names=['output'],
        dynamic_axes={
            'audio': {0: 'batch', 1: 'audio_length'},
            'audio_mask': {0: 'batch', 1: 'audio_length'},
            'video': {0: 'batch', 1: 'video_length'},
            'output': {0: 'batch', 1: 'sequence_length'}
        }
    )
    print(f"Model exported to {save_path}")

def verify_inference(model_path):
    """Verify ONNX model inference"""
    import onnxruntime as ort
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(model_path)
    
    # Get input names
    input_names = [input.name for input in session.get_inputs()]
    
    # Create dummy inputs
    dummy_inputs = {
        'audio': np.random.randn(1, 1000, 80).astype(np.float32),
        'audio_mask': np.ones((1, 1000), dtype=np.bool_),
        'video': np.random.randn(1, 100, 3, 224, 224).astype(np.float32),
        'video_length': np.array([100], dtype=np.int64)
    }
    
    # Run inference
    outputs = session.run(None, dummy_inputs)
    print("ONNX model inference verification successful!")
    return outputs[0].shape

def main():
    parser = argparse.ArgumentParser(description='Export AVSR model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='avsr_model.onnx',
                      help='Output path for ONNX model')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for export')
    args = parser.parse_args()
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Define input shapes
    input_shapes = {
        "audio": (args.batch_size, 1000, 80),  # Typical audio mel spectrogram shape
        "audio_mask": (args.batch_size, 1000),  # Audio mask shape
        "video": (args.batch_size, 100, 3, 224, 224),  # Video frames shape
        "video_mask": (args.batch_size, 100)  # Video mask shape
    }
    
    # Export model
    print("Exporting model to ONNX format...")
    export_to_onnx(model, args.output, input_shapes)
    
    # Validate exported model
    print("Validating ONNX model...")
    validate_onnx_model(args.output)
    
    # Verify inference
    print("Verifying inference...")
    output_shape = verify_inference(args.output)
    print(f"Model output shape: {output_shape}")
    
    print("\nModel export and validation complete!")

if __name__ == '__main__':
    main()
