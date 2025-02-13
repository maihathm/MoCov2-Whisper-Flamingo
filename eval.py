import os
import torch
import argparse
import editdistance
import pytorch_lightning as pl
from tqdm import tqdm

from config import get_config
from datamodule.data_module import DataModule
from models.av_net import AVNet
from train import AVSRModule


def compute_metrics(predictions, targets):
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER)
    """
    total_wer, total_cer = 0, 0
    total_words, total_chars = 0, 0
    
    for pred, target in zip(predictions, targets):
        # Convert to words
        pred_words = pred.split()
        target_words = target.split()
        
        # Compute WER
        wer = editdistance.eval(pred_words, target_words)
        total_wer += wer
        total_words += len(target_words)
        
        # Compute CER
        cer = editdistance.eval(pred, target)
        total_cer += cer
        total_chars += len(target)
    
    wer = (total_wer / total_words) * 100 if total_words > 0 else 0
    cer = (total_cer / total_chars) * 100 if total_chars > 0 else 0
    
    return wer, cer


def evaluate_model(model, dataloader, device, modality="AV"):
    """
    Evaluate model on given dataloader
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {modality} model"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Get model predictions
            predictions, prediction_lengths = model.model.inference(
                (batch["audio"], batch["audio_mask"], batch["video"], batch["video_lengths"]),
                device,
                Lambda=model.config["model"].get("lambda", 0.6),
                beamWidth=model.config["model"].get("beam_width", 5)
            )
            
            # Convert predictions and targets to text
            for pred, pred_len, target, target_len in zip(
                predictions, prediction_lengths,
                batch["targets"], batch["target_lengths"]
            ):
                pred_text = model.tokenizer.decode(pred[:pred_len].cpu().numpy())
                target_text = model.tokenizer.decode(target[:target_len].cpu().numpy())
                
                all_predictions.append(pred_text)
                all_targets.append(target_text)
    
    # Compute metrics
    wer, cer = compute_metrics(all_predictions, all_targets)
    
    return {
        "wer": wer,
        "cer": cer,
        "predictions": all_predictions,
        "targets": all_targets
    }


def ablation_study(model, dataloader, device):
    """
    Perform ablation study testing different modalities and fusion methods
    """
    results = {}
    
    # Test audio-only
    model.model.modal = "AO"
    results["audio_only"] = evaluate_model(model, dataloader, device, "Audio-only")
    
    # Test video-only
    model.model.modal = "VO"
    results["video_only"] = evaluate_model(model, dataloader, device, "Video-only")
    
    # Test audio-visual with gate cross attention
    model.model.modal = "AV"
    results["av_gate"] = evaluate_model(model, dataloader, device, "AV-Gate")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AVSR model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--ablation", action="store_true", help="Perform ablation study")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Initialize data module
    data_module = DataModule(config)
    data_module.setup("test")
    test_dataloader = data_module.test_dataloader()
    
    # Load model from checkpoint
    model = AVSRModule.load_from_checkpoint(
        args.checkpoint,
        config=config,
        strict=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.ablation:
        # Perform ablation study
        results = ablation_study(model, test_dataloader, device)
        
        # Print and save results
        print("\nAblation Study Results:")
        for modality, metrics in results.items():
            print(f"\n{modality}:")
            print(f"WER: {metrics['wer']:.2f}%")
            print(f"CER: {metrics['cer']:.2f}%")
            
            # Save detailed results
            with open(os.path.join(args.output, f"{modality}_results.txt"), "w") as f:
                for pred, target in zip(metrics["predictions"], metrics["targets"]):
                    f.write(f"Pred: {pred}\n")
                    f.write(f"Target: {target}\n")
                    f.write("-" * 50 + "\n")
    else:
        # Evaluate full model
        results = evaluate_model(model, test_dataloader, device)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"WER: {results['wer']:.2f}%")
        print(f"CER: {results['cer']:.2f}%")
        
        # Save detailed results
        with open(os.path.join(args.output, "evaluation_results.txt"), "w") as f:
            f.write(f"Overall WER: {results['wer']:.2f}%\n")
            f.write(f"Overall CER: {results['cer']:.2f}%\n\n")
            f.write("Detailed Results:\n")
            f.write("=" * 50 + "\n")
            for pred, target in zip(results["predictions"], results["targets"]):
                f.write(f"Pred: {pred}\n")
                f.write(f"Target: {target}\n")
                f.write("-" * 50 + "\n")


if __name__ == "__main__":
    main()
