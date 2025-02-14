import logging
import sys
import torch

def setup_logging(level=logging.DEBUG):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def log_tensor_info(name, tensor):
    """Helper function to log tensor information"""
    if isinstance(tensor, torch.Tensor):
        logger = logging.getLogger('tensor_debug')
        shape_str = 'x'.join(str(dim) for dim in tensor.shape)
        logger.debug(f"{name} - Shape: {shape_str}, Type: {tensor.dtype}, Device: {tensor.device}")
        if tensor.numel() > 0:
            logger.debug(f"{name} - Range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")
    else:
        logger = logging.getLogger('tensor_debug')
        logger.debug(f"{name} - Not a tensor. Type: {type(tensor)}")
