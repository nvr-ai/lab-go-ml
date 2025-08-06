#!/usr/bin/env python3
"""
FP16 Quantization Script for ONNX Models

This script provides comprehensive FP16 quantization capabilities optimized for GPU/CPU inference.
Supports dynamic input dimensions and preserves model accuracy while reducing memory footprint
and improving inference speed.

Requirements:
- ONNX Runtime 1.16+
- ONNX 1.15+
- Target: <1% accuracy loss, ~50% memory reduction, 1.5-2x speedup
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantization_fp16.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FP16Quantizer:
    """
    Advanced FP16 quantization engine with support for dynamic shapes and model validation.
    
    This quantizer implements state-of-the-art FP16 conversion techniques optimized for
    object detection models with minimal accuracy degradation.
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        opset_version: int = 17,
        validate_model: bool = True
    ):
        """
        Initialize FP16 quantizer with model and configuration parameters.
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path for quantized output model (auto-generated if None)
            opset_version: ONNX opset version to use (17+ recommended)
            validate_model: Whether to validate model structure before quantization
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path) if output_path else self._generate_output_path()
        self.opset_version = opset_version
        self.validate_model = validate_model
        
        # Quantization statistics
        self.stats = {
            'original_size_mb': 0,
            'quantized_size_mb': 0,
            'compression_ratio': 0,
            'conversion_time_seconds': 0,
            'fp32_nodes': 0,
            'fp16_nodes': 0,
            'preserved_nodes': 0
        }
        
        # Model-specific configurations
        self.preserve_nodes = [
            # Nodes that should remain in FP32 for numerical stability
            'Softmax',
            'LogSoftmax', 
            'NormalizeL2',
            'ReduceSum',
            'ReduceMean',
            'ReduceMax',
            'ReduceMin'
        ]
        
        # Input/output node preservation for compatibility
        self.preserve_io_types = True
        
    def _generate_output_path(self) -> Path:
        """Generate output path with FP16 suffix."""
        stem = self.model_path.stem
        suffix = self.model_path.suffix
        return self.model_path.parent / f"{stem}_fp16{suffix}"
    
    def load_and_validate_model(self) -> onnx.ModelProto:
        """
        Load ONNX model and perform comprehensive validation.
        
        Returns:
            Loaded and validated ONNX model
            
        Raises:
            ValueError: If model validation fails
            FileNotFoundError: If model file doesn't exist
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        logger.info(f"Loading ONNX model from: {self.model_path}")
        
        try:
            model = onnx.load(str(self.model_path))
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {e}")
        
        # Get original model size
        self.stats['original_size_mb'] = self.model_path.stat().st_size / (1024 * 1024)
        
        if self.validate_model:
            logger.info("Validating model structure...")
            try:
                onnx.checker.check_model(model)
                logger.info("✓ Model validation passed")
            except Exception as e:
                logger.warning(f"Model validation warning: {e}")
                
        # Log model information
        self._log_model_info(model)
        
        return model
    
    def _log_model_info(self, model: onnx.ModelProto) -> None:
        """Log comprehensive model information."""
        logger.info("=== Model Information ===")
        logger.info(f"ONNX Version: {model.opset_import[0].version}")
        logger.info(f"Producer: {model.producer_name} {model.producer_version}")
        logger.info(f"Model Version: {model.model_version}")
        
        # Count node types
        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
            
        logger.info(f"Total Nodes: {len(model.graph.node)}")
        logger.info(f"Node Types: {len(node_types)}")
        
        # Log input/output shapes
        logger.info("Inputs:")
        for input_tensor in model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            logger.info(f"  {input_tensor.name}: {shape}")
            
        logger.info("Outputs:")
        for output_tensor in model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            logger.info(f"  {output_tensor.name}: {shape}")
    
    def convert_to_fp16(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Convert ONNX model to FP16 with intelligent node preservation.
        
        Args:
            model: Input ONNX model
            
        Returns:
            FP16-quantized ONNX model
        """
        logger.info("Starting FP16 conversion...")
        start_time = time.time()
        
        # Create a copy of the model for modification
        fp16_model = onnx.ModelProto()
        fp16_model.CopyFrom(model)
        
        # Convert weights and activations to FP16
        fp16_model = self._convert_weights_to_fp16(fp16_model)
        fp16_model = self._convert_activations_to_fp16(fp16_model)
        
        # Update opset version if needed
        if self.opset_version > model.opset_import[0].version:
            fp16_model.opset_import[0].version = self.opset_version
            
        # Record conversion time
        self.stats['conversion_time_seconds'] = time.time() - start_time
        
        logger.info(f"✓ FP16 conversion completed in {self.stats['conversion_time_seconds']:.2f}s")
        
        return fp16_model
    
    def _convert_weights_to_fp16(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Convert model weights/initializers to FP16."""
        logger.info("Converting weights to FP16...")
        
        fp32_count = 0
        fp16_count = 0
        
        for initializer in model.graph.initializer:
            if initializer.data_type == TensorProto.FLOAT:
                # Convert FP32 weights to FP16
                fp32_weights = numpy_helper.to_array(initializer)
                fp16_weights = fp32_weights.astype(np.float16)
                
                # Check for potential overflow/underflow
                if np.any(np.isinf(fp16_weights)) or np.any(np.isnan(fp16_weights)):
                    logger.warning(f"Numerical instability detected in {initializer.name}, keeping FP32")
                    fp32_count += 1
                    continue
                    
                # Update initializer with FP16 data
                new_initializer = numpy_helper.from_array(fp16_weights, initializer.name)
                initializer.CopyFrom(new_initializer)
                fp16_count += 1
            else:
                fp32_count += 1
                
        self.stats['fp32_nodes'] = fp32_count
        self.stats['fp16_nodes'] = fp16_count
        
        logger.info(f"Converted {fp16_count} weights to FP16, preserved {fp32_count} in FP32")
        
        return model
    
    def _convert_activations_to_fp16(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Convert activation tensors to FP16."""
        logger.info("Converting activations to FP16...")
        
        preserved_count = 0
        
        # Update value info (intermediate tensors)
        for value_info in model.graph.value_info:
            if (value_info.type.tensor_type.elem_type == TensorProto.FLOAT and
                not self._should_preserve_node(value_info.name)):
                value_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            else:
                preserved_count += 1
        
        # Update input/output types if not preserving I/O types
        if not self.preserve_io_types:
            for input_tensor in model.graph.input:
                if input_tensor.type.tensor_type.elem_type == TensorProto.FLOAT:
                    input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT16
                    
            for output_tensor in model.graph.output:
                if output_tensor.type.tensor_type.elem_type == TensorProto.FLOAT:
                    output_tensor.type.tensor_type.elem_type = TensorProto.FLOAT16
        else:
            preserved_count += len(model.graph.input) + len(model.graph.output)
            
        self.stats['preserved_nodes'] = preserved_count
        
        logger.info(f"Preserved {preserved_count} nodes in FP32 for stability")
        
        return model
    
    def _should_preserve_node(self, node_name: str) -> bool:
        """Check if a node should be preserved in FP32."""
        return any(preserve_type in node_name for preserve_type in self.preserve_nodes)
    
    def save_quantized_model(self, model: onnx.ModelProto) -> None:
        """
        Save quantized model with validation and statistics.
        
        Args:
            model: FP16-quantized ONNX model
        """
        logger.info(f"Saving quantized model to: {self.output_path}")
        
        try:
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            onnx.save(model, str(self.output_path))
            
            # Calculate file size and compression ratio
            self.stats['quantized_size_mb'] = self.output_path.stat().st_size / (1024 * 1024)
            self.stats['compression_ratio'] = (
                self.stats['original_size_mb'] / self.stats['quantized_size_mb']
                if self.stats['quantized_size_mb'] > 0 else 0
            )
            
            logger.info("✓ Model saved successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save quantized model: {e}")
    
    def validate_quantized_model(self) -> bool:
        """
        Validate the quantized model for correctness.
        
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Validating quantized model...")
        
        try:
            # Load and check the quantized model
            quantized_model = onnx.load(str(self.output_path))
            onnx.checker.check_model(quantized_model)
            
            logger.info("✓ Quantized model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Quantized model validation failed: {e}")
            return False
    
    def print_quantization_summary(self) -> None:
        """Print comprehensive quantization summary."""
        logger.info("\n" + "="*60)
        logger.info("         FP16 QUANTIZATION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Original Model Size:    {self.stats['original_size_mb']:.2f} MB")
        logger.info(f"Quantized Model Size:   {self.stats['quantized_size_mb']:.2f} MB")
        logger.info(f"Size Reduction:         {(1 - self.stats['quantized_size_mb']/self.stats['original_size_mb'])*100:.1f}%")
        logger.info(f"Compression Ratio:      {self.stats['compression_ratio']:.2f}x")
        logger.info(f"Conversion Time:        {self.stats['conversion_time_seconds']:.2f}s")
        
        logger.info("\nNode Conversion Statistics:")
        logger.info(f"  FP16 Nodes:           {self.stats['fp16_nodes']}")
        logger.info(f"  FP32 Preserved:       {self.stats['fp32_nodes']}")
        logger.info(f"  Stability Preserved:  {self.stats['preserved_nodes']}")
        
        logger.info(f"\nModel Paths:")
        logger.info(f"  Original:  {self.model_path}")
        logger.info(f"  Quantized: {self.output_path}")
        
        logger.info("="*60)
    
    def quantize(self) -> str:
        """
        Execute complete FP16 quantization pipeline.
        
        Returns:
            Path to quantized model
            
        Raises:
            RuntimeError: If quantization fails
        """
        try:
            # Load and validate original model
            model = self.load_and_validate_model()
            
            # Convert to FP16
            fp16_model = self.convert_to_fp16(model)
            
            # Save quantized model
            self.save_quantized_model(fp16_model)
            
            # Validate quantized model
            if not self.validate_quantized_model():
                raise RuntimeError("Quantized model validation failed")
            
            # Print summary
            self.print_quantization_summary()
            
            return str(self.output_path)
            
        except Exception as e:
            logger.error(f"FP16 quantization failed: {e}")
            raise RuntimeError(f"Quantization pipeline failed: {e}")


def main():
    """Main CLI entry point for FP16 quantization."""
    parser = argparse.ArgumentParser(
        description="FP16 Quantization for ONNX Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic FP16 quantization
  python quantize_fp16.py --model model.onnx
  
  # Custom output path
  python quantize_fp16.py --model model.onnx --output model_fp16.onnx
  
  # Preserve I/O types for compatibility
  python quantize_fp16.py --model model.onnx --preserve-io-types
  
  # Skip model validation (faster)
  python quantize_fp16.py --model model.onnx --no-validate
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to input ONNX model"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path for quantized output model (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version to use (default: 17)"
    )
    
    parser.add_argument(
        "--preserve-io-types",
        action="store_true",
        help="Preserve input/output tensor types as FP32 for compatibility"
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip model validation (faster but less safe)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Initialize quantizer
        quantizer = FP16Quantizer(
            model_path=args.model,
            output_path=args.output,
            opset_version=args.opset_version,
            validate_model=not args.no_validate
        )
        
        # Set I/O preservation
        quantizer.preserve_io_types = args.preserve_io_types
        
        # Execute quantization
        output_path = quantizer.quantize()
        
        logger.info(f"✓ FP16 quantization completed successfully!")
        logger.info(f"Quantized model saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ FP16 quantization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())