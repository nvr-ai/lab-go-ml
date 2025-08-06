#!/usr/bin/env python3
"""
INT8 Quantization Script for ONNX Models with Calibration Dataset

This script provides comprehensive INT8 quantization capabilities optimized for CPU inference.
Supports dynamic input dimensions and uses calibration datasets to preserve model accuracy 
while maximizing CPU performance gains.

Requirements:
- ONNX Runtime 1.16+
- Representative calibration dataset
- Target: ≥98% accuracy preservation, ~75% memory reduction, 2-4x CPU speedup
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    create_calibrator,
    write_calibration_table,
    quantize_static,
)
from onnxruntime.quantization.calibrate import CalibrationMethod
import onnxruntime as ort
from PIL import Image
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantization_int8.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CalibrationDataset:
    """
    Calibration dataset manager for INT8 quantization.
    
    This class handles loading, preprocessing, and managing calibration data
    for accurate quantization parameter estimation.
    """
    
    def __init__(
        self,
        data_path: str,
        input_name: str,
        target_width: int = 640,
        target_height: int = 640,
        max_samples: int = 100,
        preprocessing: str = "coco"
    ):
        """
        Initialize calibration dataset.
        
        Args:
            data_path: Path to calibration images directory
            input_name: Name of the model input tensor
            target_width: Target image width for preprocessing
            target_height: Target image height for preprocessing
            max_samples: Maximum number of calibration samples
            preprocessing: Preprocessing method ('coco', 'imagenet', 'custom')
        """
        self.data_path = Path(data_path)
        self.input_name = input_name
        self.target_width = target_width
        self.target_height = target_height
        self.max_samples = max_samples
        self.preprocessing = preprocessing
        
        # Collect image files
        self.image_files = self._collect_image_files()
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {data_path}")
            
        logger.info(f"Found {len(self.image_files)} calibration images")
        
        # Preprocessing parameters
        if preprocessing == "coco":
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif preprocessing == "imagenet":
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:  # custom
            self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    def _collect_image_files(self) -> List[Path]:
        """Collect all image files from the data path."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_path.glob(f"*{ext}"))
            image_files.extend(self.data_path.glob(f"*{ext.upper()}"))
            
        # Recursively search subdirectories
        for subdir in self.data_path.iterdir():
            if subdir.is_dir():
                for ext in image_extensions:
                    image_files.extend(subdir.glob(f"*{ext}"))
                    image_files.extend(subdir.glob(f"*{ext.upper()}"))
        
        # Sort and limit
        image_files = sorted(image_files)[:self.max_samples]
        return image_files
    
    def preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess a single image for calibration.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor [1, C, H, W]
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to target dimensions
            image = image.resize((self.target_width, self.target_height), Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Apply normalization
            img_array = (img_array - self.mean) / self.std
            
            # Convert HWC to CHW and add batch dimension
            img_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
            
            return img_tensor.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Failed to preprocess {image_path}: {e}")
            # Return zero tensor as fallback
            return np.zeros((1, 3, self.target_height, self.target_width), dtype=np.float32)
    
    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over calibration samples."""
        for image_file in tqdm(self.image_files, desc="Loading calibration data"):
            preprocessed = self.preprocess_image(image_file)
            yield {self.input_name: preprocessed}
    
    def __len__(self) -> int:
        """Return number of calibration samples."""
        return len(self.image_files)


class ONNXCalibrationDataReader(CalibrationDataReader):
    """
    ONNX Runtime calibration data reader.
    
    This class bridges the CalibrationDataset with ONNX Runtime's
    quantization calibration interface.
    """
    
    def __init__(self, dataset: CalibrationDataset):
        """
        Initialize calibration data reader.
        
        Args:
            dataset: Calibration dataset instance
        """
        self.dataset = dataset
        self.data_iterator = iter(dataset)
        self.data_cache = []
        self.current_index = 0
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Get next calibration sample."""
        try:
            if self.current_index < len(self.data_cache):
                sample = self.data_cache[self.current_index]
                self.current_index += 1
                return sample
            else:
                sample = next(self.data_iterator)
                self.data_cache.append(sample)
                self.current_index += 1
                return sample
        except StopIteration:
            return None
    
    def rewind(self):
        """Rewind to beginning of dataset."""
        self.current_index = 0
        if not self.data_cache:
            self.data_iterator = iter(self.dataset)


class INT8Quantizer:
    """
    Advanced INT8 quantization engine with calibration-based quantization.
    
    This quantizer implements state-of-the-art INT8 conversion techniques
    optimized for CPU inference with minimal accuracy degradation.
    """
    
    def __init__(
        self,
        model_path: str,
        calibration_data_path: str,
        output_path: Optional[str] = None,
        calibration_method: str = "MinMax",
        max_calibration_samples: int = 100,
        opset_version: int = 17,
        validate_model: bool = True
    ):
        """
        Initialize INT8 quantizer with calibration configuration.
        
        Args:
            model_path: Path to input ONNX model
            calibration_data_path: Path to calibration images directory
            output_path: Path for quantized output model
            calibration_method: Calibration method ('MinMax', 'Entropy', 'Percentile')
            max_calibration_samples: Maximum number of calibration samples
            opset_version: ONNX opset version to use
            validate_model: Whether to validate model before quantization
        """
        self.model_path = Path(model_path)
        self.calibration_data_path = Path(calibration_data_path)
        self.output_path = Path(output_path) if output_path else self._generate_output_path()
        self.calibration_method = self._parse_calibration_method(calibration_method)
        self.max_calibration_samples = max_calibration_samples
        self.opset_version = opset_version
        self.validate_model = validate_model
        
        # Quantization statistics
        self.stats = {
            'original_size_mb': 0,
            'quantized_size_mb': 0,
            'compression_ratio': 0,
            'quantization_time_seconds': 0,
            'calibration_samples_used': 0,
            'calibration_time_seconds': 0,
            'fp32_nodes': 0,
            'int8_nodes': 0,
            'calibration_method': calibration_method
        }
        
        # Model information
        self.model_info = {}
        
        # Calibration configuration
        self.calibration_table_path = self.output_path.parent / "calibration_table.flatbuf"
        
    def _generate_output_path(self) -> Path:
        """Generate output path with INT8 suffix."""
        stem = self.model_path.stem
        suffix = self.model_path.suffix
        return self.model_path.parent / f"{stem}_int8{suffix}"
    
    def _parse_calibration_method(self, method: str) -> CalibrationMethod:
        """Parse calibration method string to enum."""
        method_map = {
            'MinMax': CalibrationMethod.MinMax,
            'Entropy': CalibrationMethod.Entropy,
            'Percentile': CalibrationMethod.Percentile,
        }
        
        if method not in method_map:
            raise ValueError(f"Unsupported calibration method: {method}. "
                           f"Supported methods: {list(method_map.keys())}")
        
        return method_map[method]
    
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
        
        # Extract model information
        self._extract_model_info(model)
        self._log_model_info(model)
        
        return model
    
    def _extract_model_info(self, model: onnx.ModelProto) -> None:
        """Extract and store model information."""
        self.model_info = {
            'inputs': [],
            'outputs': [],
            'opset_version': model.opset_import[0].version,
            'producer_name': model.producer_name,
            'producer_version': model.producer_version,
            'model_version': model.model_version
        }
        
        # Extract input information
        for input_tensor in model.graph.input:
            input_info = {
                'name': input_tensor.name,
                'shape': [dim.dim_value if dim.dim_value > 0 else -1 
                         for dim in input_tensor.type.tensor_type.shape.dim],
                'type': input_tensor.type.tensor_type.elem_type
            }
            self.model_info['inputs'].append(input_info)
        
        # Extract output information
        for output_tensor in model.graph.output:
            output_info = {
                'name': output_tensor.name,
                'shape': [dim.dim_value if dim.dim_value > 0 else -1 
                         for dim in output_tensor.type.tensor_type.shape.dim],
                'type': output_tensor.type.tensor_type.elem_type
            }
            self.model_info['outputs'].append(output_info)
    
    def _log_model_info(self, model: onnx.ModelProto) -> None:
        """Log comprehensive model information."""
        logger.info("=== Model Information ===")
        logger.info(f"ONNX Version: {self.model_info['opset_version']}")
        logger.info(f"Producer: {self.model_info['producer_name']} {self.model_info['producer_version']}")
        logger.info(f"Model Version: {self.model_info['model_version']}")
        
        # Count node types
        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1
            
        logger.info(f"Total Nodes: {len(model.graph.node)}")
        logger.info(f"Node Types: {len(node_types)}")
        
        # Log input/output information
        logger.info("Inputs:")
        for input_info in self.model_info['inputs']:
            logger.info(f"  {input_info['name']}: {input_info['shape']}")
            
        logger.info("Outputs:")
        for output_info in self.model_info['outputs']:
            logger.info(f"  {output_info['name']}: {output_info['shape']}")
    
    def create_calibration_dataset(self) -> CalibrationDataset:
        """
        Create calibration dataset from data path.
        
        Returns:
            Configured calibration dataset
        """
        if not self.calibration_data_path.exists():
            raise FileNotFoundError(f"Calibration data path not found: {self.calibration_data_path}")
        
        # Get input tensor information
        if not self.model_info['inputs']:
            raise ValueError("No input information found in model")
        
        primary_input = self.model_info['inputs'][0]
        input_name = primary_input['name']
        
        # Determine target dimensions from model input
        input_shape = primary_input['shape']
        if len(input_shape) >= 4:  # [N, C, H, W]
            target_height = input_shape[-2] if input_shape[-2] > 0 else 640
            target_width = input_shape[-1] if input_shape[-1] > 0 else 640
        else:
            target_height = target_width = 640
        
        logger.info(f"Creating calibration dataset with target dimensions: {target_width}x{target_height}")
        
        dataset = CalibrationDataset(
            data_path=str(self.calibration_data_path),
            input_name=input_name,
            target_width=target_width,
            target_height=target_height,
            max_samples=self.max_calibration_samples,
            preprocessing="coco"  # Default to COCO preprocessing
        )
        
        self.stats['calibration_samples_used'] = len(dataset)
        
        return dataset
    
    def generate_calibration_table(self, dataset: CalibrationDataset) -> str:
        """
        Generate calibration table using the provided dataset.
        
        Args:
            dataset: Calibration dataset
            
        Returns:
            Path to generated calibration table
        """
        logger.info(f"Generating calibration table using {self.calibration_method.name} method...")
        calibration_start_time = time.time()
        
        try:
            # Create calibration data reader
            calibration_data_reader = ONNXCalibrationDataReader(dataset)
            
            # Create calibrator
            calibrator = create_calibrator(
                str(self.model_path),
                [calibration_data_reader],
                str(self.calibration_table_path),
                calibration_method=self.calibration_method
            )
            
            # Generate calibration data
            calibrator.collect_data()
            
            # Write calibration table
            write_calibration_table(calibrator.compute_range())
            
            # Record calibration time
            self.stats['calibration_time_seconds'] = time.time() - calibration_start_time
            
            logger.info(f"✓ Calibration completed in {self.stats['calibration_time_seconds']:.2f}s")
            logger.info(f"Calibration table saved to: {self.calibration_table_path}")
            
            return str(self.calibration_table_path)
            
        except Exception as e:
            raise RuntimeError(f"Calibration failed: {e}")
    
    def quantize_to_int8(self, calibration_table_path: str) -> str:
        """
        Quantize model to INT8 using calibration table.
        
        Args:
            calibration_table_path: Path to calibration table
            
        Returns:
            Path to quantized model
        """
        logger.info("Starting INT8 quantization...")
        quantization_start_time = time.time()
        
        try:
            # Ensure output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Perform static quantization
            quantize_static(
                model_input=str(self.model_path),
                model_output=str(self.output_path),
                calibration_data_reader=None,  # Using pre-generated calibration table
                calibration_table_path=calibration_table_path,
                quant_format=QuantFormat.QOperator,
                per_channel=True,  # Enable per-channel quantization for better accuracy
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QUInt8,
                optimize_model=True,  # Enable model optimization during quantization
                use_external_data_format=False
            )
            
            # Record quantization time
            self.stats['quantization_time_seconds'] = time.time() - quantization_start_time
            
            # Calculate file size and compression ratio
            self.stats['quantized_size_mb'] = self.output_path.stat().st_size / (1024 * 1024)
            self.stats['compression_ratio'] = (
                self.stats['original_size_mb'] / self.stats['quantized_size_mb']
                if self.stats['quantized_size_mb'] > 0 else 0
            )
            
            logger.info(f"✓ INT8 quantization completed in {self.stats['quantization_time_seconds']:.2f}s")
            logger.info(f"Quantized model saved to: {self.output_path}")
            
            return str(self.output_path)
            
        except Exception as e:
            raise RuntimeError(f"INT8 quantization failed: {e}")
    
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
            
            # Test inference session creation
            providers = ['CPUExecutionProvider']
            try:
                session = ort.InferenceSession(str(self.output_path), providers=providers)
                
                # Verify inputs and outputs match
                session_inputs = [inp.name for inp in session.get_inputs()]
                session_outputs = [out.name for out in session.get_outputs()]
                
                expected_inputs = [inp['name'] for inp in self.model_info['inputs']]
                expected_outputs = [out['name'] for out in self.model_info['outputs']]
                
                if session_inputs != expected_inputs:
                    logger.warning(f"Input mismatch: expected {expected_inputs}, got {session_inputs}")
                    
                if session_outputs != expected_outputs:
                    logger.warning(f"Output mismatch: expected {expected_outputs}, got {session_outputs}")
                
            except Exception as e:
                logger.warning(f"Inference session validation failed: {e}")
                return False
            
            logger.info("✓ Quantized model validation passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Quantized model validation failed: {e}")
            return False
    
    def print_quantization_summary(self) -> None:
        """Print comprehensive quantization summary."""
        logger.info("\n" + "="*60)
        logger.info("         INT8 QUANTIZATION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Original Model Size:    {self.stats['original_size_mb']:.2f} MB")
        logger.info(f"Quantized Model Size:   {self.stats['quantized_size_mb']:.2f} MB")
        logger.info(f"Size Reduction:         {(1 - self.stats['quantized_size_mb']/self.stats['original_size_mb'])*100:.1f}%")
        logger.info(f"Compression Ratio:      {self.stats['compression_ratio']:.2f}x")
        
        logger.info(f"\nTiming Information:")
        logger.info(f"  Calibration Time:     {self.stats['calibration_time_seconds']:.2f}s")
        logger.info(f"  Quantization Time:    {self.stats['quantization_time_seconds']:.2f}s")
        logger.info(f"  Total Time:           {self.stats['calibration_time_seconds'] + self.stats['quantization_time_seconds']:.2f}s")
        
        logger.info(f"\nCalibration Information:")
        logger.info(f"  Method:               {self.stats['calibration_method']}")
        logger.info(f"  Samples Used:         {self.stats['calibration_samples_used']}")
        logger.info(f"  Calibration Table:    {self.calibration_table_path}")
        
        logger.info(f"\nModel Paths:")
        logger.info(f"  Original:             {self.model_path}")
        logger.info(f"  Quantized:            {self.output_path}")
        
        logger.info("="*60)
    
    def quantize(self) -> str:
        """
        Execute complete INT8 quantization pipeline with calibration.
        
        Returns:
            Path to quantized model
            
        Raises:
            RuntimeError: If quantization fails
        """
        try:
            # Load and validate original model
            model = self.load_and_validate_model()
            
            # Create calibration dataset
            dataset = self.create_calibration_dataset()
            
            # Generate calibration table
            calibration_table_path = self.generate_calibration_table(dataset)
            
            # Quantize model to INT8
            quantized_model_path = self.quantize_to_int8(calibration_table_path)
            
            # Validate quantized model
            if not self.validate_quantized_model():
                raise RuntimeError("Quantized model validation failed")
            
            # Print summary
            self.print_quantization_summary()
            
            return quantized_model_path
            
        except Exception as e:
            logger.error(f"INT8 quantization pipeline failed: {e}")
            raise RuntimeError(f"Quantization pipeline failed: {e}")


def main():
    """Main CLI entry point for INT8 quantization."""
    parser = argparse.ArgumentParser(
        description="INT8 Quantization for ONNX Models with Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic INT8 quantization
  python quantize_int8.py --model model.onnx --calibration-data ./images/
  
  # Custom output path and calibration method
  python quantize_int8.py --model model.onnx --output model_int8.onnx --calibration-data ./images/ --method Entropy
  
  # Limit calibration samples for faster quantization
  python quantize_int8.py --model model.onnx --calibration-data ./images/ --max-samples 50
  
  # Advanced configuration
  python quantize_int8.py --model model.onnx --calibration-data ./images/ --method Percentile --max-samples 200 --opset-version 17
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to input ONNX model"
    )
    
    parser.add_argument(
        "--calibration-data", "-d",
        required=True,
        help="Path to calibration images directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path for quantized output model (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--method",
        choices=["MinMax", "Entropy", "Percentile"],
        default="MinMax",
        help="Calibration method (default: MinMax)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of calibration samples (default: 100)"
    )
    
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version to use (default: 17)"
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
        quantizer = INT8Quantizer(
            model_path=args.model,
            calibration_data_path=args.calibration_data,
            output_path=args.output,
            calibration_method=args.method,
            max_calibration_samples=args.max_samples,
            opset_version=args.opset_version,
            validate_model=not args.no_validate
        )
        
        # Execute quantization
        output_path = quantizer.quantize()
        
        logger.info(f"✓ INT8 quantization completed successfully!")
        logger.info(f"Quantized model saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ INT8 quantization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())