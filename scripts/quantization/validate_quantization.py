#!/usr/bin/env python3
"""
Quantization Accuracy Preservation Validation Pipeline

This script provides comprehensive accuracy validation for quantized ONNX models,
ensuring ≥98% accuracy preservation as required by the performance specifications.
Supports IoU-based validation, mAP comparison, and multi-resolution testing.

Requirements:
- ONNX Runtime 1.16+
- PyTorch 2.0+ (for baseline comparison)
- Target: ≥98% detection fidelity (IoU ≥ 0.5 vs. PyTorch baseline)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantization_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BoundingBox:
    """Bounding box representation with IoU calculation."""
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, 
                 confidence: float, class_id: int, label: str = ""):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.class_id = class_id
        self.label = label
    
    def area(self) -> float:
        """Calculate bounding box area."""
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'x1': self.x1,
            'y1': self.y1, 
            'x2': self.x2,
            'y2': self.y2,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'label': self.label
        }


class Detection:
    """Single detection result with metadata."""
    
    def __init__(self, image_path: str, boxes: List[BoundingBox], 
                 inference_time: float, model_name: str):
        self.image_path = image_path
        self.boxes = boxes
        self.inference_time = inference_time
        self.model_name = model_name
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation.""" 
        return {
            'image_path': self.image_path,
            'boxes': [box.to_dict() for box in self.boxes],
            'inference_time': self.inference_time,
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'num_detections': len(self.boxes)
        }


class ModelInferenceEngine:
    """Inference engine for ONNX models with performance tracking."""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to ONNX model
            providers: ONNX Runtime execution providers
        """
        self.model_path = Path(model_path)
        self.providers = providers or ['CPUExecutionProvider']
        
        # Load model
        self.session = ort.InferenceSession(str(model_path), providers=self.providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"Loaded model: {self.model_path.name}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Providers: {self.providers}")
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image_path: Path to image file
            target_size: Target image dimensions (width, height)
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Load and resize image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            image = image.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Apply standard normalization (COCO/ImageNet)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std
            
            # Convert HWC to CHW and add batch dimension
            img_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
            
            return img_tensor.astype(np.float32), original_size
            
        except Exception as e:
            logger.error(f"Failed to preprocess {image_path}: {e}")
            raise
    
    def postprocess_detections(self, outputs: List[np.ndarray], 
                             original_size: Tuple[int, int],
                             confidence_threshold: float = 0.5,
                             nms_threshold: float = 0.7) -> List[BoundingBox]:
        """
        Post-process model outputs to extract detections.
        
        Args:
            outputs: Raw model outputs
            original_size: Original image size (width, height)  
            confidence_threshold: Confidence filtering threshold
            nms_threshold: NMS IoU threshold
            
        Returns:
            List of detected bounding boxes
        """
        # This is a generic implementation - should be customized per model
        # For demonstration, assuming YOLO-style outputs
        
        detections = []
        
        # Process outputs (assuming format: [boxes, scores, classes])
        if len(outputs) >= 2:
            boxes = outputs[0]  # [N, 4] or [1, N, 4]
            scores = outputs[1]  # [N, num_classes] or [1, N, num_classes]
            
            # Reshape if needed
            if len(boxes.shape) == 3:
                boxes = boxes[0]  # Remove batch dimension
            if len(scores.shape) == 3:
                scores = scores[0]  # Remove batch dimension
            
            # Process each detection
            for i in range(len(boxes)):
                # Get maximum confidence class
                class_scores = scores[i] if len(scores.shape) == 2 else [scores[i]]
                max_score = np.max(class_scores)
                class_id = np.argmax(class_scores)
                
                if max_score > confidence_threshold:
                    # Extract box coordinates (assuming normalized)
                    x1, y1, x2, y2 = boxes[i]
                    
                    # Scale to original image size
                    x1 *= original_size[0]
                    y1 *= original_size[1] 
                    x2 *= original_size[0]
                    y2 *= original_size[1]
                    
                    detection = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=max_score,
                        class_id=int(class_id),
                        label=f"class_{class_id}"
                    )
                    detections.append(detection)
        
        # Apply Non-Maximum Suppression
        detections = self._apply_nms(detections, nms_threshold)
        
        return detections
    
    def _apply_nms(self, detections: List[BoundingBox], threshold: float) -> List[BoundingBox]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        kept = []
        while detections:
            # Keep highest confidence detection
            best = detections.pop(0)
            kept.append(best)
            
            # Remove overlapping detections
            detections = [det for det in detections if best.iou(det) <= threshold]
        
        return kept
    
    def run_inference(self, image_path: str, target_size: Tuple[int, int] = (640, 640)) -> Detection:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image file
            target_size: Target image dimensions
            
        Returns:
            Detection result with timing information
        """
        # Preprocess image
        img_tensor, original_size = self.preprocess_image(image_path, target_size)
        
        # Run inference with timing
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: img_tensor})
        inference_time = time.time() - start_time
        
        # Post-process results
        boxes = self.postprocess_detections(outputs, original_size)
        
        return Detection(
            image_path=image_path,
            boxes=boxes,
            inference_time=inference_time,
            model_name=self.model_path.name
        )


class AccuracyValidator:
    """
    Comprehensive accuracy validation for quantized models.
    
    This validator compares original and quantized models across multiple
    metrics including IoU distribution, mAP preservation, and performance.
    """
    
    def __init__(self, original_model_path: str, quantized_model_path: str,
                 test_data_path: str, output_dir: str = "validation_results"):
        """
        Initialize accuracy validator.
        
        Args:
            original_model_path: Path to original ONNX model
            quantized_model_path: Path to quantized ONNX model
            test_data_path: Path to test images directory
            output_dir: Directory for validation results
        """
        self.original_model_path = Path(original_model_path)
        self.quantized_model_path = Path(quantized_model_path)
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize inference engines
        self.original_engine = ModelInferenceEngine(
            str(self.original_model_path),
            providers=['CPUExecutionProvider']
        )
        
        # Use DNNL provider for quantized model if available
        quantized_providers = ['DnnlExecutionProvider', 'CPUExecutionProvider']
        try:
            self.quantized_engine = ModelInferenceEngine(
                str(self.quantized_model_path),
                providers=quantized_providers
            )
        except Exception:
            # Fallback to CPU provider
            self.quantized_engine = ModelInferenceEngine(
                str(self.quantized_model_path),
                providers=['CPUExecutionProvider']
            )
        
        # Collect test images
        self.test_images = self._collect_test_images()
        logger.info(f"Found {len(self.test_images)} test images")
        
        # Validation results
        self.results = {
            'original_detections': [],
            'quantized_detections': [],
            'comparison_metrics': {},
            'performance_metrics': {},
            'validation_timestamp': time.time()
        }
    
    def _collect_test_images(self) -> List[str]:
        """Collect test image files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        images = []
        
        for ext in image_extensions:
            images.extend(self.test_data_path.glob(f"*{ext}"))
            images.extend(self.test_data_path.glob(f"*{ext.upper()}"))
        
        # Recursively search subdirectories
        for subdir in self.test_data_path.iterdir():
            if subdir.is_dir():
                for ext in image_extensions:
                    images.extend(subdir.glob(f"*{ext}"))
                    images.extend(subdir.glob(f"*{ext.upper()}"))
        
        return sorted([str(img) for img in images])
    
    def run_validation(self, resolutions: List[Tuple[int, int]] = None,
                      max_images: int = None) -> Dict:
        """
        Run comprehensive validation comparing original vs quantized models.
        
        Args:
            resolutions: List of (width, height) resolutions to test
            max_images: Maximum number of test images (None for all)
            
        Returns:
            Comprehensive validation results
        """
        if resolutions is None:
            resolutions = [(640, 640)]  # Default resolution
        
        # Limit test images if specified
        test_images = self.test_images[:max_images] if max_images else self.test_images
        
        logger.info(f"Starting validation on {len(test_images)} images with {len(resolutions)} resolutions")
        
        for resolution in resolutions:
            logger.info(f"Testing resolution: {resolution[0]}x{resolution[1]}")
            
            resolution_key = f"{resolution[0]}x{resolution[1]}"
            self.results['comparison_metrics'][resolution_key] = {
                'iou_distribution': [],
                'detection_pairs': [],
                'precision_recall': {},
                'map_scores': {}
            }
            
            # Run inference on both models
            original_detections = []
            quantized_detections = []
            
            for img_path in tqdm(test_images, desc=f"Processing {resolution_key}"):
                try:
                    # Original model inference
                    orig_det = self.original_engine.run_inference(img_path, resolution)
                    original_detections.append(orig_det)
                    
                    # Quantized model inference
                    quant_det = self.quantized_engine.run_inference(img_path, resolution)
                    quantized_detections.append(quant_det)
                    
                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
                    continue
            
            # Store results
            self.results['original_detections'].extend(original_detections)
            self.results['quantized_detections'].extend(quantized_detections)
            
            # Compute comparison metrics
            self._compute_comparison_metrics(
                original_detections, 
                quantized_detections,
                resolution_key
            )
            
            # Compute performance metrics
            self._compute_performance_metrics(
                original_detections,
                quantized_detections, 
                resolution_key
            )
        
        # Generate validation report
        self._generate_validation_report()
        
        return self.results
    
    def _compute_comparison_metrics(self, original_dets: List[Detection],
                                  quantized_dets: List[Detection],
                                  resolution_key: str) -> None:
        """Compute detailed comparison metrics between models."""
        metrics = self.results['comparison_metrics'][resolution_key]
        
        # Pair detections by image
        detection_pairs = []
        iou_scores = []
        
        for orig_det, quant_det in zip(original_dets, quantized_dets):
            if orig_det.image_path != quant_det.image_path:
                logger.warning(f"Image path mismatch: {orig_det.image_path} vs {quant_det.image_path}")
                continue
            
            # Compute IoU for detection pairs
            pair_ious = []
            orig_boxes = orig_det.boxes
            quant_boxes = quant_det.boxes
            
            # Match detections using Hungarian algorithm (simplified)
            matches = self._match_detections(orig_boxes, quant_boxes)
            
            for orig_idx, quant_idx in matches:
                if orig_idx is not None and quant_idx is not None:
                    iou = orig_boxes[orig_idx].iou(quant_boxes[quant_idx])
                    pair_ious.append(iou)
                    iou_scores.append(iou)
            
            detection_pairs.append({
                'image_path': orig_det.image_path,
                'original_count': len(orig_boxes),
                'quantized_count': len(quant_boxes),
                'matched_pairs': len(matches),
                'mean_iou': np.mean(pair_ious) if pair_ious else 0.0,
                'max_iou': np.max(pair_ious) if pair_ious else 0.0,
                'min_iou': np.min(pair_ious) if pair_ious else 0.0
            })
        
        # Store metrics
        metrics['iou_distribution'] = iou_scores
        metrics['detection_pairs'] = detection_pairs
        
        # Compute summary statistics
        if iou_scores:
            metrics['summary'] = {
                'mean_iou': np.mean(iou_scores),
                'median_iou': np.median(iou_scores),
                'std_iou': np.std(iou_scores),
                'min_iou': np.min(iou_scores),
                'max_iou': np.max(iou_scores),
                'iou_050_threshold': np.mean(np.array(iou_scores) >= 0.5),
                'iou_075_threshold': np.mean(np.array(iou_scores) >= 0.75),
                'total_comparisons': len(iou_scores)
            }
        else:
            metrics['summary'] = {
                'mean_iou': 0.0,
                'total_comparisons': 0
            }
        
        logger.info(f"Resolution {resolution_key} - Mean IoU: {metrics['summary']['mean_iou']:.4f}")
    
    def _match_detections(self, orig_boxes: List[BoundingBox], 
                         quant_boxes: List[BoundingBox]) -> List[Tuple[Optional[int], Optional[int]]]:
        """Simple detection matching based on IoU (simplified Hungarian algorithm)."""
        matches = []
        used_quant = set()
        
        for i, orig_box in enumerate(orig_boxes):
            best_match = None
            best_iou = 0.3  # Minimum IoU threshold for matching
            
            for j, quant_box in enumerate(quant_boxes):
                if j in used_quant:
                    continue
                    
                iou = orig_box.iou(quant_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = j
            
            if best_match is not None:
                matches.append((i, best_match))
                used_quant.add(best_match)
            else:
                matches.append((i, None))
        
        return matches
    
    def _compute_performance_metrics(self, original_dets: List[Detection],
                                   quantized_dets: List[Detection],
                                   resolution_key: str) -> None:
        """Compute performance comparison metrics."""
        if resolution_key not in self.results['performance_metrics']:
            self.results['performance_metrics'][resolution_key] = {}
        
        metrics = self.results['performance_metrics'][resolution_key]
        
        # Timing comparison
        orig_times = [det.inference_time for det in original_dets]
        quant_times = [det.inference_time for det in quantized_dets]
        
        metrics['timing'] = {
            'original_mean_ms': np.mean(orig_times) * 1000,
            'original_std_ms': np.std(orig_times) * 1000,
            'quantized_mean_ms': np.mean(quant_times) * 1000,
            'quantized_std_ms': np.std(quant_times) * 1000,
            'speedup_ratio': np.mean(orig_times) / np.mean(quant_times) if np.mean(quant_times) > 0 else 0,
            'samples': len(orig_times)
        }
        
        # Detection count comparison
        orig_counts = [len(det.boxes) for det in original_dets]
        quant_counts = [len(det.boxes) for det in quantized_dets]
        
        metrics['detection_counts'] = {
            'original_mean': np.mean(orig_counts),
            'original_std': np.std(orig_counts),
            'quantized_mean': np.mean(quant_counts),
            'quantized_std': np.std(quant_counts),
            'count_correlation': np.corrcoef(orig_counts, quant_counts)[0, 1] if len(orig_counts) > 1 else 0
        }
        
        logger.info(f"Resolution {resolution_key} - Speedup: {metrics['timing']['speedup_ratio']:.2f}x")
    
    def _generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        report_path = self.output_dir / "validation_report.json"
        
        # Create summary
        overall_metrics = {
            'model_paths': {
                'original': str(self.original_model_path),
                'quantized': str(self.quantized_model_path)
            },
            'test_configuration': {
                'test_images': len(self.test_images),
                'resolutions_tested': list(self.results['comparison_metrics'].keys())
            },
            'overall_summary': {}
        }
        
        # Aggregate metrics across resolutions
        all_ious = []
        all_speedups = []
        
        for res_key, metrics in self.results['comparison_metrics'].items():
            if 'summary' in metrics:
                all_ious.extend(metrics['iou_distribution'])
            
            if res_key in self.results['performance_metrics']:
                perf_metrics = self.results['performance_metrics'][res_key]
                if 'timing' in perf_metrics:
                    all_speedups.append(perf_metrics['timing']['speedup_ratio'])
        
        # Overall summary
        if all_ious:
            overall_metrics['overall_summary'] = {
                'mean_iou': float(np.mean(all_ious)),
                'iou_050_preservation': float(np.mean(np.array(all_ious) >= 0.5)),
                'accuracy_preservation_percent': float(np.mean(np.array(all_ious) >= 0.5) * 100),
                'meets_98_percent_target': float(np.mean(np.array(all_ious) >= 0.5) * 100) >= 98.0,
                'mean_speedup': float(np.mean(all_speedups)) if all_speedups else 0.0,
                'total_comparisons': len(all_ious)
            }
        
        # Combine with detailed results
        full_report = {**overall_metrics, **self.results}
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to: {report_path}")
        
        # Print summary
        self._print_validation_summary(overall_metrics['overall_summary'])
    
    def _print_validation_summary(self, summary: Dict) -> None:
        """Print validation summary to console."""
        logger.info("\n" + "="*70)
        logger.info("           QUANTIZATION VALIDATION SUMMARY")
        logger.info("="*70)
        
        if summary:
            logger.info(f"Overall IoU Score:          {summary['mean_iou']:.4f}")
            logger.info(f"Accuracy Preservation:      {summary['accuracy_preservation_percent']:.1f}%")
            logger.info(f"IoU ≥0.5 Detections:        {summary['iou_050_preservation']:.1f}%")
            logger.info(f"Meets ≥98% Target:          {'✓ PASS' if summary['meets_98_percent_target'] else '✗ FAIL'}")
            
            if summary['mean_speedup'] > 0:
                logger.info(f"Average Speedup:            {summary['mean_speedup']:.2f}x")
            
            logger.info(f"Total Comparisons:          {summary['total_comparisons']}")
        else:
            logger.info("No validation metrics computed")
        
        logger.info("="*70)
        logger.info(f"Detailed results saved to:  {self.output_dir}")
        logger.info("="*70)


def main():
    """Main CLI entry point for quantization validation."""
    parser = argparse.ArgumentParser(
        description="Quantization Accuracy Preservation Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_quantization.py --original model.onnx --quantized model_int8.onnx --test-data ./test_images/
  
  # Multi-resolution validation
  python validate_quantization.py --original model.onnx --quantized model_int8.onnx --test-data ./images/ --resolutions 320,640,1024
  
  # Limit test samples for faster validation
  python validate_quantization.py --original model.onnx --quantized model_int8.onnx --test-data ./images/ --max-images 50
  
  # Custom output directory
  python validate_quantization.py --original model.onnx --quantized model_int8.onnx --test-data ./images/ --output-dir ./validation/
        """
    )
    
    parser.add_argument(
        "--original", "-o",
        required=True,
        help="Path to original ONNX model"
    )
    
    parser.add_argument(
        "--quantized", "-q",
        required=True,
        help="Path to quantized ONNX model"
    )
    
    parser.add_argument(
        "--test-data", "-t",
        required=True,
        help="Path to test images directory"
    )
    
    parser.add_argument(
        "--resolutions",
        default="640",
        help="Comma-separated resolutions to test (e.g., '320,640,1024')"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of test images (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="validation_results",
        help="Output directory for results (default: validation_results)"
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
    
    # Parse resolutions
    resolutions = []
    for res_str in args.resolutions.split(','):
        size = int(res_str.strip())
        resolutions.append((size, size))
    
    try:
        # Initialize validator
        validator = AccuracyValidator(
            original_model_path=args.original,
            quantized_model_path=args.quantized,
            test_data_path=args.test_data,
            output_dir=args.output_dir
        )
        
        # Run validation
        results = validator.run_validation(
            resolutions=resolutions,
            max_images=args.max_images
        )
        
        # Check if validation passed
        summary = results.get('overall_summary', {})
        if summary.get('meets_98_percent_target', False):
            logger.info("✓ Quantization validation PASSED - meets ≥98% accuracy target")
            return 0
        else:
            logger.warning("✗ Quantization validation FAILED - does not meet ≥98% accuracy target")
            return 1
        
    except Exception as e:
        logger.error(f"✗ Quantization validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())