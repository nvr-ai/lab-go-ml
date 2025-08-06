#!/usr/bin/env python3
"""
Quantization Pipeline CLI

Unified command-line interface for ONNX model quantization workflows.
Supports FP16, INT8 quantization, and accuracy validation with comprehensive
reporting and integration with the Go inference pipeline.

Requirements:
- ONNX Runtime 1.16+
- Calibration dataset for INT8 quantization
- Test dataset for accuracy validation
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Import quantization modules
from quantize_fp16 import FP16Quantizer
from quantize_int8 import INT8Quantizer
from validate_quantization import AccuracyValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantization_cli.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class QuantizationPipeline:
    """
    Comprehensive quantization pipeline orchestrator.
    
    This class manages the complete quantization workflow including
    FP16/INT8 conversion, validation, and Go runtime integration.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize quantization pipeline.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.results = {
            'pipeline_start_time': time.time(),
            'stages_completed': [],
            'quantized_models': {},
            'validation_results': {},
            'performance_metrics': {},
            'errors': []
        }
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Quantization pipeline initialized")
        logger.info(f"Target model: {self.config['model_path']}")
        logger.info(f"Output directory: {self.config['output_dir']}")
    
    def _validate_config(self) -> None:
        """Validate pipeline configuration."""
        required_keys = ['model_path', 'output_dir']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # Check model file exists
        model_path = Path(self.config['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✓ Configuration validation passed")
    
    def run_fp16_quantization(self) -> Optional[str]:
        """
        Run FP16 quantization stage.
        
        Returns:
            Path to FP16 quantized model, or None if skipped/failed
        """
        if not self.config.get('enable_fp16', True):
            logger.info("FP16 quantization skipped (disabled in config)")
            return None
        
        logger.info("Starting FP16 quantization stage...")
        stage_start = time.time()
        
        try:
            # Configure FP16 quantizer
            model_path = Path(self.config['model_path'])
            output_path = Path(self.config['output_dir']) / f"{model_path.stem}_fp16{model_path.suffix}"
            
            quantizer = FP16Quantizer(
                model_path=str(model_path),
                output_path=str(output_path),
                opset_version=self.config.get('opset_version', 17),
                validate_model=self.config.get('validate_model', True)
            )
            
            # Set preservation options
            quantizer.preserve_io_types = self.config.get('preserve_io_types', True)
            
            # Run quantization
            quantized_path = quantizer.quantize()
            
            # Store results
            stage_time = time.time() - stage_start
            self.results['quantized_models']['fp16'] = {
                'path': quantized_path,
                'stats': quantizer.stats,
                'processing_time': stage_time
            }
            self.results['stages_completed'].append('fp16_quantization')
            
            logger.info(f"✓ FP16 quantization completed in {stage_time:.2f}s")
            return quantized_path
            
        except Exception as e:
            error_msg = f"FP16 quantization failed: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return None
    
    def run_int8_quantization(self) -> Optional[str]:
        """
        Run INT8 quantization stage with calibration.
        
        Returns:
            Path to INT8 quantized model, or None if skipped/failed
        """
        if not self.config.get('enable_int8', True):
            logger.info("INT8 quantization skipped (disabled in config)")
            return None
        
        calibration_data = self.config.get('calibration_data_path')
        if not calibration_data:
            logger.warning("INT8 quantization skipped (no calibration data provided)")
            return None
        
        logger.info("Starting INT8 quantization stage...")
        stage_start = time.time()
        
        try:
            # Configure INT8 quantizer
            model_path = Path(self.config['model_path'])
            output_path = Path(self.config['output_dir']) / f"{model_path.stem}_int8{model_path.suffix}"
            
            quantizer = INT8Quantizer(
                model_path=str(model_path),
                calibration_data_path=calibration_data,
                output_path=str(output_path),
                calibration_method=self.config.get('calibration_method', 'MinMax'),
                max_calibration_samples=self.config.get('max_calibration_samples', 100),
                opset_version=self.config.get('opset_version', 17),
                validate_model=self.config.get('validate_model', True)
            )
            
            # Run quantization
            quantized_path = quantizer.quantize()
            
            # Store results
            stage_time = time.time() - stage_start
            self.results['quantized_models']['int8'] = {
                'path': quantized_path,
                'stats': quantizer.stats,
                'processing_time': stage_time,
                'calibration_table': str(quantizer.calibration_table_path)
            }
            self.results['stages_completed'].append('int8_quantization')
            
            logger.info(f"✓ INT8 quantization completed in {stage_time:.2f}s")
            return quantized_path
            
        except Exception as e:
            error_msg = f"INT8 quantization failed: {e}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return None
    
    def run_accuracy_validation(self, quantized_models: Dict[str, str]) -> Dict:
        """
        Run accuracy validation for quantized models.
        
        Args:
            quantized_models: Dictionary mapping quantization type to model path
            
        Returns:
            Validation results dictionary
        """
        if not self.config.get('enable_validation', True):
            logger.info("Accuracy validation skipped (disabled in config)")
            return {}
        
        test_data = self.config.get('test_data_path')
        if not test_data:
            logger.warning("Accuracy validation skipped (no test data provided)")
            return {}
        
        logger.info("Starting accuracy validation stage...")
        validation_results = {}
        
        original_model = self.config['model_path']
        
        for quant_type, quant_model_path in quantized_models.items():
            if not quant_model_path:
                continue
                
            logger.info(f"Validating {quant_type.upper()} quantization...")
            stage_start = time.time()
            
            try:
                # Configure validator
                output_dir = Path(self.config['output_dir']) / f"validation_{quant_type}"
                
                validator = AccuracyValidator(
                    original_model_path=original_model,
                    quantized_model_path=quant_model_path,
                    test_data_path=test_data,
                    output_dir=str(output_dir)
                )
                
                # Parse resolutions
                resolutions = []
                for size in self.config.get('validation_resolutions', [640]):
                    resolutions.append((size, size))
                
                # Run validation
                results = validator.run_validation(
                    resolutions=resolutions,
                    max_images=self.config.get('max_validation_images')
                )
                
                # Store results
                stage_time = time.time() - stage_start
                validation_results[quant_type] = {
                    'results': results,
                    'validation_time': stage_time,
                    'output_dir': str(output_dir)
                }
                
                # Check accuracy target
                summary = results.get('overall_summary', {})
                accuracy_target_met = summary.get('meets_98_percent_target', False)
                
                logger.info(f"✓ {quant_type.upper()} validation completed in {stage_time:.2f}s")
                logger.info(f"  Accuracy: {summary.get('accuracy_preservation_percent', 0):.1f}% "
                          f"({'PASS' if accuracy_target_met else 'FAIL'} ≥98% target)")
                
            except Exception as e:
                error_msg = f"{quant_type.upper()} validation failed: {e}"
                logger.error(error_msg)
                self.results['errors'].append(error_msg)
                validation_results[quant_type] = {'error': str(e)}
        
        if validation_results:
            self.results['stages_completed'].append('accuracy_validation')
        
        return validation_results
    
    def generate_go_integration_config(self, quantized_models: Dict[str, str]) -> str:
        """
        Generate Go runtime configuration for quantized models.
        
        Args:
            quantized_models: Dictionary mapping quantization type to model path
            
        Returns:
            Path to generated Go configuration file
        """
        logger.info("Generating Go runtime integration configuration...")
        
        # Create Go configuration
        go_config = {
            'quantized_models': {},
            'recommended_providers': {},
            'performance_profiles': {},
            'generated_timestamp': time.time(),
            'original_model': self.config['model_path']
        }
        
        for quant_type, model_path in quantized_models.items():
            if not model_path:
                continue
                
            # Get model stats
            model_stats = self.results['quantized_models'].get(quant_type, {}).get('stats', {})
            
            go_config['quantized_models'][quant_type] = {
                'path': model_path,
                'size_mb': model_stats.get('quantized_size_mb', 0),
                'compression_ratio': model_stats.get('compression_ratio', 1.0)
            }
            
            # Recommended execution providers
            if quant_type == 'fp16':
                go_config['recommended_providers'][quant_type] = [
                    {'provider': 'CUDAExecutionProvider', 'priority': 10, 'enabled': True},
                    {'provider': 'CPUExecutionProvider', 'priority': 1, 'enabled': True}
                ]
            elif quant_type == 'int8':
                go_config['recommended_providers'][quant_type] = [
                    {'provider': 'DnnlExecutionProvider', 'priority': 10, 'enabled': True},
                    {'provider': 'CPUExecutionProvider', 'priority': 1, 'enabled': True}
                ]
            
            # Performance profiles based on validation results
            validation_results = self.results['validation_results'].get(quant_type, {})
            if 'results' in validation_results:
                perf_metrics = validation_results['results'].get('performance_metrics', {})
                go_config['performance_profiles'][quant_type] = perf_metrics
        
        # Save configuration
        config_path = Path(self.config['output_dir']) / "go_integration_config.json"
        with open(config_path, 'w') as f:
            json.dump(go_config, f, indent=2, default=str)
        
        logger.info(f"Go integration configuration saved to: {config_path}")
        return str(config_path)
    
    def generate_pipeline_report(self) -> str:
        """
        Generate comprehensive pipeline execution report.
        
        Returns:
            Path to generated report file
        """
        logger.info("Generating pipeline execution report...")
        
        # Calculate total execution time
        total_time = time.time() - self.results['pipeline_start_time']
        self.results['total_execution_time'] = total_time
        
        # Create comprehensive report
        report = {
            'pipeline_configuration': self.config,
            'execution_summary': {
                'total_time_seconds': total_time,
                'stages_completed': self.results['stages_completed'],
                'stages_attempted': len(self.results['stages_completed']),
                'errors_encountered': len(self.results['errors']),
                'success_rate': len([s for s in self.results['stages_completed'] if 'failed' not in s]) / max(1, len(self.results['stages_completed']))
            },
            'quantized_models': self.results['quantized_models'],
            'validation_results': self.results['validation_results'],
            'errors': self.results['errors'],
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = Path(self.config['output_dir']) / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary
        self._print_pipeline_summary(report)
        
        logger.info(f"Pipeline report saved to: {report_path}")
        return str(report_path)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []
        
        # Check for quantization success
        if 'fp16' in self.results['quantized_models']:
            fp16_stats = self.results['quantized_models']['fp16']['stats']
            compression = fp16_stats.get('compression_ratio', 1.0)
            if compression > 1.8:
                recommendations.append("FP16 quantization achieved good compression ratio - recommended for GPU deployment")
        
        if 'int8' in self.results['quantized_models']:
            int8_stats = self.results['quantized_models']['int8']['stats']
            compression = int8_stats.get('compression_ratio', 1.0)
            if compression > 3.0:
                recommendations.append("INT8 quantization achieved excellent compression - recommended for CPU deployment")
        
        # Check validation results
        for quant_type, validation in self.results['validation_results'].items():
            if 'results' in validation:
                summary = validation['results'].get('overall_summary', {})
                accuracy = summary.get('accuracy_preservation_percent', 0)
                
                if accuracy >= 98.0:
                    recommendations.append(f"{quant_type.upper()} quantization meets accuracy target - production ready")
                elif accuracy >= 95.0:
                    recommendations.append(f"{quant_type.upper()} quantization has good accuracy - consider fine-tuning")
                else:
                    recommendations.append(f"{quant_type.upper()} quantization has low accuracy - requires investigation")
        
        # Performance recommendations
        if len(self.results['errors']) > 0:
            recommendations.append("Review errors and consider adjusting quantization parameters")
        
        if not self.results['validation_results']:
            recommendations.append("Run accuracy validation with representative test data")
        
        return recommendations
    
    def _print_pipeline_summary(self, report: Dict) -> None:
        """Print pipeline execution summary."""
        logger.info("\n" + "="*80)
        logger.info("                    QUANTIZATION PIPELINE SUMMARY")
        logger.info("="*80)
        
        summary = report['execution_summary']
        logger.info(f"Total Execution Time:       {summary['total_time_seconds']:.2f}s")
        logger.info(f"Stages Completed:           {summary['stages_attempted']}")
        logger.info(f"Success Rate:               {summary['success_rate']:.1%}")
        logger.info(f"Errors Encountered:         {summary['errors_encountered']}")
        
        logger.info("\nQuantized Models:")
        for quant_type, model_info in self.results['quantized_models'].items():
            stats = model_info['stats']
            logger.info(f"  {quant_type.upper()}:")
            logger.info(f"    Size: {stats.get('original_size_mb', 0):.1f} → {stats.get('quantized_size_mb', 0):.1f} MB")
            logger.info(f"    Compression: {stats.get('compression_ratio', 1.0):.1f}x")
            logger.info(f"    Path: {Path(model_info['path']).name}")
        
        if self.results['validation_results']:
            logger.info("\nValidation Results:")
            for quant_type, validation in self.results['validation_results'].items():
                if 'results' in validation:
                    summary = validation['results'].get('overall_summary', {})
                    accuracy = summary.get('accuracy_preservation_percent', 0)
                    target_met = summary.get('meets_98_percent_target', False)
                    logger.info(f"  {quant_type.upper()}: {accuracy:.1f}% accuracy ({'PASS' if target_met else 'FAIL'})")
        
        if report['recommendations']:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("="*80)
        logger.info(f"Full report: {self.config['output_dir']}")
        logger.info("="*80)
    
    def run_complete_pipeline(self) -> Dict:
        """
        Execute the complete quantization pipeline.
        
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete quantization pipeline...")
        
        # Stage 1: FP16 Quantization
        fp16_model = self.run_fp16_quantization()
        
        # Stage 2: INT8 Quantization  
        int8_model = self.run_int8_quantization()
        
        # Stage 3: Accuracy Validation
        quantized_models = {}
        if fp16_model:
            quantized_models['fp16'] = fp16_model
        if int8_model:
            quantized_models['int8'] = int8_model
        
        validation_results = self.run_accuracy_validation(quantized_models)
        self.results['validation_results'] = validation_results
        
        # Stage 4: Go Integration Config
        if quantized_models:
            self.generate_go_integration_config(quantized_models)
            self.results['stages_completed'].append('go_integration_config')
        
        # Stage 5: Final Report
        report_path = self.generate_pipeline_report()
        self.results['report_path'] = report_path
        self.results['stages_completed'].append('final_report')
        
        logger.info("✓ Complete quantization pipeline finished")
        return self.results


def create_config_from_args(args) -> Dict:
    """Create pipeline configuration from CLI arguments."""
    config = {
        'model_path': args.model,
        'output_dir': args.output_dir,
        'enable_fp16': args.fp16,
        'enable_int8': args.int8,
        'enable_validation': args.validate,
        'opset_version': args.opset_version,
        'validate_model': not args.no_model_validation,
        'preserve_io_types': args.preserve_io_types
    }
    
    # INT8 specific configuration
    if args.calibration_data:
        config['calibration_data_path'] = args.calibration_data
        config['calibration_method'] = args.calibration_method
        config['max_calibration_samples'] = args.max_calibration_samples
    
    # Validation specific configuration
    if args.test_data:
        config['test_data_path'] = args.test_data
        config['validation_resolutions'] = [int(r.strip()) for r in args.resolutions.split(',')]
        config['max_validation_images'] = args.max_validation_images
    
    return config


def main():
    """Main CLI entry point for quantization pipeline."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ONNX Model Quantization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline with FP16, INT8, and validation
  python cli.py --model model.onnx --calibration-data ./calib_images/ --test-data ./test_images/ --output-dir ./quantized/
  
  # FP16 only
  python cli.py --model model.onnx --fp16 --no-int8 --output-dir ./quantized/
  
  # INT8 with custom calibration method
  python cli.py --model model.onnx --int8 --calibration-data ./images/ --calibration-method Entropy --max-calibration-samples 200
  
  # Multi-resolution validation
  python cli.py --model model.onnx --test-data ./images/ --resolutions "320,640,1024" --validate
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to input ONNX model"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="quantized_models",
        help="Output directory for quantized models and results"
    )
    
    # Quantization options
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 quantization (default: enabled)"
    )
    
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16 quantization"
    )
    
    parser.add_argument(
        "--int8",
        action="store_true",
        default=True,
        help="Enable INT8 quantization (default: enabled if calibration data provided)"
    )
    
    parser.add_argument(
        "--no-int8",
        dest="int8",
        action="store_false",
        help="Disable INT8 quantization"
    )
    
    # INT8 calibration options
    parser.add_argument(
        "--calibration-data",
        help="Path to calibration images directory (required for INT8)"
    )
    
    parser.add_argument(
        "--calibration-method",
        choices=["MinMax", "Entropy", "Percentile"],
        default="MinMax",
        help="Calibration method for INT8 quantization"
    )
    
    parser.add_argument(
        "--max-calibration-samples",
        type=int,
        default=100,
        help="Maximum calibration samples"
    )
    
    # Validation options
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Enable accuracy validation (default: enabled if test data provided)"
    )
    
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Disable accuracy validation"
    )
    
    parser.add_argument(
        "--test-data",
        help="Path to test images directory (required for validation)"
    )
    
    parser.add_argument(
        "--resolutions",
        default="640",
        help="Comma-separated resolutions for validation (e.g., '320,640,1024')"
    )
    
    parser.add_argument(
        "--max-validation-images",
        type=int,
        help="Maximum validation images (default: all)"
    )
    
    # Model options
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version"
    )
    
    parser.add_argument(
        "--preserve-io-types",
        action="store_true",
        help="Preserve input/output tensor types for compatibility"
    )
    
    parser.add_argument(
        "--no-model-validation",
        action="store_true",
        help="Skip model validation (faster but less safe)"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate arguments
    if args.int8 and not args.calibration_data:
        logger.warning("INT8 quantization requires calibration data - disabling INT8")
        args.int8 = False
    
    if args.validate and not args.test_data:
        logger.warning("Accuracy validation requires test data - disabling validation")
        args.validate = False
    
    try:
        # Create pipeline configuration
        config = create_config_from_args(args)
        
        # Initialize and run pipeline
        pipeline = QuantizationPipeline(config)
        results = pipeline.run_complete_pipeline()
        
        # Determine exit code based on results
        if results['errors']:
            logger.warning(f"Pipeline completed with {len(results['errors'])} errors")
            return 1
        
        # Check validation results
        validation_passed = True
        for validation in results.get('validation_results', {}).values():
            if 'results' in validation:
                summary = validation['results'].get('overall_summary', {})
                if not summary.get('meets_98_percent_target', True):
                    validation_passed = False
                    break
        
        if not validation_passed:
            logger.warning("Some quantized models did not meet 98% accuracy target")
            return 2
        
        logger.info("✓ Quantization pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"✗ Quantization pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())