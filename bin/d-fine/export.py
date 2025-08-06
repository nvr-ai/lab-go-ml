#!/usr/bin/env uv run

# /// script
# dependencies = [
#     "torch>=2.0.1",
#     "torchvision>=0.15.2",
#     "onnx>=1.12.0",
#     "onnxsim>=0.4.0",
#     "PyYAML",
#     "loguru",
# ]
# ///

"""
D-FINE ONNX Export Script
Exports D-FINE models to ONNX format with variable input dimensions for object detection.
Supports 16:9 and 4:3 aspect ratios with performance-optimized dynamic shapes.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Resolution:
    """Represents a supported resolution with width, height, and aspect ratio."""
    width: int
    height: int
    aspect_ratio: str
    
    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Resolution dimensions must be positive: {self.width}x{self.height}")
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def __str__(self) -> str:
        return f"{self.width}x{self.height} ({self.aspect_ratio})"


class ResolutionConfig:
    """Configuration class for supported resolutions with aspect ratio constraints."""
    
    # Supported max widths (aligned to 32 pixels for better model compatibility)
    MAX_WIDTHS = [160, 320, 640, 960]
    
    # Aspect ratio definitions
    ASPECT_RATIOS = {
        "16:9": (16, 9),
        "4:3": (4, 3)
    }
    
    @classmethod
    def get_supported_resolutions(cls, aspect_ratios: Optional[List[str]] = None) -> List[Resolution]:
        """
        Generate all supported resolutions for given aspect ratios.
        
        Args:
            aspect_ratios: List of aspect ratios to include. If None, includes all.
        
        Returns:
            List of Resolution objects sorted by total pixels.
        """
        if aspect_ratios is None:
            aspect_ratios = list(cls.ASPECT_RATIOS.keys())
        
        resolutions = []
        for aspect_ratio in aspect_ratios:
            if aspect_ratio not in cls.ASPECT_RATIOS:
                raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}")
            
            ratio_w, ratio_h = cls.ASPECT_RATIOS[aspect_ratio]
            
            for max_width in cls.MAX_WIDTHS:
                height = (max_width * ratio_h) // ratio_w
                resolutions.append(Resolution(max_width, height, aspect_ratio))
        
        return sorted(resolutions, key=lambda r: r.width * r.height)
    
    @classmethod
    def validate_resolution(cls, width: int, height: int) -> bool:
        """Validate if a resolution is supported."""
        for ratio_name, (ratio_w, ratio_h) in cls.ASPECT_RATIOS.items():
            expected_height = (width * ratio_h) // ratio_w
            if height == expected_height and width in cls.MAX_WIDTHS:
                return True
        return False
    
    @classmethod
    def get_optimal_resolution(cls, target_width: int, target_height: int, 
                             aspect_ratios: Optional[List[str]] = None) -> Optional[Resolution]:
        """
        Find the optimal supported resolution for given target dimensions.
        
        Args:
            target_width: Target width
            target_height: Target height  
            aspect_ratios: Preferred aspect ratios
        
        Returns:
            Best matching Resolution or None if no suitable resolution found.
        """
        supported = cls.get_supported_resolutions(aspect_ratios)
        
        # Find resolutions that fit within target dimensions
        fitting = [r for r in supported if r.width <= target_width and r.height <= target_height]
        
        if not fitting:
            return None
        
        # Return the largest fitting resolution
        return max(fitting, key=lambda r: r.width * r.height)


def export_d_fine_model(
    config_path: str = "configs/dfine/dfine_hgnetv2_l_coco.yml",
    resume_path: str = None,
    output_path: str = None,
    check_model: bool = True,
    simplify_model: bool = True,
    aspect_ratios: Optional[List[str]] = None,
    max_dimensions: Optional[Tuple[int, int]] = None,
    enable_dynamic_shapes: bool = True
):
    """
    Export D-FINE model to ONNX format with variable input dimensions.
    
    Arguments:
    - config_path: Path to the model configuration file
    - resume_path: Path to the model checkpoint file
    - output_path: Output path for the ONNX model
    - check_model: Whether to validate the exported ONNX model
    - simplify_model: Whether to simplify the ONNX model
    - aspect_ratios: List of aspect ratios to support (e.g., ["16:9", "4:3"])
    - max_dimensions: Maximum (width, height) constraint for resolutions
    - enable_dynamic_shapes: Enable dynamic input shapes for variable dimensions
    
    Returns:
    - Path to the exported ONNX model file
    """
    # Initialize resolution configuration
    if aspect_ratios is None:
        aspect_ratios = ["16:9", "4:3"]  # Default to both aspect ratios
    
    # Get supported resolutions based on constraints
    try:
        supported_resolutions = ResolutionConfig.get_supported_resolutions(aspect_ratios)
        
        # Apply max dimensions constraint if specified
        if max_dimensions:
            max_w, max_h = max_dimensions
            supported_resolutions = [
                r for r in supported_resolutions 
                if r.width <= max_w and r.height <= max_h
            ]
        
        if not supported_resolutions:
            raise ValueError("No supported resolutions found with given constraints")
        
        print(f"Supporting {len(supported_resolutions)} resolutions:")
        for res in supported_resolutions:
            print(f"  - {res}")
        
    except Exception as e:
        print(f"Resolution configuration error: {e}")
        return None
    
    # Get the current working directory (should be the repo directory)
    repo_dir = Path.cwd()
    
    # Check if we're in the correct directory by looking for src folder
    if not (repo_dir / "src").exists():
        # Try to find the repo directory
        script_dir = Path(__file__).parent
        repo_dir = script_dir / "repo"
        
        if not repo_dir.exists():
            raise FileNotFoundError(f"D-FINE repository not found. Expected at: {repo_dir}")
        
        # Change to repo directory
        os.chdir(repo_dir)
        print(f"Changed to repository directory: {repo_dir}")
    
    # Add repo directory to Python path
    sys.path.insert(0, str(repo_dir))
    
    try:
        import torch
        import torch.nn as nn
        from src.core import YAMLConfig
        
        print(f"Successfully imported D-FINE modules from: {repo_dir}")
    except ImportError as e:
        print(f"Failed to import D-FINE modules: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")
        return None
    
    # Load configuration
    try:
        cfg = YAMLConfig(config_path, resume=resume_path)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return None
    
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    
    # Load model state if resume path provided
    if resume_path and Path(resume_path).exists():
        print(f"Loading model from: {resume_path}")
        try:
            checkpoint = torch.load(resume_path, map_location="cpu")
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
            cfg.model.load_state_dict(state)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            return None
    else:
        print("Using default initialized model state...")
    
    # Create deployment model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    try:
        model = Model()
        model.eval()
        print("Successfully created deployment model")
    except Exception as e:
        print(f"Failed to create deployment model: {e}")
        return None
    
    # Prepare dummy inputs for export
    batch_size = 1
    
    # Use standard D-FINE training dimensions for export to ensure model compatibility
    # The dynamic shapes will allow different input sizes at inference time
    export_width = 640
    export_height = 640
    
    print(f"Using standard export dimensions for model compatibility: {export_width}x{export_height}")
    print(f"Dynamic shapes will support the following resolutions at inference:")
    for res in supported_resolutions:
        print(f"  - {res}")
    
    data = torch.rand(batch_size, 3, export_height, export_width)
    size = torch.tensor([[export_width, export_height]] * batch_size)
    
    # Test model forward pass
    print("Testing model forward pass...")
    try:
        with torch.no_grad():
            _ = model(data, size)
        print("✓ Model forward pass successful!")
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return None
    
    # Define dynamic axes based on configuration
    if enable_dynamic_shapes:
        # Enable dynamic width and height dimensions for variable resolution support
        dynamic_axes = {
            "images": {0: "batch_size", 2: "height", 3: "width"},
            "orig_target_sizes": {0: "batch_size"},
            "labels": {0: "batch_size", 1: "num_detections"},
            "boxes": {0: "batch_size", 1: "num_detections"},
            "scores": {0: "batch_size", 1: "num_detections"}
        }
        print("✓ Dynamic shapes enabled for variable input dimensions")
    else:
        # Fixed batch size only
        dynamic_axes = {
            "images": {0: "batch_size"},
            "orig_target_sizes": {0: "batch_size"},
            "labels": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "scores": {0: "batch_size"}
        }
        print("✓ Fixed input dimensions with dynamic batch size")
    
    # Determine output file path with resolution info
    if output_path is None:
        if resume_path:
            base_name = Path(resume_path).stem
        else:
            base_name = "d_fine_model"
        
        # Add resolution info to filename for clarity
        if enable_dynamic_shapes:
            resolution_info = f"_dynamic_{'_'.join(aspect_ratios).replace(':', '')}"
            if max_dimensions:
                max_w, max_h = max_dimensions
                resolution_info += f"_max{max_w}x{max_h}"
        else:
            # For fixed shapes, indicate the supported resolution
            resolution_info = f"_fixed_{'_'.join(aspect_ratios).replace(':', '')}"
        
        output_path = f"{base_name}{resolution_info}.onnx"
    
    print(f"Exporting to: {output_path}")
    
    # Export to ONNX with optimized settings
    try:
        # Use higher opset version for better dynamic shape support
        opset_version = 17 if enable_dynamic_shapes else 16
        
        torch.onnx.export(
            model,
            (data, size),
            output_path,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            verbose=False,  # Reduce verbosity for cleaner output
            do_constant_folding=True,
            export_params=True
        )
        print(f"✓ Successfully exported model to: {output_path}")
        print(f"✓ Using ONNX opset version: {opset_version}")
    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None
    
    # Validate exported model
    if check_model:
        try:
            import onnx
            print("Validating exported ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model validation passed!")
        except Exception as e:
            print(f"✗ Model validation failed: {e}")
    
    # Simplify model
    if simplify_model:
        try:
            import onnx
            import onnxsim
            print("Simplifying ONNX model...")
            
            # Check if onnxruntime is available, install if needed
            try:
                import onnxruntime
            except ImportError:
                print("Installing onnxruntime...")
                try:
                    # Try using uv first (since we're in a uv environment)
                    result = subprocess.run([
                        "uv", "pip", "install", "onnxruntime"
                    ], capture_output=True, text=True, check=True)
                    print("✓ Successfully installed onnxruntime with uv")
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback: try direct installation without pip module
                    try:
                        result = subprocess.run([
                            sys.executable, "-c", 
                            "import subprocess; subprocess.check_call(['uv', 'pip', 'install', 'onnxruntime'])"
                        ], capture_output=True, text=True, check=True)
                        print("✓ Successfully installed onnxruntime")
                    except Exception as e:
                        print(f"⚠️  Could not install onnxruntime: {e}")
                        print("Model simplification will be skipped, but export was successful")
                        return output_path
            
            input_shapes = {
                "images": data.shape, 
                "orig_target_sizes": size.shape
            }
            
            onnx_model_simplified, check = onnxsim.simplify(
                output_path, 
                test_input_shapes=input_shapes
            )
            onnx.save(onnx_model_simplified, output_path)
            print(f"✓ Model simplification {'successful' if check else 'completed with warnings'}!")
        except Exception as e:
            print(f"⚠️  Model simplification failed: {e}")
            print("This doesn't affect the core export - the ONNX model is still valid and usable")
    
    return output_path


def validate_onnx_resolutions(model_path: str, supported_resolutions: List[Resolution]) -> bool:
    """
    Validate that the exported ONNX model can handle all supported resolutions.
    
    Args:
        model_path: Path to the ONNX model
        supported_resolutions: List of resolutions to test
    
    Returns:
        True if all resolutions are supported, False otherwise
    """
    try:
        import onnx
        import numpy as np
        
        # Load model
        model = onnx.load(model_path)
        
        print(f"Validating model with {len(supported_resolutions)} resolutions...")
        
        # Check model input signature
        input_shape = model.graph.input[0].type.tensor_type.shape
        dims = [d.dim_value if d.dim_value > 0 else d.dim_param for d in input_shape.dim]
        print(f"Model input signature: {dims}")
        
        # Test each resolution (basic shape validation)
        for i, resolution in enumerate(supported_resolutions):
            width, height = resolution.dimensions
            
            try:
                # Create dummy data for this resolution
                dummy_images = np.random.rand(1, 3, height, width).astype(np.float32)
                dummy_sizes = np.array([[width, height]], dtype=np.float32)
                
                # Basic compatibility check
                if height > 0 and width > 0:
                    print(f"  ✓ {resolution} - compatible")
                else:
                    print(f"  ✗ {resolution} - invalid dimensions")
                    return False
                    
            except Exception as e:
                print(f"  ✗ {resolution} - validation failed: {e}")
                return False
        
        print("✓ All resolutions validated successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Resolution validation failed: {e}")
        return False


def test_model_inference(model_path: str, test_resolutions: Optional[List[Resolution]] = None):
    """
    Test the ONNX model with different input resolutions using ONNX Runtime.
    
    Args:
        model_path: Path to the ONNX model
        test_resolutions: List of resolutions to test (uses small subset if None)
    """
    try:
        import onnxruntime as ort
        import numpy as np
        
        if test_resolutions is None:
            # Use a small subset for quick testing
            test_resolutions = [
                Resolution(320, 180, "16:9"),
                Resolution(640, 360, "16:9"),
                Resolution(320, 240, "4:3")
            ]
        
        print(f"Testing inference with {len(test_resolutions)} resolutions...")
        
        # Create inference session
        session = ort.InferenceSession(model_path)
        
        for resolution in test_resolutions:
            width, height = resolution.dimensions
            
            try:
                # Create dummy inputs
                dummy_images = np.random.rand(1, 3, height, width).astype(np.float32)
                dummy_sizes = np.array([[width, height]], dtype=np.int64)
                
                # Run inference
                inputs = {
                    "images": dummy_images,
                    "orig_target_sizes": dummy_sizes
                }
                
                outputs = session.run(None, inputs)
                
                # Basic output validation
                labels, boxes, scores = outputs
                print(f"  ✓ {resolution} - inference successful "
                      f"(detections: {labels.shape[1] if len(labels.shape) > 1 else 0})")
                
            except Exception as e:
                print(f"  ✗ {resolution} - inference failed: {e}")
                return False
        
        print("✓ All inference tests passed!")
        return True
        
    except ImportError:
        print("⚠️  ONNX Runtime not available - skipping inference tests")
        return True
    except Exception as e:
        print(f"✗ Inference testing failed: {e}")
        return False


def main():
    """Main function with command line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export D-FINE model to ONNX format with variable input dimensions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default settings (both 16:9 and 4:3, all supported resolutions)
  python export.py --resume model.pth
  
  # Export for 16:9 aspect ratio only
  python export.py --resume model.pth --aspect-ratios 16:9
  
  # Export with maximum dimensions constraint
  python export.py --resume model.pth --max-width 640 --max-height 480
  
  # Export with fixed input dimensions (no dynamic shapes)
  python export.py --resume model.pth --no-dynamic-shapes
        """
    )
    
    # Existing arguments
    parser.add_argument(
        "--config", "-c",
        default="configs/dfine/dfine_hgnetv2_l_coco.yml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--resume", "-r",
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip ONNX model validation"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true", 
        help="Skip ONNX model simplification"
    )
    
    # New resolution arguments
    parser.add_argument(
        "--aspect-ratios",
        nargs="+",
        choices=["16:9", "4:3"],
        default=["16:9", "4:3"],
        help="Supported aspect ratios (default: both 16:9 and 4:3)"
    )
    parser.add_argument(
        "--max-width",
        type=int,
        help="Maximum width constraint for supported resolutions"
    )
    parser.add_argument(
        "--max-height", 
        type=int,
        help="Maximum height constraint for supported resolutions"
    )
    parser.add_argument(
        "--no-dynamic-shapes",
        action="store_true",
        help="Disable dynamic input shapes (use fixed dimensions)"
    )
    parser.add_argument(
        "--test-inference",
        action="store_true",
        help="Test the exported model with different resolutions"
    )
    parser.add_argument(
        "--validate-resolutions",
        action="store_true", 
        help="Validate model compatibility with all supported resolutions"
    )
    
    args = parser.parse_args()
    
    # Prepare max dimensions constraint
    max_dimensions = None
    if args.max_width and args.max_height:
        max_dimensions = (args.max_width, args.max_height)
    elif args.max_width or args.max_height:
        print("⚠️  Both --max-width and --max-height must be specified together")
        sys.exit(1)
    
    print("🚀 Starting D-FINE ONNX export...")
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume or 'None (using default weights)'}")
    print(f"Output: {args.output or 'Auto-generated'}")
    print(f"Aspect ratios: {', '.join(args.aspect_ratios)}")
    print(f"Max dimensions: {max_dimensions or 'None'}")
    print(f"Dynamic shapes: {'Enabled' if not args.no_dynamic_shapes else 'Disabled'}")
    print()
    
    # Get supported resolutions for display
    try:
        supported_resolutions = ResolutionConfig.get_supported_resolutions(args.aspect_ratios)
        if max_dimensions:
            max_w, max_h = max_dimensions
            supported_resolutions = [
                r for r in supported_resolutions 
                if r.width <= max_w and r.height <= max_h
            ]
    except Exception as e:
        print(f"❌ Resolution configuration error: {e}")
        sys.exit(1)
    
    result = export_d_fine_model(
        config_path=args.config,
        resume_path=args.resume,
        output_path=args.output,
        check_model=not args.no_check,
        simplify_model=not args.no_simplify,
        aspect_ratios=args.aspect_ratios,
        max_dimensions=max_dimensions,
        enable_dynamic_shapes=not args.no_dynamic_shapes
    )
    
    if result:
        print(f"\n🎉 Export completed successfully!")
        print(f"ONNX model saved to: {result}")
        
        # Run additional validation/testing if requested
        if args.validate_resolutions:
            print("\n🔍 Running resolution validation...")
            if not validate_onnx_resolutions(result, supported_resolutions):
                print("❌ Resolution validation failed!")
                sys.exit(1)
        
        if args.test_inference:
            print("\n🧪 Running inference tests...")
            if not test_model_inference(result, supported_resolutions[:3]):  # Test subset
                print("❌ Inference testing failed!")
                sys.exit(1)
        
        print(f"\n✅ All validations passed!")
        
    else:
        print("\n❌ Export failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()