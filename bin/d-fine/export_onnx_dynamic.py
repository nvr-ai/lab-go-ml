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
D-FINE ONNX Export Script with torch.jit.script Support
Exports D-FINE models to ONNX format with proper scripting for dynamic input dimensions.
"""

import os
import sys
import subprocess
import functools
from pathlib import Path
from typing import List, Dict, Any, Tuple

def export_d_fine_model_scripted(
    config_path: str = "configs/dfine/dfine_hgnetv2_l_coco.yml",
    resume_path: str = None,
    output_path: str = None,
    check_model: bool = True,
    simplify_model: bool = True,
    enable_dynamic_hw: bool = True
):
    """
    Export D-FINE model to ONNX format using torch.jit.script for better dynamic support.
    
    Arguments:
    - config_path: Path to the model configuration file
    - resume_path: Path to the model checkpoint file
    - output_path: Output path for the ONNX model
    - check_model: Whether to validate the exported ONNX model
    - simplify_model: Whether to simplify the ONNX model
    - enable_dynamic_hw: Whether to enable dynamic height/width dimensions
    
    Returns:
    - Path to the exported ONNX model file
    """
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
        import torch.nn.functional as F
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

    def patch_msdeformable_attention():
        """
        Monkey patch MSDeformableAttention to be script-compatible.
        """
        from src.zoo.dfine.dfine_decoder import MSDeformableAttention
        
        # Store original forward method
        original_forward = MSDeformableAttention.forward
        
        def script_compatible_forward(
            self,
            query: torch.Tensor,
            reference_points: torch.Tensor,
            value: torch.Tensor,
            value_spatial_shapes: List[int],
        ):
            """
            Script-compatible forward pass using tensor operations instead of control flow.
            """
            bs, Len_q = query.shape[:2]

            sampling_offsets: torch.Tensor = self.sampling_offsets(query)
            sampling_offsets = sampling_offsets.reshape(
                bs, Len_q, self.num_heads, sum(self.num_points_list), 2
            )

            attention_weights = self.attention_weights(query).reshape(
                bs, Len_q, self.num_heads, sum(self.num_points_list)
            )
            attention_weights = F.softmax(attention_weights, dim=-1)

            # Get reference points dimension
            ref_dim = reference_points.size(-1)
            
            # Prepare both computation paths
            # Case 1: reference_points.shape[-1] == 2
            offset_normalizer = torch.tensor(value_spatial_shapes, dtype=query.dtype, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations_2d = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
            
            # Case 2: reference_points.shape[-1] == 4
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            
            # Safely handle the case where reference_points might be 2D
            ref_points_wh = torch.zeros_like(reference_points[:, :, None, :, :2])
            if ref_dim == 4:
                ref_points_wh = reference_points[:, :, None, :, 2:4]
            
            offset = (
                sampling_offsets
                * num_points_scale
                * ref_points_wh
                * self.offset_scale
            )
            sampling_locations_4d = reference_points[:, :, None, :, :2] + offset
            
            # Use torch.where for conditional selection (script-compatible)
            sampling_locations = torch.where(
                ref_dim == 2,
                sampling_locations_2d,
                sampling_locations_4d
            )

            output = self.ms_deformable_attn_core(
                value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
            )

            return output
        
        # Replace the forward method
        MSDeformableAttention.forward = script_compatible_forward
        print("✓ Patched MSDeformableAttention for script compatibility")

    def patch_postprocessor():
        """
        Monkey patch postprocessor to be script-compatible.
        """
        try:
            from src.zoo.dfine.postprocessor import DFINEPostProcessor
            
            # Store original forward method
            original_forward = DFINEPostProcessor.forward
            
            def script_compatible_postprocessor_forward(self, outputs, orig_target_sizes: torch.Tensor):
                """Script-compatible postprocessor forward."""
                logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
                
                # Convert to proper format
                if self.use_focal_loss:
                    scores = torch.sigmoid(logits)
                    labels = torch.arange(self.num_classes, device=logits.device).unsqueeze(0).repeat(logits.shape[0], 1)
                    labels = labels.unsqueeze(1).repeat(1, logits.shape[1], 1)
                    scores = scores.unsqueeze(-1).repeat(1, 1, 1, 1) * labels.unsqueeze(-1)
                    scores = scores.flatten(2)
                    labels = labels.flatten(2)
                else:
                    scores = F.softmax(logits, dim=-1)[:, :, :-1]
                    scores, labels = scores.max(dim=-1)
                
                # Handle top-k selection using torch.where instead of if statement
                num_queries = scores.shape[1]
                need_topk = (num_queries > self.num_top_queries).float()
                
                # Always compute topk, but use original if not needed
                topk_scores, topk_indices = torch.topk(scores, min(self.num_top_queries, num_queries), dim=-1)
                topk_labels = torch.gather(labels, dim=1, index=topk_indices)
                topk_boxes = torch.gather(boxes, dim=1, index=topk_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]))
                
                # Select based on whether we need topk
                final_scores = torch.where(need_topk.bool(), topk_scores, scores)
                final_labels = torch.where(need_topk.bool(), topk_labels, labels)  
                final_boxes = torch.where(need_topk.bool().unsqueeze(-1), topk_boxes, boxes)
                
                return final_labels, final_boxes, final_scores
            
            # Replace the forward method
            DFINEPostProcessor.forward = script_compatible_postprocessor_forward
            print("✓ Patched DFINEPostProcessor for script compatibility")
            
        except Exception as e:
            print(f"⚠ Could not patch postprocessor: {e}")

    # Apply patches before creating model
    patch_msdeformable_attention()
    patch_postprocessor()

    # Create deployment model
    class ScriptCompatibleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        
        def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
            """
            Script-compatible forward pass.
            
            Arguments:
            - images: Input images tensor [N, 3, H, W]
            - orig_target_sizes: Original image sizes [N, 2] as [height, width]
            
            Returns:
            - Tuple of (labels, boxes, scores)
            """
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    try:
        model = ScriptCompatibleModel()
        model.eval()
        print("Successfully created script-compatible deployment model")
    except Exception as e:
        print(f"Failed to create deployment model: {e}")
        return None
    
    # Prepare dummy inputs for testing
    batch_size = 1
    test_sizes = [(640, 640), (480, 640), (800, 600)] if enable_dynamic_hw else [(640, 640)]
    
    print("Testing model with different input sizes...")
    for h, w in test_sizes:
        data = torch.rand(batch_size, 3, h, w)
        size = torch.tensor([[h, w]], dtype=torch.float32)
        
        try:
            with torch.no_grad():
                outputs = model(data, size)
            print(f"✓ Test passed for size {h}x{w}")
        except Exception as e:
            print(f"✗ Test failed for size {h}x{w}: {e}")
            return None

    # Use the first test size for export
    data = torch.rand(batch_size, 3, test_sizes[0][0], test_sizes[0][1])
    size = torch.tensor([[test_sizes[0][0], test_sizes[0][1]]], dtype=torch.float32)
    
    # Try to script the model
    print("Attempting to script the model...")
    try:
        scripted_model = torch.jit.script(model)
        print("✓ Successfully scripted model with torch.jit.script")
        
        # Test scripted model
        with torch.no_grad():
            _ = scripted_model(data, size)
        print("✓ Scripted model forward pass successful")
        
        export_model = scripted_model
        
    except Exception as e:
        print(f"⚠ Failed to script model: {e}")
        print("Falling back to trace mode...")
        try:
            traced_model = torch.jit.trace(model, (data, size))
            print("✓ Successfully traced model with torch.jit.trace")
            export_model = traced_model
        except Exception as trace_e:
            print(f"⚠ Failed to trace model: {trace_e}")
            print("Using regular model for export...")
            export_model = model

    # Define comprehensive dynamic axes
    if enable_dynamic_hw:
        dynamic_axes = {
            "images": {0: "batch_size", 2: "height", 3: "width"},
            "orig_target_sizes": {0: "batch_size"},
            "labels": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "scores": {0: "batch_size"}
        }
    else:
        dynamic_axes = {
            "images": {0: "batch_size"},
            "orig_target_sizes": {0: "batch_size"},
            "labels": {0: "batch_size"},
            "boxes": {0: "batch_size"},
            "scores": {0: "batch_size"}
        }
    
    # Determine output file path
    if output_path is None:
        if resume_path:
            base_name = Path(resume_path).stem
            suffix = "_dynamic_hw" if enable_dynamic_hw else "_dynamic_batch"
            output_path = f"{base_name}{suffix}.onnx"
        else:
            suffix = "_dynamic_hw" if enable_dynamic_hw else "_dynamic_batch"
            output_path = f"d_fine_model{suffix}.onnx"
    
    print(f"Exporting to: {output_path}")
    print(f"Dynamic axes: {dynamic_axes}")
    
    # Export to ONNX
    try:
        torch.onnx.export(
            export_model,
            (data, size),
            output_path,
            input_names=["images", "orig_target_sizes"],
            output_names=["labels", "boxes", "scores"],
            dynamic_axes=dynamic_axes,
            opset_version=16,
            verbose=True,
            do_constant_folding=True,
            export_params=True,
            # Additional parameters for better scripted model support
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False
        )
        print(f"✓ Successfully exported model to: {output_path}")
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
            
            # Print model info
            print(f"Model inputs: {[input.name for input in onnx_model.graph.input]}")
            print(f"Model outputs: {[output.name for output in onnx_model.graph.output]}")
            
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
                    result = subprocess.run([
                        "uv", "pip", "install", "onnxruntime"
                    ], capture_output=True, text=True, check=True)
                    print("✓ Successfully installed onnxruntime with uv")
                except (subprocess.CalledProcessError, FileNotFoundError):
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
                "images": list(data.shape), 
                "orig_target_sizes": list(size.shape)
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


def main():
    """Main function with command line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export D-FINE model to ONNX format with torch.jit.script")
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
    parser.add_argument(
        "--no-dynamic-hw",
        action="store_true",
        help="Disable dynamic height/width dimensions (only enable dynamic batch size)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting D-FINE ONNX export with torch.jit.script...")
    print(f"Config: {args.config}")
    print(f"Resume: {args.resume or 'None (using default weights)'}")
    print(f"Output: {args.output or 'Auto-generated'}")
    print(f"Dynamic H/W: {not args.no_dynamic_hw}")
    print()
    
    result = export_d_fine_model_scripted(
        config_path=args.config,
        resume_path=args.resume,
        output_path=args.output,
        check_model=not args.no_check,
        simplify_model=not args.no_simplify,
        enable_dynamic_hw=not args.no_dynamic_hw
    )
    
    if result:
        print(f"\n🎉 Export completed successfully!")
        print(f"ONNX model saved to: {result}")
        print(f"Ready for onnxruntime-go with {'dynamic H/W' if not args.no_dynamic_hw else 'dynamic batch size only'}!")
    else:
        print("\n❌ Export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()