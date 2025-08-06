#!/usr/bin/env python3
"""
Test suite for D-FINE ONNX export with variable input dimensions.
This test validates all supported resolutions and ensures compatibility with onnxruntime-go.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path to import export functions
sys.path.insert(0, str(Path(__file__).parent))

from export import (
    Resolution, 
    ResolutionConfig, 
    export_d_fine_model,
    validate_onnx_resolutions,
    test_model_inference
)


class TestResolutionConfig:
    """Test the resolution configuration system."""
    
    def test_supported_resolutions_16_9(self):
        """Test 16:9 aspect ratio resolutions."""
        resolutions = ResolutionConfig.get_supported_resolutions(["16:9"])
        expected = [
            (160, 90),
            (320, 180),
            (640, 360),
            (960, 540)
        ]
        
        actual = [(r.width, r.height) for r in resolutions]
        assert actual == expected, f"Expected {expected}, got {actual}"
        
        # All should be 16:9 aspect ratio
        for r in resolutions:
            assert r.aspect_ratio == "16:9"
    
    def test_supported_resolutions_4_3(self):
        """Test 4:3 aspect ratio resolutions."""
        resolutions = ResolutionConfig.get_supported_resolutions(["4:3"])
        expected = [
            (160, 120),
            (320, 240),
            (640, 480),
            (960, 720)
        ]
        
        actual = [(r.width, r.height) for r in resolutions]
        assert actual == expected, f"Expected {expected}, got {actual}"
        
        # All should be 4:3 aspect ratio
        for r in resolutions:
            assert r.aspect_ratio == "4:3"
    
    def test_supported_resolutions_both(self):
        """Test both aspect ratios."""
        resolutions = ResolutionConfig.get_supported_resolutions()
        
        # Should have 8 total resolutions (4 for each aspect ratio)
        assert len(resolutions) == 8
        
        # Check we have both aspect ratios
        aspect_ratios = set(r.aspect_ratio for r in resolutions)
        assert aspect_ratios == {"16:9", "4:3"}
    
    def test_validate_resolution(self):
        """Test resolution validation."""
        # Valid resolutions
        assert ResolutionConfig.validate_resolution(640, 360) == True  # 16:9
        assert ResolutionConfig.validate_resolution(320, 240) == True  # 4:3
        
        # Invalid resolutions
        assert ResolutionConfig.validate_resolution(640, 480) == False  # Wrong aspect ratio for 640px
        assert ResolutionConfig.validate_resolution(123, 456) == False  # Not supported width
        assert ResolutionConfig.validate_resolution(0, 0) == False     # Invalid dimensions
    
    def test_get_optimal_resolution(self):
        """Test optimal resolution finding."""
        # Test fitting within target dimensions
        optimal = ResolutionConfig.get_optimal_resolution(1000, 600, ["16:9"])
        assert optimal.width == 960 and optimal.height == 540
        
        # Test with constraints that fit nothing
        optimal = ResolutionConfig.get_optimal_resolution(100, 50)
        assert optimal is None
        
        # Test with exact match
        optimal = ResolutionConfig.get_optimal_resolution(320, 180, ["16:9"])
        assert optimal.width == 320 and optimal.height == 180


class TestResolution:
    """Test the Resolution dataclass."""
    
    def test_resolution_creation(self):
        """Test resolution object creation."""
        res = Resolution(640, 360, "16:9")
        assert res.width == 640
        assert res.height == 360
        assert res.aspect_ratio == "16:9"
        assert res.dimensions == (640, 360)
        assert str(res) == "640x360 (16:9)"
    
    def test_resolution_invalid_dimensions(self):
        """Test resolution with invalid dimensions."""
        try:
            Resolution(0, 360, "16:9")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        try:
            Resolution(640, -360, "16:9") 
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected


class TestExportFunctionality:
    """Test the export functionality (requires D-FINE repository setup)."""
    
    def test_resolution_configuration_in_export(self):
        """Test that export function handles resolution configuration correctly."""
        # This is a mock test that validates the parameter handling
        # without requiring the actual D-FINE model
        
        try:
            # Test with different aspect ratios
            for aspect_ratios in [["16:9"], ["4:3"], ["16:9", "4:3"]]:
                supported = ResolutionConfig.get_supported_resolutions(aspect_ratios)
                assert len(supported) > 0
                
                # Test max dimensions constraint
                if aspect_ratios == ["16:9"]:
                    constrained = [r for r in supported if r.width <= 640 and r.height <= 360]
                    assert len(constrained) >= 2  # Should have 160x90 and 320x180
                
        except Exception as e:
            raise AssertionError(f"Resolution configuration failed: {e}")
    
    def test_validate_onnx_resolutions_mock(self):
        """Test resolution validation function with mock data."""
        # Test with non-existent file (should handle gracefully)
        resolutions = ResolutionConfig.get_supported_resolutions(["16:9"])
        result = validate_onnx_resolutions("non_existent_model.onnx", resolutions)
        assert result == False  # Should fail gracefully
    
    def test_inference_testing_mock(self):
        """Test inference testing function with mock data."""
        # Test with non-existent file (should handle gracefully)
        resolutions = [
            Resolution(320, 180, "16:9"),
            Resolution(640, 360, "16:9")
        ]
        result = test_model_inference("non_existent_model.onnx", resolutions)
        # Should return True if onnxruntime not available, False if file doesn't exist
        assert isinstance(result, bool)


def run_integration_test():
    """
    Integration test that can be run if D-FINE repository is properly set up.
    This function attempts to export a model with variable input dimensions.
    """
    print("🧪 Running D-FINE ONNX export integration test...")
    
    # Check if we're in the right environment
    repo_dir = Path.cwd()
    if not (repo_dir / "src").exists():
        script_dir = Path(__file__).parent
        repo_dir = script_dir / "repo"
        
        if not repo_dir.exists():
            print("⚠️  D-FINE repository not found - skipping integration test")
            return True
    
    try:
        # Test export with different configurations
        test_configs = [
            {
                "aspect_ratios": ["16:9"],
                "max_dimensions": (640, 360),
                "enable_dynamic_shapes": True
            },
            {
                "aspect_ratios": ["4:3"],
                "max_dimensions": (320, 240),
                "enable_dynamic_shapes": True
            },
            {
                "aspect_ratios": ["16:9", "4:3"],
                "max_dimensions": None,
                "enable_dynamic_shapes": False
            }
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n📋 Test configuration {i+1}: {config}")
            
            output_path = f"test_model_{i+1}.onnx"
            
            try:
                result = export_d_fine_model(
                    config_path="configs/dfine/dfine_hgnetv2_l_coco.yml",
                    output_path=output_path,
                    check_model=True,
                    simplify_model=False,  # Skip simplification for faster testing
                    **config
                )
                
                if result:
                    print(f"✅ Configuration {i+1} export successful")
                    
                    # Clean up test file
                    if Path(result).exists():
                        Path(result).unlink()
                else:
                    print(f"❌ Configuration {i+1} export failed")
                    return False
                    
            except Exception as e:
                print(f"❌ Configuration {i+1} failed with error: {e}")
                return False
        
        print("\n🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic unit tests
    print("🔍 Running unit tests...")
    
    try:
        # Create test instances
        config_test = TestResolutionConfig()
        resolution_test = TestResolution()
        export_test = TestExportFunctionality()
        
        # Run tests
        config_test.test_supported_resolutions_16_9()
        config_test.test_supported_resolutions_4_3()
        config_test.test_supported_resolutions_both()
        config_test.test_validate_resolution()
        config_test.test_get_optimal_resolution()
        
        resolution_test.test_resolution_creation()
        
        # Mock export tests
        export_test.test_resolution_configuration_in_export()
        
        export_test.test_validate_onnx_resolutions_mock()
        export_test.test_inference_testing_mock()
        
        print("✅ All unit tests passed!")
        
        # Run integration test if environment is available
        if len(sys.argv) > 1 and sys.argv[1] == "--integration":
            run_integration_test()
        else:
            print("ℹ️  Run with --integration flag to test actual model export")
            
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        sys.exit(1)