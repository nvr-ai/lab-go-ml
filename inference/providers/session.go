// Package providers - Inference sessions.
package providers

import (
	"fmt"
	"os"

	"github.com/nvr-ai/go-ml/images"
	ort "github.com/yalue/onnxruntime_go"
)

// Session represents a model session from the onnxruntime.
type Session struct {
	Session *ort.AdvancedSession
	Inputs  []ort.ArbitraryTensor
	Outputs []ort.ArbitraryTensor
}

// Close releases the resources associated with the Session.
//
// Returns:
//   - No return values.
func (s *Session) Close() error {
	if s.Inputs != nil {
		for _, input := range s.Inputs {
			input.Destroy()
		}
		s.Inputs = nil
	}

	if s.Outputs != nil {
		for _, output := range s.Outputs {
			output.Destroy()
		}
		s.Outputs = nil
	}

	if s.Session != nil {
		err := s.Session.Destroy()
		if err != nil {
			return fmt.Errorf("error destroying ORT session: %w", err)
		}
		s.Session = nil
	}

	return nil
}

// NewSessionArgs represents the arguments for creating a new ONNX detector session.
type NewSessionArgs struct {
	// The path to the ONNX model file.
	ModelPath string
	// The inputs of the model.
	Inputs []images.Rect
	// The outputs of the model.
	Outputs []images.Rect
}

// NewSession creates a new ONNX detector session.
//
// This function creates a new ONNX Runtime session with preallocated input and output tensors,
// sets up session options, and execution providers (EPs).
//
// Order of operations:
//  1. Library path check: Ensures native runtime is accessible.
//  2. Environment setup: Required to prepare ONNX Runtime internals.
//  3. Tensor allocation: Prepares fixed-shape buffers for input/output data.
//
// 4. Session options: Controls performance and hardware acceleration (e.g., threading, optimization
// level). 5. Execution providers: Enables GPU or optimized CPU paths if configured (e.g., CoreML,
// OpenVINO).
//  6. Session creation: Loads model and binds resources, creating the runnable inference engine.
//  7. Resource management: Defensive cleanup to avoid native leaks.
//
// **Note: Cleanup of tensors must be handled by the caller.**
//
// Arguments:
//   - provider: The provider for the session.
//   - args: The arguments for the session.
//
// Returns:
//   - *Session: Wrapped Session struct that holds the native session and tensors for inference.
//   - error: An error if the session creation fails.
func NewSession(provider ExecutionProvider, args NewSessionArgs) (*Session, error) {
	// Check if the shared library exists before trying to use it.
	libPath := GetSharedLibPath()
	if _, err := os.Stat(libPath); os.IsNotExist(err) {
		return nil, fmt.Errorf(
			"ONNX Runtime library not found at %s. On macOS ARM64, you need to build ONNX Runtime from source or disable ONNX Runtime. Error: %w",
			libPath,
			err,
		)
	}

	// (optionally) Enable verbose logging for troubleshooting native ONNX Runtime operations.
	ort.SetEnvironmentLogLevel(ort.LoggingLevelVerbose)
	// Point ONNX Runtime to the exact shared library path (overrides default search).
	ort.SetSharedLibraryPath(libPath)

	// Initialize the ONNX Runtime environment (native layer setup).
	// Required once per process; it loads the native library and prepares internal state.
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("error initializing ORT environment: %w", err)
	}

	// Create the shapes for the input tensors to align with a typical image input that
	// follows the common deep learning format [batch, channels, height, width].
	// The shapes are copied, and are no longer needed after this function returns.
	var inputs []ort.ArbitraryTensor
	for _, rect := range args.Inputs {
		inputTensor, err := ort.NewEmptyTensor[float32](
			// TODO: Determine if images.Rect should be float32 vs. int64 for onnx tensoring.
			ort.NewShape(1, 3, int64(rect.X1), int64(rect.Y1), int64(rect.X2), int64(rect.Y2)),
		)
		if err != nil {
			inputTensor.Destroy()
			return nil, fmt.Errorf("error creating input tensor: %w", err)
		}
		inputs = append(inputs, inputTensor)
	}

	// Create the shape for the output tensor to align with the expected output of a YOLO-style
	// detector (e.g., 8400 anchors × 84 values). Similar to the input tensor allocation, but for
	// model outputs where dimensions and size depend on the specific model architecture (e.g.,
	// detection outputs).
	// The shape is copied, and is no longer needed after this function returns.
	var outputs []ort.ArbitraryTensor
	for _, rect := range args.Outputs {
		outputTensor, err := ort.NewEmptyTensor[float32](
			// TODO: Determine if images.Rect should be float32 vs. int64 for onnx tensoring.
			ort.NewShape(1, 84, int64(rect.X1), int64(rect.Y1), int64(rect.X2), int64(rect.Y2)),
		)
		if err != nil {
			outputTensor.Destroy()
			return nil, fmt.Errorf("error creating output tensor: %w", err)
		}
		outputs = append(outputs, outputTensor)
	}

	// Create session options that control execution behavior and optimizations.
	// Session options configure how ONNX Runtime executes your model, such as threading,
	// optimization level, and hardware acceleration.
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("error creating ORT session options: %w", err)
	}
	defer options.Destroy()

	// Set intra-op parallelism threads for node execution inside the model graph (e.g., matrix
	// multiplication).
	options.SetIntraOpNumThreads(0)
	// Set inter-op parallelism threads for parallel execution of independent graph nodes (e.g.,
	// parallel layers).
	options.SetInterOpNumThreads(0)
	// Enables advanced graph rewrites (e.g., fusion, constant folding) to improve performance during
	// graph loading.
	// Graph optimizations can fuse operations, remove redundancies, and boost runtime speed.
	options.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableExtended)

	// Execution Providers (EPs) let ONNX Runtime leverage specialized hardware or optimized
	// libraries. This block configures EPs like CoreML (Apple GPU) or OpenVINO (Intel CPU/GPU) if
	// requested in config.
	// Proper EP setup can dramatically accelerate inference.
	switch provider.Backend() {
	case CoreMLProviderBackend:
		err = options.AppendExecutionProviderCoreML(0)
		if err != nil {
			return nil, fmt.Errorf("error enabling CoreML: %w", err)
		}
	case OpenVINOProviderBackend:
		opts, ok := provider.Options().(OpenVINOOptions)
		if !ok {
			return nil, fmt.Errorf("invalid options type for OpenVINO: %T", provider.Options())
		}
		// See:
		// https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#summary-of-options
		config := map[string]string{
			"device_id":              opts.DeviceID,
			"device_type":            opts.DeviceType,
			"precision":              fmt.Sprintf("%f", opts.Precision),
			"num_of_threads":         fmt.Sprintf("%d", opts.NumOfThreads),
			"disable_dynamic_shapes": fmt.Sprintf("%t", opts.DisableDynamicShapes),
			"model_priority":         fmt.Sprintf("%d", opts.ModelPriority),
		}
		err = options.AppendExecutionProviderOpenVINO(config)
		if err != nil {
			return nil, fmt.Errorf("error enabling OpenVINO: %w", err)
		}
	case CUDAProviderBackend:
		opts, ok := provider.Options().(CUDAOptions)
		if !ok {
			return nil, fmt.Errorf("invalid options type for CUDA: %T", provider.Options())
		}
		cuda, err := opts.ToNativeProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("error converting CUDA options: %w", err)
		}
		err = options.AppendExecutionProviderCUDA(cuda)
		if err != nil {
			return nil, fmt.Errorf("error enabling CUDA: %w", err)
		}
	}

	// Finally, create an advanced ONNX Runtime session binding input/output tensors with options.
	//
	// This call ties everything together:
	// - Loads the ONNX model into an inference session.
	// - Binds preallocated input/output tensors for zero-copy data exchange.
	// - Applies all configured options and execution providers.
	//
	// An error here means model loading or setup failed (invalid model, incompatible ops, etc).
	session, err := ort.NewAdvancedSession(
		args.ModelPath,      // Path to ONNX model file
		[]string{"images"},  // Input node names expected by model
		[]string{"output0"}, // Output node names expected by model
		inputs,              // Preallocated input tensors
		outputs,             // Preallocated output tensors
		options,             // Session options configured above
	)
	if err != nil {
		for _, input := range inputs {
			input.Destroy()
		}
		for _, output := range outputs {
			output.Destroy()
		}
		return nil, fmt.Errorf("error creating ORT session: %w", err)
	}

	// Return the session with preallocated tensors and options.
	// **Cleanup of tensors must be handled by the caller.**
	return &Session{
		Session: session,
		Inputs:  inputs,
		Outputs: outputs,
	}, nil
}
