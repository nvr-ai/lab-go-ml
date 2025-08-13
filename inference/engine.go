// Package inference - Inference engine interface and implementations.
package inference

import (
	"context"
	"errors"
	"image"

	"github.com/nvr-ai/go-ml/inference/detectors"
	"github.com/nvr-ai/go-ml/inference/providers"
	"github.com/nvr-ai/go-ml/models"
	"github.com/nvr-ai/go-ml/models/model"
)

// Engine defines the interface for ML inference engines
type Engine interface {
	Predict(ctx context.Context, img image.Image) (interface{}, error)
	Close() error
}

// EngineBuilder helps build test scenarios with fluent API.
type EngineBuilder struct {
	engine   Engine
	provider providers.ExecutionProvider
	model    model.Model
	session  providers.Session
	detector *detectors.Detector
	err      error
}

// NewEngineBuilder creates a new engine builder.
//
// Returns:
//   - *EngineBuilder: The engine builder.
func NewEngineBuilder() *EngineBuilder {
	return &EngineBuilder{}
}

// WithProvider sets the provider for the engine.
//
// Arguments:
//   - provider: The provider to use for the engine.
//
// Returns:
//   - *EngineBuilder: The engine builder.
func (b *EngineBuilder) WithProvider(args providers.Config) *EngineBuilder {
	if b.HasError() {
		return b
	}

	provider, err := providers.NewProvider(args.Options)
	if err != nil {
		b.err = err
		return b
	}
	b.provider = provider
	return b
}

// WithModel sets the model for the engine.
//
// Arguments:
//   - args: The model arguments.
//
// Returns:
//   - *EngineBuilder: The engine builder.
func (b *EngineBuilder) WithModel(args model.NewModelArgs) *EngineBuilder {
	if b.err != nil {
		return b
	}
	model, err := models.NewModel(args)
	if err != nil {
		b.err = err
		return b
	}
	b.model = model

	opts := b.model.Options()

	session, err := providers.NewSession(b.provider, providers.NewSessionArgs{
		ModelPath: opts.Path,
		Inputs:    opts.Inputs,
		Outputs:   opts.Outputs,
	})
	if err != nil {
		b.err = err
		return b
	}
	b.session = *session

	return b
}

// WithDetector sets the detector for the engine.
//
// Arguments:
//   - cfg: The detector configuration.
//
// Returns:
//   - *EngineBuilder: The engine builder.
func (b *EngineBuilder) WithDetector(cfg detectors.Config) *EngineBuilder {
	if b.HasError() {
		return b
	}

	detector, err := detectors.NewDetector(b.provider, b.model, cfg)
	if err != nil {
		b.err = err
		return b
	}
	b.detector = detector
	return b
}

// HasError checks if the engine builder has errors.
//
// Returns:
//   - bool: True if there are errors, false otherwise.
func (b *EngineBuilder) HasError() bool {
	return b.err != nil
}

// engine implements the Engine interface.
type engine struct {
	provider providers.ExecutionProvider
	model    model.Model
	session  providers.Session
	detector *detectors.Detector
}

// Predict predicts the output of the model.
//
// Arguments:
//   - ctx: The context for the prediction.
//   - img: The image to predict.
//
// Returns:
//   - interface{}: The output of the model.
//   - error: The error if any.
func (e *engine) Predict(ctx context.Context, img image.Image) (interface{}, error) {
	return e.detector.Predict(ctx, img)
}
func (e *engine) Close() error {
	return e.session.Close()
}

// MustBuild builds the engine and panics if there is an error.
//
// Returns:
//   - *Engine: The engine.
func (b *EngineBuilder) MustBuild() Engine {
	e, err := b.Build()
	if err != nil {
		panic(err)
	}
	return e
}

// Build builds the engine.
//
// Returns:
//   - *Engine: The engine.
//   - error: The error if any.
func (b *EngineBuilder) Build() (Engine, error) {
	if b.HasError() {
		return nil, b.err
	}
	if b.provider == nil {
		return nil, errors.New("provider not configured")
	}
	if b.detector == nil {
		return nil, errors.New("detector not configured")
	}
	if b.model == nil {
		return nil, errors.New("model not configured")
	}

	return &engine{
		provider: b.provider,
		detector: b.detector,
		model:    b.model,
	}, nil
}
