.PHONY: test
.DEFAULT_GOAL := help

export SHELL=/bin/bash
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.24

tidy: ## Clean and pull in dependencies.
	go mod tidy

webcam: tidy ## Run the webcam command.
	go run ./cmd/gocv/webcam/main.go

setup: ## Create a virtual environment.
	@if [ ! -d ".venv" ]; then uv venv; fi
	@uv sync
	@echo "Run `source .venv/bin/activate` to activate the virtual environment."

help:
	@sed '1s/^# //' docs/readme.md | head -n 1
	@echo
	@echo "Usage:"
	@printf "  make \033[36m<target>\033[0m [<variable>=<value>,...]\n"
	@echo
	@echo "Commands:"
	@grep -E '^[a-z\/-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'


test: ## Run all tests.
	go test ./...