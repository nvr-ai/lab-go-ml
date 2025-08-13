.PHONY: llms.txt diagrams

# watch: ## Build the docs and watch for changes.
# 	trap 'kill 0' SIGINT; \
# 	npm run docs:watch & \
# 	(sleep 4 && npm run docs:serve) & \
# 	wait

diagrams:
	@for file in diagrams/*.mmd; do \
		npx mmdc \
		--input $$file \
		--output $${file%.mmd}.png \
		--configFile diagrams/mermaid.json \
		--cssFile diagrams/diagrams.css; \
		echo "Generated $$file"; \
	done

watch:
	@find diagrams -not -path "*/*.png" | entr -c make diagrams