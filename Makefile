PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

format: rustfmt							## Run all formatting scripts
	make -C python format
.PHONY: format

fmt: format
.PHONY: fmt

rustfmt:								## Run rustfmt
	cargo fmt -- src/*.rs tests/*.rs benches/*.rs
	@echo -e "\e[1;32mrustfmt clean!\e[0m"
	make -C python rustfmt
.PHONY: rustfmt

lint: typos reuse ruff clippy			## Run all linting scripts
	make -C python lint
.PHONY: lint

typos:			## Run typos over the source code and documentation
	typos --sort
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

clippy:			## Run clippy lint checks
	cargo clippy --all-targets --all-features
	@echo -e "\e[1;32mclippy clean!\e[0m"
	make -C python clippy
.PHONY: clippy

ruff:			## Run ruff checks over the source code
	ruff check python scripts
	@echo -e "\e[1;32mruff lint clean!\e[0m"
.PHONY: ruff

# }}}

# {{{ testing

pin:			## Pin dependencies versions to requirements.txt
	cargo update
	make -C python pin
.PHONY: pin

build:			## Build the project in debug mode
	cargo build --locked --all-features --verbose
.PHONY: build

test:			## Run cargo test
	RUST_BACKTRACE=1 cargo test --tests
.PHONY: test

# }}}

# {{{

clean:			## Remove various build artifacts
	rm -rf target
	make -C python clean
.PHONY: clean

purge: clean	## Remove various temporary files
	rm -rf .ruff_cache
	make -C python purge
.PHONY: purge

# }}}

