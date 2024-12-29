PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

format: isort black rustfmt pyproject	## Run all formatting scripts
.PHONY: format

fmt: format
.PHONY: fmt

isort:									## Run ruff isort fixes over the source code
	ruff check --fix --select=I python
	ruff check --fix --select=RUF022 python
	@echo -e "\e[1;32mruff isort clean!\e[0m"
.PHONY: isort

black:								## Run ruff format over the source code
	ruff format python
	@echo -e "\e[1;32mruff format clean!\e[0m"
.PHONY: black

rustfmt:						## Run rustfmt
	cargo fmt -- src/*.rs
	@echo -e "\e[1;32mrustfmt clean!\e[0m"
.PHONY: rustfmt

pyproject:							## Run pyproject-fmt over the configuration
	$(PYTHON) -m pyproject_fmt \
		--indent 4 --max-supported-python '3.13' \
		pyproject.toml
	@echo -e "\e[1;32mpyproject clean!\e[0m"
.PHONY: pyproject

lint: typos reuse ruff mypy			## Run all linting scripts
.PHONY: lint

typos:			## Run typos over the source code and documentation
	typos --sort
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

ruff:			## Run ruff checks over the source code
	ruff check python
	@echo -e "\e[1;32mruff lint clean!\e[0m"
.PHONY: ruff

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy python
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

# }}}

# {{{ testing

REQUIREMENTS=\
	requirements-dev.txt \
	requirements.txt

requirements-dev.txt: pyproject.toml
	uv pip compile --upgrade --universal --python-version '3.10' \
		--extra dev --extra docs \
		-o $@ $<
.PHONY: requirements-dev.txt

requirements.txt: pyproject.toml
	uv pip compile --upgrade --universal --python-version '3.10' \
		-o $@ $<
.PHONY: requirements.txt

pin: $(REQUIREMENTS)	## Pin dependencies versions to requirements.txt
.PHONY: pin

pip-install:			## Install pinned dependencies from requirements.txt
	$(PYTHON) -m pip install --upgrade pip wheel maturin
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --verbose --editable --no-build-isolation .
.PHONY: pip-install

build:			## Build the project in debug mode
	cargo build --locked --all-features --verbose
.PHONY: build

pytest:					## Run pytest tests
	$(PYTHON) -m pytest
.PHONY: pytest

test:					## Run cargo test
	RUST_BACKTRACE=1 cargo test --all-features
.PHONY: test

# }}}

# {{{

clean:						## Remove various build artifacts
	rm -rf *.png
	rm -rf build dist
	rm -rf docs/_build
.PHONY: clean

purge: clean				## Remove various temporary files
	rm -rf .ruff_cache .pytest_cache .mypy_cache
.PHONY: purge

# }}}

