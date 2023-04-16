PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

fmt: black		## Run all formatting scripts
	$(PYTHON) -m pyproject_fmt --indent 4 pyproject.toml
	$(PYTHON) -m isort python/mittagleffler tests
	rustfmt src/*.rs
.PHONY: fmt

black:			## Run black over the source code
	$(PYTHON) -m black \
		--safe --target-version py38 --preview \
		python/mittagleffler tests
.PHONY: black

flake8:			## Run flake8 checks over the source code
	PYTHONWARNINGS=ignore $(PYTHON) -m flake8 python/mittagleffler tests
	@echo -e "\e[1;32mflake8 clean!\e[0m"
.PHONY: flake8

pylint:			## Run pylint checks over the source code
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint python/mittagleffler tests/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"
.PHONY: pylint

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy \
		--strict --show-error-codes \
		python/mittagleffler tests
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

reuse:			## Check REUSE license compliance
	reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

# }}}

# {{{ testing

install:				## Install dependencies
	$(PYTHON) -m pip install --upgrade pip wheel maturin
	maturin develop --extras dev
.PHONY: pip-install

test:					## Run pytest tests
	$(PYTHON) -m pytest -rswx --durations=25 -v -s
.PHONY: test

# }}}

ctags:			## Regenerate ctags
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python
.PHONY: ctags
