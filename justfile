PYTHON := "python -X dev"
PYPROJECT := "python/pyproject.toml"

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc('Reformat all source code')]
format: isort black rustfmt justfmt

[doc('Run ruff isort fixes over the source code')]
isort:
    ruff check --config {{ PYPROJECT }} --fix --select=I scripts benches
    ruff check --config {{ PYPROJECT }} --fix --select=RUF022 scripts benches
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc('Run ruff format over the source code')]
black:
    ruff format --config {{ PYPROJECT }} scripts benches
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc("Run rustfmt over the source code")]
rustfmt:
    cargo fmt -- src/*.rs tests/*.rs benches/*.rs
    @echo -e "\e[1;32mrustfmt clean!\e[0m"

[doc('Run just --fmt over the justfile')]
justfmt:
    just --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

# }}}
# {{{ linting

[doc('Run all linting checks over the source code')]
lint: typos reuse ruff clippy

[doc('Run typos over the source code and documentation')]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc('Check REUSE license compliance')]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc('Run ruff checks over the source code')]
ruff:
    ruff check --config {{ PYPROJECT }} scripts benches
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc("Run clippy lint checks")]
clippy:
    cargo clippy --all-targets --all-features
    @echo -e "\e[1;32mclippy clean!\e[0m"

# }}}
# {{{ testing

[doc("Pin dependencies in Cargo.lock")]
pin:
    cargo update --verbose

[doc("Build the project in debug mode")]
build:
    cargo build --locked --all-features --verbose

[doc("Run cargo tests")]
test:
    RUST_BACKTRACE=1 cargo test --tests

# }}}
# {{{ cleanup

[doc("Remove build artifacts")]
clean:
    rm -rf target
    @just python/clean

[doc("Remove all generated files")]
purge: clean
    rm -rf .ruff_cache
    @just python/purge

# }}}
