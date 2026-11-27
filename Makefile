.PHONY: venv py-build py-test cpp-build cpp-test test lint docs clean

PYTHON ?= .venv/bin/python
CPP_BUILD_DIR ?= build/cpp

venv:
	uv venv --python 3.12 .venv
	uv pip install --python $(PYTHON) numpy pandas scikit-learn pytest ruff mypy nanobind scikit-build-core ninja

py-build:
	uv pip install --python $(PYTHON) --no-build-isolation -e .

py-test: py-build
	$(PYTHON) -m pytest tests/python -v --tb=short

cpp-build:
	cmake -S . -B $(CPP_BUILD_DIR) -DMITER_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
	cmake --build $(CPP_BUILD_DIR) --parallel

cpp-test: cpp-build
	ctest --test-dir $(CPP_BUILD_DIR) --output-on-failure

test: cpp-test py-test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy

# Starlight site in docs/dist (needs Node.js); the Python reference comes from the installed package.
docs: py-build
	cd docs && $(abspath $(PYTHON)) scripts/api_reference.py && $(abspath $(PYTHON)) scripts/release_notes.py && npm ci && npm run build

clean:
	rm -rf build/ .pytest_cache/ docs/dist/
