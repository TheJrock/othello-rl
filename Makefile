# Every step of the project as one command (spec §4.1). `make help` lists them.
# The report's reproducibility appendix is this file.

PY ?= .venv/bin/python
JOBS ?= 4
SERVER ?=
SERVER_DIR ?= othello-rl
RUN ?=

.PHONY: help setup test test-py test-web data sweep-4x4 pilot-6x6 sweep-6x6 sync figures report demo web-dev

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'

setup:  ## create .venv (Python 3.12) and install Python + web dependencies
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ".[dev]"
	cd web && npm install

test: test-py test-web  ## run all tests (stubs show up as "skipped: TODO ...")

test-py:
	$(PY) -m pytest

test-web:
	cd web && npm run typecheck && npm test

data:  ## build the fixed benchmarks in data/ (refuses to overwrite existing files)
	$(PY) scripts/build_data.py solve4x4
	$(PY) scripts/build_data.py openings --board 4
	$(PY) scripts/build_data.py openings --board 6
	$(PY) scripts/build_data.py endgame6
	$(PY) scripts/build_data.py regret-positions

sweep-4x4:  ## E1: 4x4, 2x2 conditions x 5 seeds   (JOBS=8 to run 8 at once)
	$(PY) scripts/sweep.py --config configs/4x4.yaml --exp e1-4x4 --seeds 0 1 2 3 4 --jobs $(JOBS)

pilot-6x6:  ## P: 6x6, 2x2 conditions x 1 seed
	$(PY) scripts/sweep.py --config configs/6x6.yaml --exp pilot-6x6 --seeds 0 --jobs $(JOBS)

sweep-6x6:  ## E2: 6x6, 2x2 conditions x 3 seeds
	$(PY) scripts/sweep.py --config configs/6x6.yaml --exp e2-6x6 --seeds 0 1 2 --jobs $(JOBS)

sync:  ## copy runs/ back from the server: make sync SERVER=user@host
	@test -n "$(SERVER)" || (echo "usage: make sync SERVER=user@host [SERVER_DIR=othello-rl]"; exit 1)
	rsync -av $(SERVER):$(SERVER_DIR)/runs/ runs/

figures:  ## runs/ -> figures/ and tables/
	$(PY) scripts/analyze.py --exp e1-4x4 e2-6x6

report:  ## build report/main.pdf (needs tectonic: `brew install tectonic`, or latexmk from MacTeX)
	cd report && if command -v tectonic >/dev/null; then tectonic main.tex; else latexmk -pdf main.tex; fi

demo:  ## export a run for the web demo: make demo RUN=runs/e2-6x6/env-pool/seed0
	@test -n "$(RUN)" || (echo "usage: make demo RUN=runs/<exp>/<target>-<sampling>/seed<k>"; exit 1)
	$(PY) scripts/export_demo.py --run $(RUN)

web-dev:  ## serve the demo locally
	cd web && npm run dev
