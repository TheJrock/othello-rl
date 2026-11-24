# Every step of the project as one command (spec §4.1). `make help` lists them.
# The report's reproducibility appendix is this file.

PY ?= .venv/bin/python
JOBS ?= 4
# training device for the sweep targets: cpu | cuda | mps (8×8 needs a GPU, spec §4.8)
DEVICE ?= cpu
SERVER ?=
SERVER_DIR ?= othello-rl
RUN ?=

.PHONY: help setup test test-py test-web data ladder sweep-4x4 sweep-4x4-old pilot-6x6 pilot-6x6-old sweep-6x6 pilot-8x8 sweep-8x8 sync figures report demo web-dev

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

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
	$(PY) scripts/build_data.py openings --board 8
	$(PY) scripts/build_data.py endgame6
	$(PY) scripts/build_data.py regret-positions

# Current settings -> runs/ladder-tuned. runs/ladder holds the pre-Nov-24 baseline: never overwrite it.
LADDER_EXP ?= ladder-tuned

ladder:  ## debugging ladder rungs 1-3 on 4x4 + exact checks (spec §4.6). DEVICE=mps: ~10 min
	$(PY) -m othello_rl.train --config configs/4x4.yaml --exp $(LADDER_EXP) --target env --sampling random --device $(DEVICE)
	$(PY) -m othello_rl.train --config configs/4x4.yaml --exp $(LADDER_EXP) --target env --sampling naive --device $(DEVICE)
	$(PY) -m othello_rl.train --config configs/4x4.yaml --exp $(LADDER_EXP) --target negamax --sampling naive --device $(DEVICE)
	$(PY) scripts/ladder_check.py runs/$(LADDER_EXP)/env-random/seed0 runs/$(LADDER_EXP)/env-naive/seed0 runs/$(LADDER_EXP)/negamax-naive/seed0

sweep-4x4:  ## E1: 4x4, 2x2 conditions x 5 seeds   (JOBS=8 to run 8 at once)
	$(PY) scripts/sweep.py --config configs/4x4.yaml --exp e1-4x4 --seeds 0 1 2 3 4 --jobs $(JOBS) -- --device $(DEVICE)

# E7 (Could, spec §4.8): the same code with the pre-Nov-24 settings, to compare learning against E1 / P
OLD_KNOBS = --eps-end 0.05 --target-update-every 1000

sweep-4x4-old:  ## E7: E1 with the pre-Nov-24 settings (eps_end 0.05, target_update_every 1000)
	$(PY) scripts/sweep.py --config configs/4x4.yaml --exp e7-4x4-old --seeds 0 1 2 3 4 --jobs $(JOBS) -- --device $(DEVICE) $(OLD_KNOBS)

pilot-6x6:  ## P: 6x6, 2x2 conditions x 1 seed
	$(PY) scripts/sweep.py --config configs/6x6.yaml --exp pilot-6x6 --seeds 0 --jobs $(JOBS) -- --device $(DEVICE)

pilot-6x6-old:  ## E7: the 6x6 pilots with the pre-Nov-24 settings (only with spare compute)
	$(PY) scripts/sweep.py --config configs/6x6.yaml --exp e7-6x6-old --seeds 0 --jobs $(JOBS) -- --device $(DEVICE) $(OLD_KNOBS)

sweep-6x6:  ## E2: 6x6, 2x2 conditions x 3 seeds
	$(PY) scripts/sweep.py --config configs/6x6.yaml --exp e2-6x6 --seeds 0 1 2 --jobs $(JOBS) -- --device $(DEVICE)

pilot-8x8:  ## P8: 8x8 env-naive, 1 seed; gate 1 must pass first (spec §4.8). Needs DEVICE=mps or cuda
	$(PY) scripts/sweep.py --config configs/8x8.yaml --exp pilot-8x8 --targets env --samplings naive --seeds 0 --jobs 1 -- --device $(DEVICE)

sweep-8x8:  ## E6: 8x8 env-naive + env-pool, 1 seed; gate 2 must pass first. Needs DEVICE=mps or cuda
	$(PY) scripts/sweep.py --config configs/8x8.yaml --exp e6-8x8 --targets env --seeds 0 --jobs $(JOBS) -- --device $(DEVICE)

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
