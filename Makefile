# Strata: run the model and the usual chores.  `make` lists the targets.
#
# Written for GNU Make 3.81, the one macOS ships in /usr/bin (Apple stays on it: 3.82 moved to GPLv3), so it uses no
# .ONESHELL, no `!=` and no `::=`: every recipe line is its own shell.  It works the same on Linux.
#
#   make run                       start the model in this terminal (Ctrl-C stops it)
#   make start / status / stop     the same in the background, with its log in strata-run.log
#   make chat PROMPT="Hi"          one request through the OpenAI API

# your own defaults (PORT := 8090, API_KEY := ...), kept out of git: Makefile.local
-include Makefile.local

# what Strata.app and its helpers are signed with: an "Apple Development" identity when the keychain has one, else
# ad-hoc.  macOS keeps a permission (the Downloads folder, the microphone, Reminders) for a signer's identity; an ad-hoc
# signature is the binary's hash, new on every build, so each build asked again and the app waited on a prompt
SIGN ?= $(shell security find-identity -v -p codesigning 2>/dev/null | sed -n 's/.*\([0-9A-F]\{40\}\) "Apple Development.*/\1/p' | head -1)
SIGN_ID = $(if $(SIGN),$(SIGN),-)

PORT        ?= 8080
HOST        ?= 127.0.0.1
API_KEY     ?=
MODEL       ?=
CONTEXT     ?=
SETUP_ARGS  ?=
PROMPT      ?= Say hello in one short sentence.
EFFORT      ?= none
MAX_TOKENS  ?= 200
REASONING_BUDGET ?=
WAIT_S      ?= 600
PY          := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CMAKE       := $(if $(wildcard .venv/bin/cmake),.venv/bin/cmake,cmake)
# rustup puts cargo in ~/.cargo/bin and on PATH only for shells started after it: found there too
CARGO       := $(or $(shell command -v cargo 2> /dev/null),$(wildcard $(HOME)/.cargo/bin/cargo),cargo)
URL         := http://$(if $(filter 0.0.0.0,$(HOST)),127.0.0.1,$(HOST)):$(PORT)
AUTH        := $(if $(API_KEY),-H "Authorization: Bearer $(API_KEY)",)
RUN_ARGS    := --port $(PORT) $(if $(filter-out 127.0.0.1,$(HOST)),--host $(HOST),) $(if $(API_KEY),--api-key $(API_KEY),)
SERVER_PAT  := strata-server --engine strata .*--port $(PORT)
# Linux on an Apple Silicon Mac (Asahi): the same engine as macOS on Vulkan; setup, build and run take that path
# (bootstrap.py's apple_silicon(): the device tree's "apple,...", else the CPU's implementer 0x61 in /proc/cpuinfo)
APPLE_LINUX := $(shell { grep -qa 'apple,' /proc/device-tree/compatible || grep -q 'CPU implementer.*0x61' /proc/cpuinfo; } 2> /dev/null && echo 1)
CONFIG      ?= $(firstword $(shell ls -t strata-*.json 2> /dev/null))
SERVER_RUN  := engine-metal/strata-server --engine strata --config $(CONFIG) $(RUN_ARGS)

# fails when another program listens on PORT: names it, and how to pick another port
PORT_FREE   = @who=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN 2> /dev/null | awk 'NR==2 {print $$1 " (pid " $$2 ")"}'); \
	if [ -n "$$who" ]; then \
	  if pgrep -f "$(SERVER_PAT)" > /dev/null; then echo "Strata already runs on port $(PORT): make status / make stop"; \
	  else echo "port $(PORT) is used by $$who - start Strata on another one: make $@ PORT=8090 (or put PORT := 8090 in Makefile.local)"; fi; \
	  exit 1; \
	fi

# make run CONTEXT=131072: sets --max-context in the installed config first (tools/set_context.py; it stays set)
SET_CONTEXT = $(if $(CONTEXT),$(PY) tools/set_context.py $(CONTEXT),@true)

# the prompt reaches `make chat` through the environment, so quotes and apostrophes in it need no shell quoting
export PROMPT MAX_TOKENS EFFORT REASONING_BUDGET

# A/B of two engine builds: with and without metal/patches/ (docs/MACOS.md, "For developers")
AB_A        ?= build-metal-a/metal/strata-metal
AB_B        ?= build-metal-b/metal/strata-metal
AB_GGUF     ?=
AB_PROMPTS  ?=
AB_ROUNDS   ?= 7

.PHONY: help setup pull check update run start stop status wait chat models build build-ab server afm app app-run app-stop tui tui-build test test-server test-engine ab clean linux-package linux-deliver linux

help:
	@echo "Strata - make targets (variables: PORT=$(PORT) HOST=$(HOST) API_KEY=... MODEL=... SETUP_ARGS=...;"
	@echo "                       defaults of your own in Makefile.local, e.g. PORT := 8090)"
	@echo ""
	@echo "  First time: make check, make pull MODEL=Q2_0, make run.  On a Mac: docs/MACOS.md"
	@echo ""
	@echo "  make setup        install or change a model (asks; SETUP_ARGS=\"--model Q2_0 --yes\" for no questions)"
	@echo "  make pull          build the engine and download a model, without starting it: make pull MODEL=Q2_0"
	@echo "                     (from Hugging Face into ../Strata-data, resumable: run it again if it stops)"
	@echo "                     MODEL: Q2_0 IQ2_XS IQ3_XXS IQ3_S IQ1_M UD-Q4_K_XL UD-IQ4_XS (setup), or GLM-5.3"
	@echo "                     GPT-OSS-120B GPT-OSS-120B-HERETIC GPT-OSS-120B-DFLASH GPT-OSS-20B QWEN3.6-35B QWEN3-CODER-30B QWEN3-CODER-30B-Q4 NUCLEUS"
	@echo "                     HUNYUAN WAN2.2"
	@echo "                     (tools/pull_model.py; GPT-OSS-120B-DFLASH, NUCLEUS and HUNYUAN convert too; WAN2.2"
	@echo "                     builds stable-diffusion.cpp's sd-cli for videos)"
	@echo "  make check         only check what this computer can run"
	@echo "  make update        update the engine and settings without starting"
	@echo "  make run           start the model here, in the foreground (Ctrl-C stops it); CONTEXT=131072 sets the"
	@echo "                     context window first (4096-262144 tokens, it stays set; a bigger one needs more memory)"
	@echo "  make start         start it in the background (log: strata-run.log) and wait until it answers"
	@echo "  make status        is it up?  (GET /health)"
	@echo "  make stop          stop the background server"
	@echo "  make server        build the server (server/, Rust) into engine-metal/strata-server"
	@echo "  make app           build Strata.app (the server, then the app): one window with the server's page (Chat,"
	@echo "                     Monitor, Firewall, About) under a control bar (model, start/stop/restart, load/unload, log)"
	@echo "                     and a menu-bar icon; the server is its child on the config's port (8090)"
	@echo "  make app-run       build it, stop a running Strata.app (and its server) and open the new one"
	@echo "  make app-stop      quit Strata.app: its server sends QUIT to the engine and frees its memory first"
	@echo "  make tui           Strata.app in a terminal (Linux on Apple Silicon, or over SSH): chat, monitor, memory, log"
	@echo "  make linux-package the engine (CPU), server and app for an x86_64 Linux PC, built here in Docker for its CPU"
	@echo "                     [LINUX_HOST=dadmin@vizio.local LINUX_DIR=/home/dadmin/Strata LINUX_CPU=ivybridge]"
	@echo "  make linux-deliver that package into LINUX_DIR/engine-metal on LINUX_HOST, and Strata in its applications"
	@echo "  make linux         push this commit there, package, deliver, and open (or switch) Strata on its desktop"
	@echo "  make chat          one chat request: make chat PROMPT=\"Write a haiku\" [MAX_TOKENS=100 EFFORT=none|low|medium|high]"
	@echo "                     REASONING_BUDGET=N: at most N tokens of thinking, then it answers (0: no cap; it counts"
	@echo "                     toward MAX_TOKENS, so keep it 64+ below; EFFORT=none turns thinking off)"
	@echo "  make models        GET /v1/models"
	@echo "  make build         compile the engine here (macOS: the Metal engine; elsewhere setup builds it)"
	@echo "  make test          the tests that need no GPU and no model"
	@echo "  make test-engine   the Metal engine protocol tests: TEST_GGUF=<a small .gguf with <|im_start|>>"
	@echo "  make build-ab      build the plain (A) and patched (B) Metal engines for an A/B"
	@echo "  make ab            A/B two engines: AB_GGUF=<shard 1> AB_PROMPTS=<ids.json> [AB_A=... AB_B=...]"
	@echo "  make clean         remove the build folders"

# on a Mac: metal/onboard_mac.py does everything (tools, builds, the model, its run config, a first answer, the app);
# elsewhere setup.py.  python3 bootstrap.py --setup is setup.py's installer on a Mac too (the Qwen3.8 quants, MTP)
setup:
ifneq ($(filter Darwin,$(shell uname -s))$(APPLE_LINUX),)
	python3 bootstrap.py --onboard $(if $(MODEL),--model $(MODEL),) $(SETUP_ARGS)
else
	python3 bootstrap.py --setup $(if $(MODEL),--model $(MODEL),) $(SETUP_ARGS)
endif

# setup's own download (Hugging Face, resumable; HF_ENDPOINT for a mirror, SETUP_ARGS="--data-dir DIR" for elsewhere);
# the models setup does not install (GLM-5.3, GPT-OSS-120B, GPT-OSS-120B-HERETIC, GPT-OSS-120B-DFLASH, GPT-OSS-20B, QWEN3.6-35B, QWEN3-CODER-30B, QWEN3-CODER-30B-Q4, NUCLEUS, HUNYUAN, WAN2.2) through tools/pull_model.py
PULL_EXTRA  := GLM-5.3 GPT-OSS-120B GPT-OSS-120B-HERETIC GPT-OSS-120B-DFLASH GPT-OSS-20B QWEN3.6-35B QWEN3-CODER-30B QWEN3-CODER-30B-Q4 LFM2.5-24B LFM2.5-24B-IQ3 LFM2.5-8B LFM2.5-8B-Q4 QWEN3-4B QWEN3-4B-Q4 NUCLEUS HUNYUAN WAN2.2 glm-5.3 gpt-oss-120b gpt-oss-120b-heretic gpt-oss-120b-dflash gpt-oss-20b qwen3.6-35b qwen3-coder-30b qwen3-coder-30b-q4 lfm2.5-24b lfm2.5-24b-iq3 lfm2.5-8b lfm2.5-8b-q4 qwen3-4b qwen3-4b-q4 nucleus hunyuan wan2.2
DATA_DIR    ?= ../Strata-data
pull:
	$(if $(filter $(MODEL),$(PULL_EXTRA)),$(PY) tools/pull_model.py $(MODEL) $(DATA_DIR),python3 bootstrap.py --setup --no-start $(if $(MODEL),--model $(MODEL),) $(SETUP_ARGS))
ifneq ($(filter Darwin,$(shell uname -s))$(APPLE_LINUX),)
	@# a chat model pulled by hand gets its run config too (the context the engine measures to fit this Mac)
	@if [ -n "$(filter $(MODEL),$(PULL_EXTRA))" ] && [ -x engine-metal/strata-metal ]; then \
	  .venv/bin/python metal/onboard_mac.py --config-only $(MODEL) --data-dir $(DATA_DIR); \
	elif [ -n "$(filter $(MODEL),$(PULL_EXTRA))" ]; then echo "next: make build, then make pull MODEL=$(MODEL) again (it writes the run config)"; fi
endif

check:
	python3 bootstrap.py --check

update:
	./update.sh

run:
	$(PORT_FREE)
	$(SET_CONTEXT)
ifeq ($(APPLE_LINUX),1)
	$(SERVER_RUN)
else
	python3 bootstrap.py $(RUN_ARGS) $(SETUP_ARGS)
endif

start:
	$(PORT_FREE)
	$(SET_CONTEXT)
	@echo "starting Strata on $(URL) (log: strata-run.log) ..."
ifeq ($(APPLE_LINUX),1)
	@nohup $(SERVER_RUN) > strata-run.log 2>&1 &
else
	@nohup python3 bootstrap.py $(RUN_ARGS) --no-browser $(SETUP_ARGS) > strata-run.log 2>&1 &
endif
	@$(MAKE) --no-print-directory wait

# /health says 503 while the model loads, 200 once it is ready
wait:
	@i=0; while [ $$i -lt $(WAIT_S) ]; do \
	  code=$$(curl -s -o /dev/null -w '%{http_code}' -m 2 $(URL)/health); \
	  if [ "$$code" = "200" ]; then echo "ready: $(URL)/v1 (OpenAI), $(URL)/v1/messages (Anthropic)"; exit 0; fi; \
	  if ! pgrep -f "$(SERVER_PAT)" > /dev/null && [ $$i -gt 20 ]; then echo "the server is not running: see strata-run.log"; tail -5 strata-run.log; exit 1; fi; \
	  sleep 2; i=$$((i + 2)); \
	done; echo "not ready after $(WAIT_S) s: see strata-run.log"; exit 1

status:
	@if ! pgrep -f "$(SERVER_PAT)" > /dev/null; then \
	  who=$$(lsof -nP -iTCP:$(PORT) -sTCP:LISTEN 2> /dev/null | awk 'NR==2 {print $$1 " (pid " $$2 ")"}'); \
	  echo "Strata is not running on port $(PORT)$${who:+ - $$who listens there}"; exit 1; \
	fi; \
	code=$$(curl -s -o /dev/null -w '%{http_code}' -m 5 $(URL)/health); \
	case "$$code" in 200) echo "up: $(URL)";; 503) echo "loading: $(URL)";; *) echo "starting (no answer yet): $(URL)";; esac

# SIGTERM, not SIGINT: a job started with `&` by a non-interactive shell has SIGINT ignored (POSIX), and Python then
# installs no Ctrl+C handler; the server maps SIGTERM to Ctrl+C's path (QUIT to the engine, #96).  A second SIGTERM
# ends the engine at once; SIGKILL only when even that does not end it.
stop:
	@pids=$$(pgrep -f "$(SERVER_PAT)"); \
	if [ -z "$$pids" ]; then echo "no Strata server on port $(PORT)"; exit 0; fi; \
	kill -TERM $$pids; \
	i=0; while kill -0 $$pids 2> /dev/null && [ $$i -lt 30 ]; do sleep 1; i=$$((i + 1)); done; \
	if kill -0 $$pids 2> /dev/null; then kill -TERM $$pids; sleep 5; fi; \
	if kill -0 $$pids 2> /dev/null; then kill -KILL $$pids; sleep 1; fi; \
	if kill -0 $$pids 2> /dev/null; then echo "still running: $$pids"; exit 1; fi; \
	echo "stopped"

# One request (tools/chat.py: it checks MAX_TOKENS and REASONING_BUDGET before sending, and exits 1 without an
# answer).  REASONING_BUDGET=N: at most N tokens of thinking (the server's reasoning_budget_tokens), then it answers;
# 0 = no cap (EFFORT=none turns thinking off).  The answer is on stdout, the token count on stderr.
chat:
	@URL="$(URL)" API_KEY="$(API_KEY)" $(PY) tools/chat.py

models:
	@curl -sS -m 10 $(AUTH) $(URL)/v1/models; echo

build: server
ifeq ($(shell uname -s),Darwin)
	$(CMAKE) -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON
	$(CMAKE) --build build-metal --target strata-metal strata-vision strata-directions -j
	@# installed where the run configs point; rm first: macOS kills a program whose binary was rewritten in place
	rm -f engine-metal/strata-metal engine-metal/strata-vision engine-metal/strata-directions engine-metal/strata-afm engine-metal/strata-voice
	cp build-metal/metal/strata-metal build-metal/metal/strata-vision build-metal/metal/strata-directions engine-metal/
	@# Apple's on-device model (macOS 26+): strata-afm for "engine": "afm" configs (make afm); skipped without FoundationModels
	@swiftc -O -parse-as-library metal/afm/strata_afm.swift -o engine-metal/strata-afm 2>/dev/null || echo "strata-afm not built: Apple's on-device model needs macOS 26 or later"
	@# "Hey Strata" (the chat's Voice button): Apple's on-device dictation; its Info.plist inside, for the microphone's prompt
	@rm -f engine-metal/strata-voice; swiftc -O -parse-as-library metal/voice/strata_voice.swift -o engine-metal/strata-voice \
	  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker metal/voice/Info.plist 2>/dev/null || echo "strata-voice not built: voice needs macOS 26 or later"
	@# macOS Reminders for the model (the chat's settings switch): an MCP server over EventKit
	@rm -f engine-metal/strata-reminders; swiftc -O -parse-as-library metal/reminders/strata_reminders.swift -o engine-metal/strata-reminders \
	  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker metal/reminders/Info.plist 2>/dev/null || echo "strata-reminders not built (macOS 14 or later)"
	@for h in strata-afm strata-voice strata-reminders; do [ -f engine-metal/$$h ] && codesign --force --sign $(SIGN_ID) engine-metal/$$h 2>/dev/null || codesign --force --sign - engine-metal/$$h; done; true
else ifeq ($(APPLE_LINUX),1)
	@# Linux on Apple Silicon: llama.cpp's Vulkan backend (metal/CMakeLists.txt picks it off macOS)
	$(CMAKE) -S . -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON
	$(CMAKE) --build build-metal --target strata-metal strata-vision -j
	mkdir -p engine-metal && rm -f engine-metal/strata-metal engine-metal/strata-vision
	cp build-metal/metal/strata-metal build-metal/metal/strata-vision engine-metal/
else ifeq ($(shell uname -s),Linux)
	@# another Linux PC: the same engine on the CPU (llama.cpp's CPU backend, built for this CPU; LINUX_CPU_ARGS for
	@# another one: make linux-package builds for the target PC's instruction set, not the build machine's)
	$(CMAKE) -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON -DSTRATA_METAL_CPU=ON $(LINUX_CPU_ARGS)
	$(CMAKE) --build build-metal --target strata-metal strata-vision -j
	mkdir -p engine-metal && rm -f engine-metal/strata-metal engine-metal/strata-vision
	cp build-metal/metal/strata-metal build-metal/metal/strata-vision engine-metal/
else
	@echo "on Windows setup builds or downloads the engine: make setup (or python3 bootstrap.py --setup --build)"
endif

# Strata for an x86_64 Linux PC, built here in Docker (linux/Dockerfile; on a Mac under Rosetta) at the path it runs
# from there (the app and the server keep their folder's path from the build), for that PC's CPU, not this one's.
# LINUX_DIR there is a git checkout of the same commit (make linux pushes it): the package is the binaries.
LINUX_HOST ?= dadmin@vizio.local
LINUX_DIR  ?= /home/dadmin/Strata
LINUX_CPU  ?= ivybridge
LINUX_WORK ?= $(abspath ../Strata-linux-build)
# ivybridge (a 2012 i7): AVX and F16C, no AVX2 / FMA; anything else: the x86-64-v3 set (AVX2, FMA)
LINUX_CPU_FLAGS = $(if $(filter ivybridge,$(LINUX_CPU)),-DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=ON -DGGML_BMI2=OFF,-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON)
LINUX_SHA  = $(shell git rev-parse --short=8 HEAD)
LINUX_PKG  = $(LINUX_WORK)/strata-linux-x86_64-$(LINUX_SHA).tar.gz
# strata-metal-vulkan: the same engine on the PC's GPUs through Vulkan (Mesa's NVK, ANV/hasvk): a config's "exe" picks it
LINUX_BINS = engine-metal/strata-metal engine-metal/strata-metal-vulkan engine-metal/strata-vision engine-metal/strata-server engine-metal/strata-files engine-metal/strata-app

linux-package:
	@test -z "$$(git status --porcelain --untracked-files=no)" || { echo "commit first: the package is built from HEAD (git archive)"; exit 1; }
	docker build -q -t strata-linux-build linux
	mkdir -p $(LINUX_WORK)/src $(LINUX_WORK)/cargo
	git archive HEAD | tar -x -C $(LINUX_WORK)/src
	docker run --rm --platform linux/amd64 -v $(LINUX_WORK)/src:$(LINUX_DIR) -v $(LINUX_WORK)/cargo:/opt/cargo/registry -w $(LINUX_DIR) -e STRATA_BUILD_SHA=$(LINUX_SHA) \
	  strata-linux-build sh -c 'make build LINUX_CPU_ARGS="-DGGML_NATIVE=OFF $(LINUX_CPU_FLAGS)" && make app \
	  && cmake -S . -B build-metal-vk -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON -DGGML_NATIVE=OFF $(LINUX_CPU_FLAGS) \
	  && cmake --build build-metal-vk --target strata-metal -j && cp build-metal-vk/metal/strata-metal engine-metal/strata-metal-vulkan'
	tar -C $(LINUX_WORK)/src -czf $(LINUX_PKG) $(LINUX_BINS)
	@echo "package: $(LINUX_PKG); make linux-deliver puts it on $(LINUX_HOST)"

linux-deliver:
	@test -f $(LINUX_PKG) || { echo "no $(LINUX_PKG): make linux-package first"; exit 1; }
	@ssh $(LINUX_HOST) "cd $(LINUX_DIR) && git rev-parse --short=8 HEAD" | grep -qx $(LINUX_SHA) || { echo "$(LINUX_HOST):$(LINUX_DIR) is not at $(LINUX_SHA): make linux pushes it"; exit 1; }
	scp -q $(LINUX_PKG) $(LINUX_HOST):/tmp/strata-linux.tar.gz
	ssh $(LINUX_HOST) "cd $(LINUX_DIR) && mkdir -p engine-metal && rm -f $(LINUX_BINS) && tar -xzf /tmp/strata-linux.tar.gz && rm -f /tmp/strata-linux.tar.gz && echo $(LINUX_SHA) > engine-metal/.build && ls -l engine-metal"
	ssh $(LINUX_HOST) "mkdir -p ~/.local/share/applications && printf '[Desktop Entry]\nType=Application\nName=Strata\nComment=Local AI models\nExec=$(LINUX_DIR)/engine-metal/strata-app\nIcon=utilities-terminal\nTerminal=false\nCategories=Development;\n' > ~/.local/share/applications/strata.desktop"

linux:
	git push -q $(LINUX_HOST):$(notdir $(LINUX_DIR)) HEAD:mac3
	$(MAKE) linux-package linux-deliver
	ssh $(LINUX_HOST) "cd $(LINUX_DIR) && engine-metal/strata-app --deploy"

# the server (server/, Rust): built and put beside the engine, where setup and the run scripts start it
# the tool-call grammar against llama.cpp's own GBNF validator (the engine's llama.cpp, after make metal)
LLAMA_SRC = $(firstword $(wildcard build-metal/_deps/llama_cpp-src-*))
LLAMA_LIB = build-metal/_deps/llama_cpp-build
grammar-check:
	c++ -std=c++17 -O1 -I$(LLAMA_SRC)/include -I$(LLAMA_SRC)/ggml/include -I$(LLAMA_SRC)/src $(LLAMA_SRC)/tests/test-gbnf-validator.cpp \
	  $(LLAMA_LIB)/src/libllama.a $(LLAMA_LIB)/ggml/src/libggml.a $(LLAMA_LIB)/ggml/src/libggml-cpu.a $(LLAMA_LIB)/ggml/src/ggml-metal/libggml-metal.a \
	  $(LLAMA_LIB)/ggml/src/ggml-blas/libggml-blas.a $(LLAMA_LIB)/ggml/src/libggml-base.a -framework Metal -framework Foundation -framework Accelerate \
	  -framework MetalKit -o build-metal/gbnf-validator
	cd server && STRATA_GBNF_VALIDATOR=$(CURDIR)/build-metal/gbnf-validator cargo test --release --lib the_tool_grammar

server:
	$(CARGO) build --release --manifest-path server/Cargo.toml
	@# rm first: a new file (macOS kills a program whose binary was rewritten in place)
	mkdir -p engine-metal && rm -f engine-metal/strata-server engine-metal/strata-files && cp server/target/release/strata-server server/target/release/strata-files engine-metal/

# the Mac app (app/, Rust): Strata.app in this folder runs the server as its child with full control (config, port,
# start/stop/restart, load/unload, live status, log, chat window, menu-bar icon); quitting it stops the server
# Apple's on-device model (macOS 26+, Apple Intelligence on): its run config strata-apple-afm.json, no download
afm:
	@swiftc -O -parse-as-library metal/afm/strata_afm.swift -o engine-metal/strata-afm.new && mv -f engine-metal/strata-afm.new engine-metal/strata-afm
	.venv/bin/python metal/onboard_mac.py --config-only AFM

app: server
	@# the app runs engine-metal/strata-metal, which only make build refreshes: say so when its source is newer
	@[ ! -f engine-metal/strata-metal ] || [ engine-metal/strata-metal -nt metal/strata_metal.cpp ] || echo "warning: engine-metal/strata-metal is older than metal/strata_metal.cpp: run make build too (the app keeps using the old engine)"
	$(CARGO) build --release --manifest-path app/Cargo.toml
ifneq ($(shell uname -s),Darwin)
	@# Linux: the app beside the engine, and in the desktop's applications (GTK, WebKitGTK and a tray icon:
	@# libgtk-3-dev libwebkit2gtk-4.1-dev libayatana-appindicator3-dev libxdo-dev to build)
	rm -f engine-metal/strata-app && cp app/target/release/strata-app engine-metal/strata-app
	mkdir -p $(HOME)/.local/share/applications
	printf '[Desktop Entry]\nType=Application\nName=Strata\nComment=Local AI models\nExec=$(CURDIR)/engine-metal/strata-app\nIcon=utilities-terminal\nTerminal=false\nCategories=Development;\n' > $(HOME)/.local/share/applications/strata.desktop
	@echo "built engine-metal/strata-app: make app-run, or Strata in the desktop's applications"
else
	rm -rf Strata.app && mkdir -p Strata.app/Contents/MacOS
	cp app/target/release/strata-app Strata.app/Contents/MacOS/Strata
	cp app/Info.plist Strata.app/Contents/Info.plist
	@# a locked keychain (an ssh session: errSecInternalComponent) signs ad-hoc instead, and says so
	codesign --force --sign $(SIGN_ID) Strata.app || { echo "signed ad-hoc: unlock the keychain (security unlock-keychain) to sign as $(SIGN_ID); macOS will ask for its permissions again"; codesign --force --sign - Strata.app; }
	@echo "built Strata.app: make app-run, or open Strata.app (or drag it to /Applications; it runs the server from $(CURDIR))"
endif

# strata-tui (tui/, Rust): Strata.app in a terminal, for Linux on an Apple Silicon Mac (and any terminal, SSH too)
tui-build: server
	$(CARGO) build --release --manifest-path tui/Cargo.toml
	rm -f engine-metal/strata-tui && cp tui/target/release/strata-tui engine-metal/

tui: tui-build
	engine-metal/strata-tui

# SIGTERM is the app's Quit: it stops its server the server's way (the engine frees its memory) and exits
APP_PAT = $(if $(filter Darwin,$(shell uname -s)),Strata.app/Contents/MacOS/Strata$$,engine-metal/strata-app$$)
app-stop:
	@if pkill -TERM -f '$(APP_PAT)'; then printf 'stopping Strata.app'; \
	  while pgrep -f '$(APP_PAT)' >/dev/null; do printf .; sleep 1; done; echo ' stopped'; else echo 'Strata.app is not running'; fi

# blue-green while the model is busy (app/src/deploy.rs): the new app's server takes the port, the old one finishes its
# requests; idle, the old app quits and the new one opens
# Linux: binaries already built for this checkout's commit (make linux-deliver writes it into engine-metal/.build) are
# used as they are - a 2012 laptop needs no hour of compiling for what another machine built; else they are built here
ifeq ($(shell uname -s),Darwin)
app-run: app
	Strata.app/Contents/MacOS/Strata --deploy
else
app-run:
	@if [ -x engine-metal/strata-app ] && [ -s engine-metal/.build ] && git diff --quiet "$$(cat engine-metal/.build)" HEAD -- app server metal CMakeLists.txt 2>/dev/null; then \
	  echo "engine-metal has the build of $$(cat engine-metal/.build): no app, server or engine change since, nothing to compile"; \
	else $(MAKE) build app && git rev-parse --short=8 HEAD > engine-metal/.build; fi
	engine-metal/strata-app --deploy
endif

# the server's tests: its unit and golden tests, then its answers against the Python server's recorded ones
# (server/golden/e2e.py --replay; the packs it needs are the ones on this Mac, a missing one is skipped)
PACKS ?= $(HOME)/dennis-akimov/Strata-data/packs
test-server:
	$(CARGO) test --release --manifest-path server/Cargo.toml
	$(CARGO) build --manifest-path server/Cargo.toml
	@for p in qwen3.6-35b-a3b gpt-oss-120b-heretic-v2 glm-5.3-flash-maya-s24; do \
	  if [ -d $(PACKS)/$$p ]; then $(PY) server/golden/e2e.py server/target/debug/strata-server $(PACKS)/$$p/tokenizer --replay mock-$$p || exit 1; fi; done
	@for p in qwen3.6-35b-a3b gpt-oss-120b-heretic-v2; do \
	  if [ -d $(PACKS)/$$p ]; then $(PY) server/golden/e2e.py server/target/debug/strata-server $(PACKS)/$$p/tokenizer --engine-strata --replay engine-$$p || exit 1; fi; done
	@if [ -d $(PACKS)/qwen3.6-35b-a3b ]; then $(PY) server/golden/e2e.py server/target/debug/strata-server $(PACKS)/qwen3.6-35b-a3b/tokenizer --mcp --replay mcp-qwen3.6-35b-a3b; fi

# tools/ has no __init__.py, so `unittest discover` will not start in it: the modules are named instead
TEST_MODULES := $(subst /,.,$(patsubst %.py,%,$(wildcard tools/test_setup_*.py))) \
                $(if $(filter Darwin,$(shell uname -s)),metal.test_setup_mac,)

test:
	$(PY) -m unittest $(TEST_MODULES)

test-engine:
	@test -n "$(TEST_GGUF)" || { echo "TEST_GGUF=<a small .gguf, e.g. LiquidAI LFM2-350M Q8_0>"; exit 1; }
	STRATA_METAL_EXE=$(or $(TEST_EXE),build-metal/metal/strata-metal) STRATA_METAL_GGUF=$(TEST_GGUF) $(PY) -m unittest metal.test_strata_metal -v

build-ab:
	$(CMAKE) -S . -B build-metal-a -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON -DSTRATA_METAL_KERNEL_PATCHES=OFF
	$(CMAKE) --build build-metal-a --target strata-metal -j
	$(CMAKE) -S . -B build-metal-b -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_METAL=ON -DSTRATA_METAL_KERNEL_PATCHES=ON
	$(CMAKE) --build build-metal-b --target strata-metal -j

ab:
	@test -n "$(AB_GGUF)" -a -n "$(AB_PROMPTS)" || { echo "AB_GGUF=<model shard 1> AB_PROMPTS=<json list of token-id lists>"; exit 1; }
	$(PY) metal/bench/ab.py --a $(AB_A) --b $(AB_B) --gguf $(AB_GGUF) --prompts $(AB_PROMPTS) --rounds $(AB_ROUNDS) --log strata-ab.log

clean:
	rm -rf build-metal build-metal-a build-metal-b
