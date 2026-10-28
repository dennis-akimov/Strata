# Strata's chat server: the Rust server (this folder) natively, or in a container with the engine on the host.
#
#   make up        the host relay + the chat container (http://127.0.0.1:$(CHAT_PORT)/)
#   make down      both stopped
#   make chat MSG="hello"   one message to the container's chat API
#   make test      the server's tests
#
# Settings: .env (cp .env.example .env).
-include .env
export

ENGINE     ?= $(STRATA_DIR)/engine-metal/strata-metal
RELAY      := target/release/strata-relay
URL        := http://127.0.0.1:$(CHAT_PORT)
KEY        = $(shell python3 -c "import json;print(json.load(open('$(STRATA_CONFIG)')).get('api_key',''))" 2>/dev/null)
AUTH       = $(if $(KEY),-H "Authorization: Bearer $(KEY)",)
MSG        ?= Say hello in one short sentence.

.PHONY: build relay relay-stop up down logs chat status test env

env:
	@test -f .env || { echo "no .env: cp .env.example .env (and set the paths and a token)"; exit 1; }

build:
	cargo build --release

# the host side of strata-relay: starts the Metal engine for the container (one at a time)
relay: env build
	@if [ -f relay.pid ] && kill -0 $$(cat relay.pid) 2> /dev/null; then echo "relay already runs (pid $$(cat relay.pid))"; \
	else nohup $(RELAY) listen --engine $(ENGINE) --addr 127.0.0.1:$(RELAY_PORT) > relay.log 2>&1 & echo $$! > relay.pid; \
	sleep 1; kill -0 $$(cat relay.pid) 2> /dev/null && echo "relay on 127.0.0.1:$(RELAY_PORT) (log: server/relay.log)" || { cat relay.log; exit 1; }; fi

relay-stop:
	@if [ -f relay.pid ]; then kill $$(cat relay.pid) 2> /dev/null; rm -f relay.pid; echo "relay stopped"; fi

up: relay
	docker compose up -d --build
	@i=0; while [ $$i -lt 600 ]; do \
	  code=$$(curl -s -o /dev/null -w '%{http_code}' -m 2 $(URL)/health); \
	  if [ "$$code" = "200" ]; then echo "ready: $(URL)/ (chat), $(URL)/v1 (API)"; exit 0; fi; \
	  sleep 2; i=$$((i + 2)); \
	done; echo "not ready: make logs"; exit 1

down:
	docker compose down
	@$(MAKE) --no-print-directory relay-stop

logs:
	docker compose logs -f --tail 100

status:
	@docker compose ps; curl -s -m 3 $(URL)/health; echo

chat:
	@curl -s $(URL)/v1/chat/completions $(AUTH) -H 'Content-Type: application/json' \
	  -d "$$(python3 -c 'import json,os; print(json.dumps({"messages": [{"role": "user", "content": os.environ["MSG"]}], "reasoning_effort": "low"}))')" | \
	  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"] if "choices" in d else d)'

test:
	cargo test --release
