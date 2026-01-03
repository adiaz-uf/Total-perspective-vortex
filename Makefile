.PHONY: all create-venv jupyter install-deps	

GREEN := $(shell tput setaf 2)
YELLOW := $(shell tput setaf 3)
BLUE := $(shell tput setaf 4)
RESET := $(shell tput sgr0)

all: create-venv install-deps

create-venv:
	python3 -m venv venv
	@echo "${GREEN}Virtual environment created!${RESET}"

jupyter:
	@venv/bin/jupyter lab

install-deps:
	venv/bin/pip install -r requirements.txt
	@echo "${GREEN}Dependencies installed!${RESET}"