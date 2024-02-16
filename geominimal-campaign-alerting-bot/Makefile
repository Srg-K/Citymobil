VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
INSTALL_FILES = $(VENV_NAME) .env

.DEFAULT : help
.PHONY : help
help:
	@echo "Following make targets are available:"
	@echo "  install - install requirements"
	@echo "  clean - delete all artifacts"
	@echo "  reinstall - clean & install"
	@echo "  run - run project"

.PHONY : install
install: $(INSTALL_FILES)

.PHONY : reinstall
reinstall: clean $(INSTALL_FILES)

.PHONY : clean
clean :
	rm -rf $(VENV_NAME)

.PHONY : run
.ONESHELL: run
run :
	@EXIT_CODE=0

	$(PYTHON) main.py
	@EXIT_CODE=$$(($$EXIT_CODE || $$?))

	## Uncomment following lines to run 2nd script
	# $(PYTHON) main2.py
	# @EXIT_CODE=$$(($$EXIT_CODE || $$?))

	# Exit code will be non-zero if one of scripts returns non-zero code
	return $$EXIT_CODE

.PHONY : run_weekly
.ONESHELL: run_weekly
run_weekly :
	@EXIT_CODE=0

	$(PYTHON) weekly_main.py
	@EXIT_CODE=$$(($$EXIT_CODE || $$?))

	## Uncomment following lines to run 2nd script
	# $(PYTHON) main2.py
	# @EXIT_CODE=$$(($$EXIT_CODE || $$?))

	# Exit code will be non-zero if one of scripts returns non-zero code
	return $$EXIT_CODE

$(VENV_NAME) :
	virtualenv -p python3 $(VENV_NAME)
	$(PIP) install -r requirements.txt

.env:
	touch .env
