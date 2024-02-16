VENV_NAME = venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip
INSTALL_FILES = $(VENV_NAME)/src .env

.DEFAULT : help
.PHONY : help
help:
	@echo "Following make targets are available:"
	@echo "  install - install requirements"
	@echo "  clean - delete all artifacts"
	@echo "  reinstall - clean & install"
	@echo "  run - run AB_tests"

.PHONY : install
install: $(INSTALL_FILES)

.PHONY : reinstall
reinstall: clean $(INSTALL_FILES)

.PHONY : clean
clean :
	rm -rf $(VENV_NAME)

.PHONY : run
run : $(INSTALL_FILES)
	$(PYTHON) main.py || true

$(VENV_NAME)/src : $(VENV_NAME)
	$(PIP) install -r requirements.txt

$(VENV_NAME) :
	virtualenv -p python3.8 $(VENV_NAME)

.env:
	touch .env
