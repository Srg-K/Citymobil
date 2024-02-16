VENV_NAME = venv

# Полный путь - чтобы по команде запуска понимать, к какому проекту относится запущенный процесс
VENV_PATH="$$(realpath $(VENV_NAME))"
PYTHON = $(VENV_PATH)/bin/python
PIP = $(VENV_PATH)/bin/pip

INSTALL_FILES = $(VENV_NAME)/pyvenv.cfg .env

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
run : $(INSTALL_FILES)
	import OMP_NUM_THREADS=1 # нагружаем только один процессор

	@EXIT_CODE=0
	
	$(PYTHON) main.py
	@EXIT_CODE=$$(($$EXIT_CODE || $$?))

	## Uncomment following lines to run 2nd script
	# $(PYTHON) main2.py
	# @EXIT_CODE=$$(($$EXIT_CODE || $$?))
	
	# Exit code will be non-zero if one of scripts returns non-zero code
	return $$EXIT_CODE

$(VENV_NAME)/pyvenv.cfg :
	virtualenv -p python3.8 $(VENV_NAME)
	$(PIP) install -r requirements.txt

.env:
	touch .env
