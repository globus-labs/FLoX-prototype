.PHONY: typehint
typehint:
	mypy --ignore-missing-imports flox

.PHONY: test
test:
	pytest flox/tests/ --disable-warnings

.PHONY: test_unit
test_unit:
	pytest flox/tests/unit --disable-warnings

.PHONY: lint
lint:
	pylint flox/

.PHONY: checklist
checklist: typehint lint

.PHONY: black
black:
	black -l 88 flox

.PHONY: clean
clean:
	find . -type f -name "*.pyc" | xargs rm -fr
	find . -type d -name __pycache__ | xargs rm -fr