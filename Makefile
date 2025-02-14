SRC = .

BLACK = black
ISORT = isort


.PHONY: format
format: 
	$(ISORT) $(SRC) ${args}
	$(BLACK) $(SRC) ${args}


.PHONY: black
black:
	docker-compose run --rm formatter black --exclude migrations --settings-path setup.cfg **/*.py


.PHONY: isort
isort: 
	docker-compose run --rm formatter isort --settings-path setup.cfg **/*.py


.PHONY: start
start:
	docker-compose up


.PHONY: down
down:
	docker-compose down -v


.PHONY: fresh
fresh:
	docker-compose build
	make start
