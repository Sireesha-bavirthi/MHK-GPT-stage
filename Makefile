.PHONY: install run clean test help

help:
	@echo "Available commands:"
	@echo "  install   Install dependencies"
	@echo "  run       Run the backend server"
	@echo "  clean     Clean up temporary files"
	@echo "  test      Run tests"

install:
	pip install -r backend/requirements.txt

run:
	cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	cd backend && pytest
