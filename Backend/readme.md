How to launch on localhost:
poetry shell
fastapi dev backend.py

Export poetry libraries to a requirements file
poetry export -f requirements.txt --output requirements.txt --without-hashes
