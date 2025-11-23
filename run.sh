export PYTHONUNBUFFERED=1
cd src
uvicorn main:app --port 8000 --reload