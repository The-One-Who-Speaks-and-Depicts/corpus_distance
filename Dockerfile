FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3-pip python3.10-venv
COPY . .
RUN pip install -r requirements.txt
RUN pip install build
RUN python3 -m build
RUN pip install dist/*.whl
