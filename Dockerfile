FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN patchright install
RUN patchright install chrome --with-deps

COPY . .

CMD ["python", "-m", "puppet.puppet"]