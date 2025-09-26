FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt ./

#xvfb
RUN apt-get update && apt-get install -y --no-install-recommends xvfb && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

RUN patchright install chrome --with-deps

COPY . .

CMD ["xvfb-run", "-a", "python", "-m", "puppet.puppet"]