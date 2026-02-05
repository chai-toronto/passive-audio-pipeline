FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

RUN pip install uv

WORKDIR /opt
RUN git clone https://github.com/google/visqol.git
WORKDIR /opt/visqol

RUN uv pip install --system "numpy<2.0" absl-py protobuf==3.20.0

RUN bazel build //python/...

WORKDIR /usr/local/lib/python3.10/site-packages

RUN mkdir -p visqol/pb2
RUN touch visqol/pb2/__init__.py

RUN find -L /opt/visqol -name "*_pb2.py" -exec cp -v {} visqol/pb2/ \;

RUN find -L /opt/visqol -name "*visqol*.so" -exec cp -v {} ./visqol_lib_py.so \;

RUN ls -l visqol_lib_py.so && chmod 755 visqol_lib_py.so

RUN echo "import visqol_lib_py" > visqol/__init__.py

WORKDIR /
RUN python -c "from visqol import visqol_lib_py; import visqol.pb2.visqol_config_pb2; print('SUCCESS: VisQOL is installed!')"

WORKDIR /app
COPY requirements.txt .

RUN uv pip install --system torch torchaudio
RUN uv pip install --system -r requirements.txt

COPY . .

CMD ["python", "main.py"]
