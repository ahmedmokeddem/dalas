
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ahmedmokeddem/dalas.git .
RUN  pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


RUN pip3 install -r dashboard/requirements.txt
RUN python -m pip install "prophet==1.1.2"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dahboard/home.py", "--server.port=8501", "--server.address=0.0.0.0"]