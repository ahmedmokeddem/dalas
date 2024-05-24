
FROM python:3.10-slim

WORKDIR /app

RUN  pip install --upgrade pip
RUN git clone https://github.com/ahmedmokeddem/dalas.git .
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install "prophet==1.1.2"
RUN pip3 install plotly
RUN pip install numpy

COPY dashboard/ ./


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "./dashboard/🌎_Home.py", "--server.port=8501", "--server.address=0.0.0.0"]