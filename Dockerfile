FROM python:3.12.7-bookworm

WORKDIR /app

COPY . /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --timeout=1000 --index-url https://pypi.python.org/simple/

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app.py"]