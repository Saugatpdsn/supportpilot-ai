FROM python:3.12-slim
WORKDIR /app
RUN groupadd -r app && useradd -r -m -g app app
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY app.py .
RUN chown -R app:app /app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]