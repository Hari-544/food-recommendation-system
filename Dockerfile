FROM python:3.10

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn

COPY . .

# Generate the similarity.npy matrix at build time directly inside Hugging Face
# This intelligently bypasses GitHub's 100MB file limit!
RUN python model/train_model.py

# Expose port 7860, required by Hugging Face Spaces!
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
