from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.llm_handler import get_answer
from io import BytesIO
import pandas as pd
import zipfile
import pdfplumber  
import pytesseract  
from PIL import Image  
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Assignment Answer API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "QnA Genie API is running"}

@app.post("/api/")
async def answer_question(question: str = Form(...), file: UploadFile = File(None)):
    logger.info(f"Received question: {question}")
    if file:
        logger.info(f"Received file: {file.filename}")

    file_content = None  # Default to None

    if file:
        # Read file in-memory
        contents = await file.read()
        file_ext = file.filename.split('.')[-1].lower()

        if file_ext == "zip":
            with zipfile.ZipFile(BytesIO(contents), "r") as zip_ref:
                for name in zip_ref.namelist():
                    if name.endswith(".csv"):
                        with zip_ref.open(name) as csv_file:
                            df = pd.read_csv(csv_file)
                            file_content = df.to_string()
                        break  # Stop after processing first CSV

        elif file_ext == "csv":
            df = pd.read_csv(BytesIO(contents))
            file_content = df.to_string()

        elif file_ext == "pdf":
            with pdfplumber.open(BytesIO(contents)) as pdf:
                file_content = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        elif file_ext == "txt":
            file_content = contents.decode("utf-8")

        elif file_ext in ["jpeg", "jpg", "png"]:
            image = Image.open(BytesIO(contents))
            file_content = pytesseract.image_to_string(image)

        elif file_ext == "json":
            file_content = json.loads(contents.decode("utf-8"))

    # Get answer using LLM
    answer = get_answer(question, file_content)

    return {"answer": clean_json_output(answer)}

def clean_json_output(response_text):
    # Remove markdown-style JSON formatting if present
    if response_text.startswith("```json"):
        response_text = response_text.strip("```json").strip("```")
    
        # Ensure the response is valid JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return response_text  # Return raw text if not valid JSON
    
    else:
        # If not JSON formatted, return the text as is
        return response_text
