# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import pipeline

# app = FastAPI()

# summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

# class MedicalTextRequest(BaseModel):
#     text: str

# class MedicalTextResponse(BaseModel):
#     summary: str

# @app.post("/summarize", response_model=MedicalTextResponse)
# async def summarize_medical_text(request: MedicalTextRequest):
#     try:
#         summary = summarizer(request.text, max_length=2000, min_length=1500, do_sample=False)[0]['summary_text']
#         return {"summary": summary}
#     except Exception as e:
#         return {"error": str(e)}

from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# Load the FalconSai model
medical_summarization = pipeline("summarization", model="t5-base", tokenizer="t5-base")

@app.post("/summarize/")
async def summarize_text(text: str):
    # Use the FalconSai model to summarize the medical text
    summary = medical_summarization(text, max_length=2000, min_length=1000, do_sample=False)[0]['summary_text']
    return {"summary": summary}

