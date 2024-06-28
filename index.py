from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from container import process_models, compare_text
import logging
import json
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewRequest(BaseModel):
    review: List[str]

class CompareRequest(BaseModel):
    texts: List[str]
    text: str

@app.post("/api/v1/review")
def get_review(request: ReviewRequest):
    try:
        review_text = request.review
        if not review_text:
            return JSONResponse({"status": 422, "message": "Invalid Input"}, status_code=422)
        
        output = [process_models(text) for text in review_text]
        print(output)
        if not output:
            print("Error from here")
            return JSONResponse({"status": 500, "message": "Data couldn't be saved", "data": {}}, status_code=500)
        return JSONResponse({"status": 200, "message": "Data saved successfully", "data": output}, status_code=200)
    except Exception as e:
        print("Error => ", e)
        return JSONResponse({"status": 500, "message": str(e)}, status_code=500)

@app.post("/api/v1/compare_text")
def compare(request: CompareRequest):
    try:
        texts = request.texts
        text = request.text
        if not texts or not text:
            return JSONResponse({"status": 422, "message": "Invalid Input"}, status_code=422)

        output = compare_text(texts, text)
        if not output:
            return JSONResponse({"status": 500, "message": "No matching text found", "data": {}})

        return JSONResponse({"status": 200, "message": "Matching text found", "data": output})
    except Exception as e:
        return JSONResponse({"status": 500, "message": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)