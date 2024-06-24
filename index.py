from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from container import process_models, compare_text
import pandas as pd
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/review")
def get_review( text_arr: List[str] ):
    try:
        if not text_arr: 
            return {"status": 422, "message": "Invalid Input" }
            
        output = [process_models(text) for text in text_arr]
        if not output:
            return {"status": 500,"message": "Data Couldn't saved", "data": {}}
        return {"status": 200,"message": "Data saved Successfully", "data": output}
    except Exception as e:
        return {"status": 500,"message": str(e)}

@app.post("/api/v1/compare_text")
def compare(texts: List[str], text:str):
    try:
        if not texts: 
            return {"status": 422, "message": "Invalid Input" }
        if not text: 
            return {"status": 422, "message": "Invalid Input" }
        output = compare_text(texts, text)
        if not output:
            return {"status": 500,"message": "Data Couldn't saved", "data": {}}
        return {"status": 200,"message": "Data saved Successfully", "data": output}
    except Exception as e:
        return {"status": 500,"message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)