from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from container import process_models, compare_text
import pandas as pd
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
async def get_review(request):
    try:
        print(request)
        body = request.model_dump_json()
        # return JSONResponse({"response": body})
        if isinstance(body, bytes):
            body = json.loads(body.decode('utf-8'))
        else:
            body = json.loads(body)
        review_text = body.get("review")
        if not review_text: 
            return JSONResponse({"status": 422, "message": "Invalid Input" }, status_code=422)
            
        output = [process_models(text) for text in review_text]
        if not output:
            return JSONResponse({"status": 500,"message": "Data Couldn't saved", "data": {}}, status_code=500)
        return JSONResponse({"status": 200,"message": "Data saved Successfully", "data": output}, status_code=200)
    except Exception as e:
        return JSONResponse({"status": 500,"message": str(e)}, status_code=500)

@app.post("/api/v1/compare_text")
def compare(request:CompareRequest):
    try:
        print(request)
        if not request.texts and not request.text:
            body = request.model_dump_json()
            if isinstance(body, bytes):
                body = json.loads(body.decode('utf-8'))
            else:
                body = json.loads(body)
            texts = body.get("texts")
            text = body.get("text")
        else:
            texts = request.texts
            text = request.text
        if not texts or not text: 
            return JSONResponse({"status": 422, "message": "Invalid Input"})

        output = compare_text(texts, text)
        if not output:
            return JSONResponse({"status": 500, "message": "No matching text found", "data": {}})

        return JSONResponse({"status": 200, "message": "Matching text found", "data": output})
    except Exception as e:
        return JSONResponse({"status": 500, "message": str(e)})

# get_review(
#     ["All around lovely brunch spot! Cute and quaint decor, very friendly and accommodating host and wait staff, delicious coffee, BYOB and endless food options. It was very difficult to choose one thing, but the combinations are unique and flavorful- far from routine. The meals are reasonably portioned and packed with flavor. I appreciate the time out into each dish as opposed to hugeee portions of hastily made eggs or pancakes. There are so many elements to the dishes. I had the gypsy eggs! Will have took back to try other options! Expect to wait to be seated and don't go with a very large group as seating will be difficult- in my mind it's worth the wait and the wait shows that many others feel the same! Have heard from multiple sources that this was a spot not to miss and I 100% concur. Thank you for great food, friendly atmosphere and BYOB"]
# )
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)