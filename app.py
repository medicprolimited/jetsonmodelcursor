from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import os
from typing import List, Optional

app = FastAPI(title="Jetson Sentence Transformers API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODEL_PATHS = {
    "mpnet": "/home/mx/jetson-containers/data/models/torch/sentence_transformers/sentence-transformers_all-mpnet-base-v2",
    "minilm": "/home/mx/jetson-containers/data/models/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2"
}

# Load models
models = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        models[name] = SentenceTransformer(path)
    else:
        print(f"Warning: Model path {path} does not exist")

class EmbeddingRequest(BaseModel):
    text: str
    model: str = "mpnet"  # default to mpnet

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    model: str = "mpnet"

@app.get("/")
async def root():
    return {"message": "Jetson Sentence Transformers API is running"}

@app.post("/embed")
async def get_embedding(request: EmbeddingRequest):
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
    
    try:
        embedding = models[request.model].encode(request.text)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_batch")
async def get_batch_embeddings(request: BatchEmbeddingRequest):
    if request.model not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} not available")
    
    try:
        embeddings = models[request.model].encode(request.texts)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 