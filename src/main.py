import torch

torch.set_grad_enabled(False)

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
ckpt_path = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReRankRequest(BaseModel):
    query: str
    passages: List[str]


class ReRankResponse(BaseModel):
    scores: List[float]


@app.post("/rerank", response_model=ReRankResponse)
async def rerank(request: ReRankRequest):
    features = tokenizer(
        [request.query] * len(request.passages),
        request.passages,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    all_scores = []
    for i in range(0, len(request.passages), BATCH_SIZE):
        batch_features = {k: v[i : i + BATCH_SIZE] for k, v in features.items()}
        batch_scores = model(**batch_features).logits
        all_scores.extend(batch_scores.detach().flatten().tolist())

    return ReRankResponse(scores=all_scores)
