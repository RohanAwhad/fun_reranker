import torch

torch.set_grad_enabled(False)

import os
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
ckpt_path = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

class ReRankRequest(BaseModel):
    query: str
    passages: List[str]


class ReRankResponse(BaseModel):
    scores: List[float]


def lambda_handler(event, context):
    request = ReRankRequest(**event)
    print(f"Query: {request.query}")
    print(f"Passages: {request.passages}")
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

    ret = ReRankResponse(scores=all_scores)
    return {
        'statusCode': 200,
        'body': ret.model_dump_json()
    }
