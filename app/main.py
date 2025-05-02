from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import shutil
import os
import sys

from app.services import read, ai_detector, get_scorer_single_essay  # Импорт OCR, AI-детектора и скорера

app = FastAPI(title="Essay Scoring API")

@app.post("/predict/")
async def predict(
    topic: str = Form(...),
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    if text:
        # Используем текст напрямую
        full_text = text
    elif file:
        # Распознаем текст с изображения
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            image_path = tmp.name

        try:
            paragraphs = read(image_path)
            full_text = " ".join(paragraphs)
        finally:
            os.remove(image_path)
    else:
        return JSONResponse(status_code=400, content={"error": "Either 'text' or 'file' must be provided."})

    # Проверяем на генерацию ИИ
    ai_score = ai_detector(full_text)

    # # Получаем оценки по 4 критериям
    scorer = get_scorer_single_essay()
    scores = scorer.predict([topic], [full_text])[0]

    return JSONResponse({
        "recognized_text": full_text,
        "ai_probability": str(round(ai_score, 3)),
        "scores": {
            "Task Response": scores[0],
            "Coherence and Cohesion": scores[1],
            "Lexical Resource": scores[2],
            "Grammatical Range and Accuracy": scores[3]
        }
    })
