import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Union
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import os
import warnings
from huggingface_hub import hf_hub_download
from app.craft_text_detector import Craft

warnings.filterwarnings('ignore')

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

class EssayScorer(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.pool = MeanPooling()
        self.dropout = nn.Dropout(0.3)
        self.classifiers = nn.ModuleList([
            nn.Linear(self.backbone.config.hidden_size, num_labels) for _ in range(4)
        ])

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.pool(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = [classifier(pooled_output) for classifier in self.classifiers]
        return logits

class EssayScoringService:
    def __init__(
        self,
        model_path: str,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 17,  # классы: от 1.0 до 9.0 с шагом 0.5 → 17 классов
        device: Union[str, torch.device] = None
    ):
        
        if not os.path.exists(model_path):
            model_path = hf_hub_download(
                repo_id="ArtemBelogur/essay_scoring_model",
                filename="best_model1.pt",  
                local_dir="models"  
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = EssayScorer(model_name, num_labels)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {i: round(1.0 + 0.5 * i, 1) for i in range(num_labels)}

    def predict(
        self,
        topics: List[str],
        essays: List[str],
        batch_size: int = 32
    ) -> List[List[float]]:
        assert len(topics) == len(essays), "Длина списков topics и essays должна совпадать"

        results = []

        for i in tqdm(range(0, len(essays), batch_size)):
            batch_essays = essays[i:i + batch_size]
            batch_topics = topics[i:i + batch_size]
            batch_inputs = [
                f"{topic} [SEP] {essay}" for topic, essay in zip(batch_topics, batch_essays)
            ]

            encodings = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(encodings["input_ids"], encodings["attention_mask"])

            batch_preds = [
                torch.argmax(logit, dim=1).cpu().tolist()
                for logit in logits  # список из 4 тензоров
            ]

            # транспонируем список: [4][batch] → [batch][4]
            batch_preds = list(map(list, zip(*batch_preds)))
            results.extend([[self.id2label[i] for i in row] for row in batch_preds])

        return results
    

def get_scorer_single_essay(model_path="app/models/best_model1.pt"):
    return EssayScoringService(model_path)


def ai_detector(essay):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = "ArtemBelogur/ai_detector_model"
    MAX_LEN = 512

    print('device = ', device)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_PATH, max_position_embeddings=MAX_LEN
    ).to(device)

    y_pred = []
    with torch.no_grad():
        inputs = tokenizer(
            essay,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits.cpu().numpy()
        y_pred.extend(
            (np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True))[:, 1]
        )

    return y_pred[0]

def sort_boxes(boxes):
    """Сортировка bounding boxes слева-направо и сверху-вниз"""
    # Вычисляем средние y-координаты для каждого бокса
    mean_y = [np.mean(box[:, 1]) for box in boxes]
    
    # Сортируем по строкам (основываясь на y-координате)
    sorted_boxes = sorted(zip(mean_y, boxes), key=lambda x: x[0])
    
    # Группируем по строкам (с учетом некоторого допуска по высоте)
    rows = []
    current_row = []
    y_threshold = 20  # Допуск для объединения в одну строку
    
    for y, box in sorted_boxes:
        if not current_row or abs(y - current_row[0][0]) < y_threshold:
            current_row.append((y, box))
        else:
            rows.append(current_row)
            current_row = [(y, box)]
    if current_row:
        rows.append(current_row)
    
    # Сортируем каждую строку слева направо по x-координате
    final_boxes = []
    for row in rows:
        # Берем среднюю x-координату для каждого бокса в строке
        sorted_row = sorted(row, key=lambda x: np.mean(x[1][:, 0]))
        final_boxes.extend([box for y, box in sorted_row])
    
    return final_boxes

def read(image_path):
    # Проверяем существует ли файл
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Читаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    model_path = 'microsoft/trocr-small-handwritten'
    
    craft = Craft(output_dir=None, crop_type="box", cuda=True)
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    # Детектим текст
    result = craft.detect_text(image_path)
    boxes = result["boxes"]
    
    # Сортируем боксы в правильном порядке
    sorted_boxes = sort_boxes(boxes)
    
    # Конвертируем изображение для PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    texts = []
    for box in sorted_boxes:
        try:
            # Получаем координаты для обрезки
            x0, y0 = box[0][0], box[0][1]
            x1, y1 = box[2][0], box[2][1]
            
            # Обрезаем изображение
            crop = pil_image.crop((x0, y0, x1, y1))
            
            # Распознаем текст
            pixel_values = processor(crop, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            texts.append(text)
        except Exception as e:
            print(f"Error processing box {box}: {str(e)}")
            continue
    
    return texts