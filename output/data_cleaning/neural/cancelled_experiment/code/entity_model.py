"""Contextual target-entity classification by a fine-tuned Russian BERT.

No keyword rules or default-person fallback. Low confidence is an abstention.
"""
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
try:
    from .train_entity_bert import inputs, CLASSES
except ImportError:
    from train_entity_bert import inputs, CLASSES

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'local/tv-entity-rubert'
MODEL_DIR = ROOT/'analysis/models/tv-entity-bert'
SCHEMA_VERSION = 'supervised-target-entity-3way-v1'


@lru_cache(maxsize=1)
def model_manifest():
    return json.loads((artifact_directory()/'training_manifest.json').read_text(encoding='utf-8'))


def artifact_directory():
    config=ROOT/'data/entity_model_config.json'
    path=ROOT/json.loads(config.read_text(encoding='utf-8'))['artifact_directory'] if config.exists() else MODEL_DIR
    if not path.resolve().is_relative_to(ROOT):raise ValueError('Model artifact must be inside project')
    return path


def model_revision():
    return model_manifest()['model_sha256']


def model_text(m):
    return inputs(m)[0]


def input_key(m, model_sha=None):
    payload = [MODEL_ID, model_sha or model_revision(), SCHEMA_VERSION, inputs(m)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def decision_label(prediction, min_score, min_margin):
    mapping = {'historical_person': 'person_likely', 'non_target_reference': 'not_target_person',
               'commemoration': 'commemoration'}
    category = prediction['category']
    if category not in mapping:
        raise ValueError('Unknown neural category '+str(category))
    if not math.isfinite(prediction['score']) or not math.isfinite(prediction['margin']):
        raise ValueError('Non-finite model score')
    if prediction['score'] < min_score or prediction['margin'] < min_margin:
        return 'uncertain_reference', 'bert_low_confidence'
    return mapping[category], 'bert_'+category


class EntityModel:
    def __init__(self, threads=4):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        directory=artifact_directory()
        if hashlib.sha256((directory/'model.safetensors').read_bytes()).hexdigest() != model_revision():
            raise ValueError('Fine-tuned BERT checksum differs from training manifest')
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True).eval()
        if [self.model.config.id2label[i] for i in range(3)] != CLASSES:
            raise ValueError('Unexpected trained model classes')

    def predict(self, mentions, batch_size=32):
        result = []
        with self.torch.inference_mode():
            for offset in range(0, len(mentions), batch_size):
                batch = mentions[offset:offset+batch_size]
                pairs = [inputs(m) for m in batch]
                encoded = self.tokenizer([v[0] for v in pairs], [v[1] for v in pairs], padding=True,
                    truncation='only_first', max_length=256, return_tensors='pt')
                probabilities = self.model(**encoded).logits.softmax(-1).tolist()
                for m, values in zip(batch, probabilities):
                    if not all(math.isfinite(v) for v in values):
                        raise RuntimeError('Non-finite BERT output')
                    indices = sorted(range(3), key=lambda i: values[i], reverse=True)
                    best, second = indices[:2]
                    result.append(dict(category=CLASSES[best], score=values[best], margin=values[best]-values[second],
                        scores=dict(zip(CLASSES, values)), input_key=input_key(m)))
        return result
