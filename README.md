# Twitter Sentiment Classification: BERT vs DistilBERT

Fine-tuned BERT and DistilBERT on a 3-class Twitter sentiment dataset (Positive / Neutral / Negative), then compared accuracy and inference speed.

## Dataset
[Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) from Kaggle. The "Irrelevant" class was dropped to keep it 3-class. 80/20 stratified train/test split.

## Approach
- Tokenized tweets with `max_length=128`
- Fine-tuned `bert-base-uncased` and `distilbert-base-uncased` with the same settings: 3 epochs, lr=2e-5, batch size 16
- Tuned the decision threshold for the minority class to improve its F1
- Wrapped the best model in a `predict()` function that returns a label with a confidence score

## Results

| Model | Accuracy | Inference time (test set) |
|---|---|---|
| BERT | 0.916 | 62.2 s |
| DistilBERT | 0.899 | 30.8 s |

BERT is about 1.7 points more accurate. DistilBERT is about 2x faster at inference.

## Usage
```python
predict("I love this game!")
# {'label': 'Positive', 'confidence': 0.98}
```
The `predict()` output above is an example of the format, not a real result.

## Stack
Python, Hugging Face Transformers and Datasets, PyTorch, scikit-learn, trained on Kaggle GPU.

## Files
- `notebook.ipynb`: full pipeline, from data prep to evaluation and prediction
