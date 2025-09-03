# Fake News Detection with DistilBERT, MobileBERT, TinyBERT, and BERT-base

This project benchmarks multiple transformer-based models (DistilBERT, MobileBERT, TinyBERT, and BERT-base) on the **Kaggle Fake News dataset** for binary classification of news articles as *fake* or *real*.  
The goal is to evaluate trade-offs between **accuracy, F1 score, inference speed, and training efficiency** across models of different sizes.

---

## Dataset

We use the [Fake News Dataset](https://www.kaggle.com/c/fake-news/data) from Kaggle:

- **Size**: 20,800 rows × 5 columns  
- **Columns**:  
  - `id`: unique identifier  
  - `title`: title of the news article  
  - `author`: author of the article  
  - `text`: full news article text  
  - `label`: target (0 = real, 1 = fake)  

The dataset is split into:  
- 70% training  
- 20% testing  
- 10% validation  

---

## Installation

Clone the repo and install the dependencies:

```bash
pip install -U transformers accelerate datasets bertviz umap-learn seaborn openpyxl
```
## Data Preprocessing
Tokenization with model-specific tokenizers (BERT, DistilBERT, MobileBERT, TinyBERT).

### Dataset Visualization
<img width="554" height="435" alt="image" src="https://github.com/user-attachments/assets/9a534dd3-0293-4984-8c90-0a6a632538ce" />
<img width="1227" height="451" alt="image" src="https://github.com/user-attachments/assets/7e4d8987-2fd9-47b7-bebf-95b5c92fa22d" />



Average 1.5 tokens per word estimated.

Train/validation/test splits are stratified by label.

## Training
Fine-tuning is performed using Hugging Face Trainer.

Fine-Tuned models in this project:

bert-base-uncased

distilbert-base-uncased

google/mobilebert-uncased

huawei-noah/TinyBERT_General_4L_312D

## Results Visualizations
### Accuracy vs F1
A bar chart comparing Accuracy and F1 score across models.

### Runtime Comparison
Bar chart showing inference runtime per model.

### Training Time Comparison
Bar chart showing training time (s) for each model.

<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/9f7fd835-0a02-40f9-bbae-1fbf209dd826" />


### Accuracy vs F1 (Zoomed Line Chart)
Line chart highlighting small differences between BERT, DistilBERT, and MobileBERT.
<img width="708" height="451" alt="image" src="https://github.com/user-attachments/assets/56c56fbd-43f1-4bbf-9378-de0dd08ee244" />


## Key Takeaways
BERT-base achieves the highest accuracy and F1 score but requires the longest training time.

DistilBERT provides nearly the same performance while reducing training and inference time significantly.

MobileBERT also performs competitively but is slower to train than DistilBERT.

TinyBERT is the fastest model with minimal resource usage but sacrifices accuracy.


## Future Work
Experiment with larger transformer models (RoBERTa, XLNet).

Apply data augmentation for handling class imbalance.

Evaluate deployment efficiency on edge devices.

References
Fake News Dataset - Kaggle

Hugging Face Transformers

yaml
Copy code
