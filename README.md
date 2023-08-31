# kaggle-Quora_Question_Pairs

## Overview
The task is to identify whether two questions are semantically identical.

1. **Feature Engineering with LightGBM Approach**:
    - This approach begins by importing required libraries such as numpy, pandas, lightgbm, NLTK, and sklearn.
    - Data is then loaded, concatenated, and preprocessed to extract features that could determine the similarity between questions.
    - Multiple feature engineering functions are defined and applied.
    - The features are used to train a LightGBM model.

2. **BERT-based Approach for Semantic Similarity**:
    - This approach leverages BERT, a well-known Transformer model for NLP tasks.
    - The training data is tokenized using BERT’s tokenizer.
    - A custom PyTorch Dataset is defined for this task.
    - BERT's Next Sentence Prediction model is utilized for training.
    - The model is then trained using the AdamW optimizer and a learning rate scheduler.

## Prerequisites

### Libraries:

- **numpy**
- **pandas**
- **lightgbm**: Gradient Boosting Framework.
- **nltk**: Natural Language Toolkit, used for stopwords here.
- **sklearn**: For splitting datasets.
- **torch**: PyTorch library, a deep learning framework.
- **transformers**: A library by HuggingFace containing pretrained transformers.

## Note:

- The LightGBM approach leverages a combination of word-based features, length-based features, and some additional features to classify pairs of questions.
  
- The BERT-based approach uses BERT's next sentence prediction to determine if two sentences are similar. However, note that the BERT model might require substantial computational resources, including a GPU for faster training.
