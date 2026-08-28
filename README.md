# Sentiment Classification with DistilBERT — Fine-Tuning & Overfitting Analysis

Fine-tuned DistilBERT on the IMDB movie review dataset for binary sentiment classification (positive/negative), then ran a controlled experiment comparing 5 vs. 3 training epochs to test whether the model was overfitting — and found it was.

## Results

| Epochs | Train Loss | Val Loss | Accuracy |
|---|---|---|---|
| 5 | 0.257 | 0.634 | 87.67% |
| **3** | **0.265** | **0.417** | **88.87%** |

Training for 3 epochs instead of 5 both **reduced the train/validation loss gap** and **improved accuracy**. This is a clear overfitting signature: by epoch 5, the model's training loss had barely moved from where it was at epoch 3, while validation loss kept climbing — meaning the extra training was fitting noise specific to the training set rather than learning anything that generalized better.

## Why this happened

- **Small training set relative to model capacity.** Only 3,000 training examples were used (a subset of IMDB's full 25,000, to keep training time reasonable), while DistilBERT has millions of trainable parameters. A model this capable can start memorizing a dataset this size within just a few epochs.
- **Loss revealed what accuracy alone would have hidden.** Both runs scored respectably on accuracy (87–89%), but the loss values told a more complete story: the 5-epoch model's predictions on unseen data were less confident/more uncertain than its training predictions, even when it landed on the correct final answer.

## Per-class performance

| Epochs | Class | Precision | Recall | F1 |
|---|---|---|---|---|
| 5 | negative | 0.89 | 0.87 | 0.88 |
| 5 | positive | 0.87 | 0.89 | 0.88 |
| **3** | **negative** | **0.91** | **0.87** | **0.89** |
| **3** | **positive** | **0.87** | **0.91** | **0.89** |

Both models perform evenly across both classes — there's no sentiment bias where the model favors detecting positive over negative reviews (or vice versa). Notably, each class's precision and recall are near-mirror images of the other's (e.g. negative recall ≈ positive precision), which is the signature of a balanced model rather than one systematically stronger on one class. The 3-epoch model is consistently a little stronger across every metric, reinforcing the overfitting finding above rather than just improving one number in isolation.

## Key methodological note

The first version of this experiment reused the already-trained model object for the second run, meaning "3 epochs" was actually "5 + 3 = 8 epochs" continuing from the same weights — an invalid comparison. The results above are from a corrected run using a freshly initialized, untrained model for the 3-epoch run, ensuring the two epoch counts are compared fairly from the same starting point.

## Training details

- **Dataset**: IMDB movie reviews (`stanfordnlp/imdb`) — 3,000 training examples, 1,000 test examples (subsampled and shuffled with a fixed seed for reproducibility)
- **Model**: `distilbert-base-uncased`, fine-tuned via Hugging Face `Trainer`
- **Max sequence length**: 256 tokens
- **Batch size**: 16
- **Metric**: accuracy

## What I'd try next

- Add weight decay or dropout to the classifier head to test whether regularization allows training for more epochs without the same overfitting cost
- Try early stopping based on validation loss instead of a fixed epoch count
- Evaluate on the full IMDB test set (25,000 examples) rather than a 1,000-example subset, for a more statistically reliable accuracy figure
