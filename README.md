# toxic
A classification model trained on internet forum posts used to identify toxic and hateful comments. <br>
Skills explored: **Random Forests**, **MLPs**, **Feature Selection**, **Confusion Matrices**, _Decision Trees_, _KNN._ <br>
Frameworks used: **Scikit-learn**, **ONNX**, _NumPy_, _Matplotlib_ <br>

## How to use this model: <br>
Download ```model.onnx, vectorizer.onnx, classify.py```. <br>
Install dependencies: ```pip install onnxruntime numpy``` <br>
1. run ```python [-i] classify.py``` to enter strings interactively. <br>
2. ```from classify import classify``` in your scripts, enter path to model & vectorizer, and call ```classify(str | np.array)```. <br>
3. ```import classify``` introduces a global variable ```classify.model``` which allows direct access to the underlying model via ```.predict()```. <br>

## Model Architecture
The model uses a **Random Forest classifier** with a relatively large ensemble size of 100 trees. This learner is well fitted to an emergent information gain when splitting inputs by feature, discovered during data analysis. The model uses sklearn's **TfidfVectorizer** to transform input text into numeric vectors. <br>
**Input**: A string with less than 5000 characters. <br>
**Output**: ```np.array``` of binary labels corresponding to classes ```['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']```. <br>

## Model metrics <br>
The validation error rate is **~3.18%**. <br>
ROC_AUC score: __*0.56*__ (subject for improvement). <br>

|  label  | precision |
| ------- | -------   |
| toxic | 0.96 |
| severe_toxic | 0.97 |
| obscene | 0.98   |
| threat | 0.97    |
| insult | 0.98    |
| identity_hate | 0.97   |

#### (Transposed) Confusion Matrix:
|          | True    | False   |
| -------- | ------- | ------- |
| Negative | 57821   | 4648    |
| Positive | 1442    | 67      |

Notice the high amounts of false negatives compared to low amounts of false positives. This is an intentional trade-off to preserve a high prediction accuracy: the most important metric if this model was used as a comment filter, since falsely identifying a comment to be offensive will result in unjust punishments to users. Training a K-Nearest Neighbor model yields an almost-opposite confusion matrix.<br>

## Known issues
Due to the highly imbalanced proportion of non-toxic comments in the training data, some classifications such as ```identity_hate``` have very high false negative rates, despite heavy feature selection on the train dataset. The author will attempt to use under/over-sampling techniques to improve this issue.
