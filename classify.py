import sys
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
import csv
import onnxruntime as rt


class Model:
    classes_ = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def __init__(self, model_path, vec_path):
        self.model = rt.InferenceSession(model_path, provider_options=Model.providers)
        self.vec = rt.InferenceSession(vec_path, provider_options=Model.providers)
        self.model_names = (self.model.get_inputs()[0].name, self.model.get_outputs()[0].name)
        self.vec_names = (self.vec.get_inputs()[0].name, self.vec.get_outputs()[0].name)

    def predict(self, comment: str | ArrayLike) -> np.array:
        X_vec = self.vec.run([self.vec_names[1]], {self.vec_names[0]: comment})[0]
        return self.model.run([self.model_names[1]], {self.model_names[0]: X_vec.astype(np.float32)})

model: Model = None

def is_interactive():
    return hasattr(sys, 'ps1') or sys.flags.interactive

def classify(comment: str | ArrayLike) -> dict:
    if (type(comment) == str):
        comment = np.array([comment])
    pred = model.predict(comment)[0]
    if (len(pred) == 1): # pretty display of single row outputs.
        return dict(zip(model.classes_, [True if y == 1 else False for y in pred[0]]))
    return pred

if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_path = Path(sys.argv[1])
        vec_path = Path(sys.argv[2])
    else:
        model_path = Path(input("Enter model path:\n"))
        if model_path is None:
            model_path = "./model.onnx"
        vec_path = Path(input("Enter vectorizer path:\n"))
        if vec_path is None:
            vec_path = "./vectorizer.onnx"
    print("Model path:", model_path)
    print("Vectorizer path:", vec_path)
    print("Loading model...")
    model = Model(model_path, vec_path)
    print("\nWelcome! ", end="")

    if is_interactive():
        print("Call classify(str) to classify text.")
    else:
        while True:
            print("Enter text to be classified:")
            text = input(">> ")
            if text in ["exit", "exit()"]:
                break
            print(classify(text))