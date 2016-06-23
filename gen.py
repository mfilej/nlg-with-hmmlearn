import pickle
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

with open("hmm.le", "rb") as f:
    le = pickle.load(f)

model = joblib.load("hmm.pkl")

symbols, _states = model.sample(100)

output = le.inverse_transform(np.squeeze(symbols))
for word in output:
    print word,
