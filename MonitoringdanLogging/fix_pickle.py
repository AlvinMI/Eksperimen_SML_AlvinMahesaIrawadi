import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

X = np.zeros((1, 13))
y = np.array([1])

model = RandomForestClassifier()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

print("✅ model.pkl BARU BERHASIL DIBUAT (Protocol 4)")