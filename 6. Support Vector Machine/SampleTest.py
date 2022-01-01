import sys
import importlib
import pandas as pd


try:
    mymodule = importlib.import_module('SVM')
except Exception as e:
    print("Error: Source File Not Found.")
    sys.exit()

data = pd.read_csv('test.csv')
X_test = data.iloc[:, 0:-1]
y_test = data.iloc[:, -1]

try:
    #for i in range(1,20):
    model = mymodule.SVM('train.csv').solve()
    print(f'Accuracy: {model.score(X_test, y_test)*100:.2f}%')
except Exception as e:
    print(f'Failed {e}')
