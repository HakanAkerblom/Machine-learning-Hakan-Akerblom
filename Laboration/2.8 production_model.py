import pandas as pd
import joblib

# Loading 100 samples
df = pd.read_csv("Laboration/test_samples.csv", index_col=0)

# Loading pipeline with model
model = joblib.load("Laboration/pipe.pkl")

# Defining X and y-test
X_test = df.drop("cardio", axis=1)
y_test = df["cardio"]

# Predicting
y_pred = model.predict_proba(X_test)

# Printing score
print("Accuracy score: ", model.score(X_test, y_test))

# Making dataframe with predictions
predictions = pd.DataFrame(y_pred, columns=['Probability class 0', 'Probability class 1'])
predictions['Prediction'] = model.predict(X_test)

# export the predictions to a csv file
predictions.to_csv('Laboration/predictions.csv', index=False)

