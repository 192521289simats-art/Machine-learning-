import pandas as pd
from io import StringIO
from sklearn.tree import DecisionTreeClassifier, export_text

# Play Tennis training data
csv_data = """Outlook,Temperature,Humidity,Wind,Play
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Rain,Cool,Normal,Strong,No
Overcast,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Sunny,Cool,Normal,Weak,Yes
Rain,Mild,Normal,Weak,Yes
Sunny,Mild,Normal,Strong,Yes
Overcast,Mild,High,Strong,Yes
Overcast,Hot,Normal,Weak,Yes
Rain,Mild,High,Strong,No
"""

data = pd.read_csv(StringIO(csv_data))

# Convert text features into numbers for the classifier
X = pd.get_dummies(data.drop(columns="Play"))
y = data["Play"]

# ID3 uses information gain; entropy is the splitting criterion
model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(X, y)

print("Decision tree:")
print(export_text(model, feature_names=list(X.columns)))

# Classify a new example
sample = pd.DataFrame([{
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}])

sample_encoded = pd.get_dummies(sample).reindex(columns=X.columns, fill_value=0)
print("New sample prediction:", model.predict(sample_encoded)[0])
