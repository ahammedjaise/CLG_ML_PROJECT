from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Load dataset
categories = ['rec.sport.baseball', 'sci.space', 'alt.atheism']
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
x = vectorizer.fit_transform(data.data)
y = data.target

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Naive Bayes
model = MultinomialNB()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

# Manual test
new_text =new_text = [
    "who are you to judge me"
]
pred_index=()
