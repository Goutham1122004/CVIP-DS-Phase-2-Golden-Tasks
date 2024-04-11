import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv("D:/Internship_Tasks/Coders_Cave_Task/Phase 2/Build a spam filter using NLP - Task 2/emails.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# Plotting predicted classes
plt.figure(figsize=(8, 6))
sns.countplot(y_pred)
plt.title('Predicted Classes Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Plotting confidence scores for spam predictions
spam_confidence_scores = model.predict_proba(X_test)[:, 1]
plt.figure(figsize=(8, 6))
sns.histplot(spam_confidence_scores, bins=20, kde=True)
plt.title('Confidence Scores for Spam Predictions')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.show()

# Saving the model and vectorizer
joblib.dump(model, 'spam_filter_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Predicting for a new email
new_email = ["Get a free vacation now! Click here to claim."]
new_email_transformed = vectorizer.transform(new_email)
prediction = model.predict(new_email_transformed)
print("Predicted class:", "spam" if prediction[0] == 1 else "Not a Spam")
