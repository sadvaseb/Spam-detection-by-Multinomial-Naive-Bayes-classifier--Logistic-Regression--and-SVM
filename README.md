# Spam detection by Multinomial Naive Bayes classifier, Logistic Regression, and SVM
 This project develops some algorithms to identify spam emails from non-spam emails in a dataset. I use multiple models (e.i., Naive Bayes classifier, Logistic Regression, and SVM). I applied multiple feature engineering technics (e.g., Count Vectorizer, Tfidf Vectorizer, and adding document length, non-word characters, etc) to improve accuracy. The highest spam detection accuracy **(AUC=98%)** is achieved by a multi-feature Logistic Regression model.

This project is part of Applied Data Science with Python Specialization program at the University of Michigan. The program available from [here](https://www.coursera.org/learn/python-text-mining) .

I start with loading the dataset and dividing it to test and train datasets.

```python
import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)

```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)

```

Estimating the percentage of spam emails in the document.

```python
spam_data['target'].mean()*100

```

Fitting the training data X_train using a Count Vectorizer with default parameters. Then, finding the longest token in the vocabulary.

```python
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer().fit(X_train)
t = [len(w) for w in vect.get_feature_names()[:]] 
vect.get_feature_names()[np.argmax(t)]

```

Fitting and transforming the training data X_train using a Count Vectorizer with default parameters. Then, fitting a multinomial Naive Bayes classifier model with smoothing alpha=0.1. Finally, finding the area under the curve (AUC) score using the transformed test data.

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

```python
vect = CountVectorizer().fit(X_train)
x_train_vectorized = vect.transform(X_train)
classify = MultinomialNB(alpha=0.1)
classify.fit(x_train_vectorized, y_train)
predicted_lables = classify.predict(vect.transform(X_test))    
roc_auc_score(y_test,predicted_lables)

```

Fitting and transforming the training data X_train using a Tfidf Vectorizer with default parameters. Then, finding the 20 features which have the smallest tf-idf and 20 features which have the largest tf-idf. Then, the features are sorted in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series is the feature name, and the data should be the tf-idf.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer().fit(X_train)
features = np.array(vect.get_feature_names())
X_train_vectorized = vect.transform(X_train)
sorted_features = X_train_vectorized.max(0).toarray()[0].argsort()
values = X_train_vectorized.max(0).toarray()[0]
smalls  = pd.Series (values[sorted_features[:20]], index = features[sorted_features[:20]])
larges  = pd.Series (values[sorted_features[:-21:-1]], index = features[sorted_features[:-21:-1]])
smalls = smalls.reset_index()
smalls = smalls.sort_values([0, 'index'])
larges= larges.reset_index()
larges = larges.sort_values([0, 'index'], ascending=[False, True])
    
(pd.Series(np.array (smalls[0]), index = np.array(smalls['index'])), 
    pd.Series(np.array (larges[0]), index = np.array(larges['index'])))


```
