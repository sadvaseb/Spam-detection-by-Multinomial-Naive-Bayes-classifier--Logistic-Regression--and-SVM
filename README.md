# Spam detection by Multinomial Naive Bayes classifier, Logistic Regression, and SVM
 This project develops some algorithms to identify spam emails from non-spam emails in a dataset. I use multiple models (e.i., Naive Bayes classifier, Logistic Regression, and SVM). I applied multiple feature engineering technics (e.g., Count Vectorizer, Tfidf Vectorizer, and adding document length, non-word characters, etc) to improve accuracy. The highest spam detection accuracy **(AUC=98%)** is achieved by a multi-feature Logistic Regression model.

This project is part of Applied Data Science with Python Specialization program at the University of Michigan. The program available from [here](https://www.coursera.org/learn/python-text-mining) . This code is only uploaded for educational purposes and it should not be used for submitting any homework or assignment.

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

Fitting and transforming the training data X_train using a Tfidf Vectorizer with default parameters. Then, finding the 20 features which have the smallest tf-idf and 20 features which have the largest tf-idf. Then, the features are sorted in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series is the feature name, and the data is the tf-idf.

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

I use Tfidf Vectorizer to fit and transform the training data X_train. Tfidf Vectorizer ignors terms that have a document frequency strictly lower than 3. Then I fitted a multinomial Naive Bayes classifier model with smoothing alpha=0.1 and compute the area under the curve (AUC) score using the transformed test data.


```python
vect = TfidfVectorizer(min_df=3).fit(X_train)
X_train_vectorized = vect.transform(X_train)
classifier = MultinomialNB(alpha=0.1)
model = classifier.fit(X_train_vectorized, y_train)
predicted_lables = model.predict(vect.transform(X_test))    
    
roc_auc_score(y_test,predicted_lables)


```

Comparing the average length of documents (number of characters) for not spam and spam documents, to find some insightful new features. 

```python
spams= spam_data[spam_data['target']== 1]
normal = spam_data[spam_data['target']== 0]    
(np.array([len(w) for w in normal['text']]).mean(), np.array([len(w) for w in spams['text']]).mean())


```

The following function has been provided to combine new features into the training data:

```python
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

```
I use a Tfidf Vectorizer to fit and transform the training data X_train. Tfidf Vectorizer ignores terms that have a document frequency strictly lower than 5. Then, I use this document-term matrix and an additional feature, the length of document (number of characters), to fit a Support Vector Classification model with regularization C=10000. Finally, the area under the curve (AUC) score is calculated for the transformed test data.

```python
from sklearn.svm import SVC

vect = TfidfVectorizer(min_df=5).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_train_new_feature = add_feature(X_train_vectorized, X_train.str.len())
classifier = SVC(C=10000)
model = classifier.fit(X_train_new_feature, y_train)

X_test_vectorized = vect.transform(X_test)
predicted_lables = model.predict(add_feature(X_test_vectorized, X_test.str.len()))    
roc_auc_score(y_test,predicted_lables)

```
Comparing the average number of digits per document for not spam and spam documents, to find some insightful new features. 

```python
spams = spam_data[spam_data['target']== 1]
normal = spam_data[spam_data['target']== 0]    
    
(np.array([sum(c.isdigit() for c in document) for document in normal['text']]).mean(), 
        np.array([sum(c.isdigit() for c in document) for document in spams['text']]).mean()) 

```

I fit and transform the training data X_train using a Tfidf Vectorizer. The Tfidf Vectorizer ignores terms that have a document frequency strictly lower than 5 and using word n-grams from n=1 to n=3 (unigrams, bigrams, and trigrams). I also added two new features to document-term matrix:

1) the length of document (number of characters)
2) number of digits per document

Then, I fit a Logistic Regression model with regularization C=100 and compute the area under the curve (AUC) score using the transformed test data.

```python
from sklearn.linear_model import LogisticRegression

vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_train_new_feature = add_feature(X_train_vectorized, X_train.str.len())
X_train_new_feature = add_feature(X_train_new_feature, 
                                  [sum(c.isdigit() for c in document) for document in X_train])

X_test_vectorized = vect.transform(X_test)
X_test_new_feature = add_feature(X_test_vectorized, X_test.str.len())
X_test_new_feature = add_feature(X_test_new_feature, 
                                  [sum(c.isdigit() for c in document) for document in X_test])

classifier = LogisticRegression(C=100)
model = classifier.fit(X_train_new_feature, y_train)
predicted_lables = model.predict(X_test_new_feature)    
roc_auc_score(y_test,predicted_lables)

```

Calculating the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents, to find some good new features. 

```python
import re

spams = spam_data[spam_data['target']== 1]
normal = spam_data[spam_data['target']== 0]   
(np.array([len(re.findall(r'\W', document)) for document in normal['text']]).mean(),
         np.array([len(re.findall(r'\W', document)) for document in spams['text']]).mean())


```

I use a Count Vectorizer to fit and transform the training data X_train. The Count Vectorizer ignores terms that have a document frequency strictly lower than 5 and using character n-grams from n=2 to n=5.

Character n-grams creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.

Then, I add three new features to the document-term matrix:

1) the length of document (number of characters)
2) number of digits per document
3) number of non-word characters (anything other than a letter, digit or underscore.)

I used a Logistic Regression model with regularization C=100 and compute the area under the curve (AUC) score using the transformed test data.

Finally, I list the 10 smallest and 10 largest coefficients from the model and return them along with the AUC score in a tuple.

This code returns a tuple (AUC score as a float, smallest coefs list, largest coefs list).

```python
vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer= 'char_wb').fit(X_train)

X_train_vectorized = vect.transform(X_train)
X_train_new_feature = add_feature(X_train_vectorized, X_train.str.len())
X_train_new_feature = add_feature(X_train_new_feature, 
                                  [sum(c.isdigit() for c in document) for document in X_train])
X_train_new_feature = add_feature(X_train_new_feature, 
                                  [len(re.findall(r'\W', document)) for document in X_train])

X_test_vectorized = vect.transform(X_test)
X_test_new_feature = add_feature(X_test_vectorized, X_test.str.len())
X_test_new_feature = add_feature(X_test_new_feature, 
                                  [sum(c.isdigit() for c in document) for document in X_test])
X_test_new_feature = add_feature(X_test_new_feature, 
                                  [len(re.findall(r'\W', document)) for document in X_test])

classifier = LogisticRegression(C=100)
model = classifier.fit(X_train_new_feature, y_train)

predicted_lables = model.predict(X_test_new_feature)    


t = classifier.coef_[0].argsort()
feature_names = np.array(vect.get_feature_names())

(roc_auc_score(y_test,predicted_lables) ,
     pd.Series([classifier.coef_[0][t[i]] for i in range (10)], index =[feature_names[t[i]] for i in range (10)]) ,
     pd.Series([classifier.coef_[0][t[-i-1]] for i in range (10)], index =[feature_names[t[-i-2]] for i in range (10)])
    )   
```