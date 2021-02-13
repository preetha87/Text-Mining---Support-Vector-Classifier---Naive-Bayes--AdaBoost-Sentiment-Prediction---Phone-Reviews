#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re, nltk
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


# In[2]:


#Cleaning Product Reviews
def normalizer(rev): 
    soup = BeautifulSoup(rev, 'lxml') 
    souped = soup.get_text()
    re1 = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", " ", souped) # removing @mentions, urls
    re2 = re.sub("[^A-Za-z]+"," ", re1) # removing numbers

    tokens = nltk.word_tokenize(re2)
    removed_letters = [word for word in tokens if len(word)>2] # removing words with length less than or equal to 2
    lower_case = [l.lower() for l in removed_letters]

    stop_words = set(stopwords.words('english'))
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmas = [wordnet_lemmatizer.lemmatize(t, pos='v') for t in filtered_result]
    return lemmas


# In[3]:


def Cross_validation(data, targets, clf_cv, model_name): #### Performs cross-validation on SVC

    kf = KFold(n_splits=10, shuffle=True, random_state=1) # 10-fold cross-validation
    scores=[]
    data_train_list = []
    targets_train_list = []
    data_test_list = []
    targets_test_list = []
    iteration = 0
    print("Performing cross-validation for {}...".format(model_name))
    for train_index, test_index in kf.split(data):
        iteration += 1
        print("Iteration ", iteration)
        data_train_cv, targets_train_cv = data[train_index], targets[train_index]
        data_test_cv, targets_test_cv = data[test_index], targets[test_index]
        data_train_list.append(data_train_cv) # appending training data for each iteration
        data_test_list.append(data_test_cv) # appending test data for each iteration
        targets_train_list.append(targets_train_cv) # appending training targets for each iteration
        targets_test_list.append(targets_test_cv) # appending test targets for each iteration
        clf_cv.fit(data_train_cv, targets_train_cv) # Fitting SVC
        score = clf_cv.score(data_test_cv, targets_test_cv) # Calculating accuracy
        print("Cross-validation accuracy: ", score)
        scores.append(score) # appending cross-validation accuracy for each iteration
    mean_accuracy = np.mean(scores)
    print("Mean cross-validation accuracy for {}: ".format(model_name), mean_accuracy)
    print("Best cross-validation accuracy for {}: ".format(model_name), max(scores))
    max_acc_index = scores.index(max(scores)) # best cross-validation accuracy
    max_acc_data_train = data_train_list[max_acc_index] # training data corresponding to best cross-validation accuracy
    max_acc_data_test = data_test_list[max_acc_index] # test data corresponding to best cross-validation accuracy
    max_acc_targets_train = targets_train_list[max_acc_index] # training targets corresponding to best cross-validation accuracy
    max_acc_targets_test = targets_test_list[max_acc_index] # test targets corresponding to best cross-validation accuracy

    return mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test


# In[4]:


def c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, targets, clf, model_name):
    clf.fit(max_acc_data_train, max_acc_targets_train) # Fitting classifier
    targets_pred = clf.predict(max_acc_data_test) # Prediction on test data
    conf_mat = confusion_matrix(max_acc_targets_test, targets_pred)
    sns.heatmap(conf_mat, annot=True)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("Confusion Matrix (Best Accuracy) - {}".format(model_name))
    plt.show()
    print('Confusion matrix: \n', conf_mat)
    print('TP: ', conf_mat[1,1])
    print('TN: ', conf_mat[0,0])
    print('FP: ', conf_mat[0,1])
    print('FN: ', conf_mat[1,0])


# CREATION OF SVC AND NAIVE BAYES CLASSIFIERS

# In[ ]:


def main():
    #### Reading training dataset as dataframe
    df = pd.read_csv("user_reviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Converting Recommended variable to a numerical variable
    def converter(column):
        if column>=8:
            return "Yes"
        else:
            return "No"
    df['Recommended'] = df['score'].apply(converter)
    
    df['Recommended'] = df['Recommended'].map({'Yes':1, 'No':0})
    #### Normalizing reviews
    df['normalized_review'] = df.extract.apply(normalizer)
    df = df[df['normalized_review'].map(len) > 0] # removing rows with normalized reviews of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
    print(df[['extract','normalized_review']].head())
    df.drop(['source', 'domain', 'score_max', 'extract','product'], axis=1, inplace=True)
    #### Saving cleaned reviews to csv
    df.to_csv('Cleaned_UserReviews_Unigrams.csv', encoding='utf-8', index=False)
    #### Reading cleaned reviews as dataframe
    cleaned_data = pd.read_csv("Cleaned_UserReviews_Unigrams.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = cleaned_data.normalized_review
    targets = cleaned_data.Recommended
    tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,1)) 
    tfidf.fit(data) #Learning vocabulary - creating variables
    data = tfidf.transform(data) # Generating tfidf matrix
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary - unigrams.csv', header=False)
    print("Shape of tfidf matrix: ", data.shape)
    #### Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
    print("Number of observations in each class before oversampling: \n", pd.Series(targets).value_counts())

    smote = SMOTE(random_state = 101)
    data,targets = smote.fit_sample(data,targets)

    print("Number of observations in each class after oversampling: \n", pd.Series(targets).value_counts())
    
    SVC_clf = LinearSVC() # SVC Model
    SVC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test = Cross_validation(data, targets, SVC_clf, "SVC") # SVC cross-validation
    c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, targets, SVC_clf, "SVC") # SVC confusion matrix

    NBC_clf = MultinomialNB() # NBC Model
    NBC_mean_accuracy, max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test = Cross_validation(data, targets, NBC_clf, "NBC") # NBC cross-validation
    c_matrix(max_acc_data_train, max_acc_data_test, max_acc_targets_train, max_acc_targets_test, targets, NBC_clf, "NBC") # NBC confusion matrix
    

    if SVC_mean_accuracy > NBC_mean_accuracy:
        clf = LinearSVC().fit(data, targets)
        joblib.dump(clf, 'svc-unigrams.sav')
    else:
        clf = MultinomialNB().fit(data, targets)
        joblib.dump(clf, 'nbc-unigrams.sav')
    

if __name__ == "__main__":
    main()


# CREATION OF AN ADABOOST CLASSIFIER

# In[5]:


def main():
    #### Reading training dataset as dataframe
    df = pd.read_csv("user_reviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Converting Recommended variable to a numerical variable
    def converter(column):
        if column>=8:
            return "Yes"
        else:
            return "No"
    df['Recommended'] = df['score'].apply(converter)
    
    df['Recommended'] = df['Recommended'].map({'Yes':1, 'No':0})
    #### Normalizing reviews
    df['normalized_review'] = df.extract.apply(normalizer)
    df = df[df['normalized_review'].map(len) > 0] # removing rows with normalized reviews of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
    print(df[['extract','normalized_review']].head())
    df.drop(['source', 'domain', 'score_max', 'extract','product'], axis=1, inplace=True)
    #### Saving cleaned reviews to csv
    df.to_csv('Cleaned_UserReviews.csv', encoding='utf-8', index=False)
    #### Reading cleaned reviews as dataframe
    cleaned_data = pd.read_csv("Cleaned_UserReviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = cleaned_data.normalized_review
    targets = cleaned_data.Recommended
    tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,1)) 
    tfidf.fit(data) #Learning vocabulary - creating variables
    data = tfidf.transform(data) # Generating tfidf matrix
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary.csv', header=False)
    print("Shape of tfidf matrix: ", data.shape)
    #### Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
    print("Number of observations in each class before oversampling: \n", pd.Series(targets).value_counts())

    smote = SMOTE(random_state = 101)
    data,targets = smote.fit_sample(data,targets)

    print("Number of observations in each class after oversampling: \n", pd.Series(targets).value_counts())
    
    abc = AdaBoostClassifier(random_state=1)
    grid_param = {'n_estimators': [5,10,20,30,40,50,60,70,80,90,100]}
    gd_sr = GridSearchCV(estimator=abc, param_grid=grid_param, scoring='accuracy', cv=10)
    gd_sr.fit(data, targets)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_ 
    print(best_result)

if __name__ == "__main__":
    main()


# In[6]:


def main():
    #### Reading training dataset as dataframe
    df = pd.read_csv("user_reviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Converting Recommended variable to a numerical variable
    def converter(column):
        if column>=8:
            return "Yes"
        else:
            return "No"
    df['Recommended'] = df['score'].apply(converter)
    
    df['Recommended'] = df['Recommended'].map({'Yes':1, 'No':0})
    #### Normalizing reviews
    df['normalized_review'] = df.extract.apply(normalizer)
    df = df[df['normalized_review'].map(len) > 0] # removing rows with normalized reviews of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
    print(df[['extract','normalized_review']].head())
    df.drop(['source', 'domain', 'score_max', 'extract','product'], axis=1, inplace=True)
    #### Saving cleaned reviews to csv
    df.to_csv('Cleaned_UserReviews.csv', encoding='utf-8', index=False)
    #### Reading cleaned reviews as dataframe
    cleaned_data = pd.read_csv("Cleaned_UserReviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = cleaned_data.normalized_review
    targets = cleaned_data.Recommended
    tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,1)) 
    tfidf.fit(data) #Learning vocabulary - creating variables
    data = tfidf.transform(data) # Generating tfidf matrix
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary.csv', header=False)
    print("Shape of tfidf matrix: ", data.shape)
    #### Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
    print("Number of observations in each class before oversampling: \n", pd.Series(targets).value_counts())

    smote = SMOTE(random_state = 101)
    data,targets = smote.fit_sample(data,targets)

    print("Number of observations in each class after oversampling: \n", pd.Series(targets).value_counts())
    
    abc = AdaBoostClassifier(random_state=1)
    grid_param = {'n_estimators': [110,120,130,140,150,160,170,180,190,200]}
    gd_sr = GridSearchCV(estimator=abc, param_grid=grid_param, scoring='accuracy', cv=10)
    gd_sr.fit(data, targets)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_ 
    print(best_result)

if __name__ == "__main__":
    main()


# In[7]:


def main():
    #### Reading training dataset as dataframe
    df = pd.read_csv("user_reviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Converting Recommended variable to a numerical variable
    def converter(column):
        if column>=8:
            return "Yes"
        else:
            return "No"
    df['Recommended'] = df['score'].apply(converter)
    
    df['Recommended'] = df['Recommended'].map({'Yes':1, 'No':0})
    #### Normalizing reviews
    df['normalized_review'] = df.extract.apply(normalizer)
    df = df[df['normalized_review'].map(len) > 0] # removing rows with normalized reviews of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
    print(df[['extract','normalized_review']].head())
    df.drop(['source', 'domain', 'score_max', 'extract','product'], axis=1, inplace=True)
    #### Saving cleaned reviews to csv
    df.to_csv('Cleaned_UserReviews.csv', encoding='utf-8', index=False)
    #### Reading cleaned reviews as dataframe
    cleaned_data = pd.read_csv("Cleaned_UserReviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = cleaned_data.normalized_review
    targets = cleaned_data.Recommended
    tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,1)) 
    tfidf.fit(data) #Learning vocabulary - creating variables
    data = tfidf.transform(data) # Generating tfidf matrix
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary.csv', header=False)
    print("Shape of tfidf matrix: ", data.shape)
    #### Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
    print("Number of observations in each class before oversampling: \n", pd.Series(targets).value_counts())

    smote = SMOTE(random_state = 101)
    data,targets = smote.fit_sample(data,targets)

    print("Number of observations in each class after oversampling: \n", pd.Series(targets).value_counts())
    
    abc = AdaBoostClassifier(random_state=1)
    grid_param = {'n_estimators': [210,220,230,240,250,260,270,280,290,300]}
    gd_sr = GridSearchCV(estimator=abc, param_grid=grid_param, scoring='accuracy', cv=10)
    gd_sr.fit(data, targets)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_ 
    print(best_result)

if __name__ == "__main__":
    main()


# In[8]:


def main():
    #### Reading training dataset as dataframe
    df = pd.read_csv("user_reviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1) # Setting this so we can see the full content of cells
    #### Converting Recommended variable to a numerical variable
    def converter(column):
        if column>=8:
            return "Yes"
        else:
            return "No"
    df['Recommended'] = df['score'].apply(converter)
    
    df['Recommended'] = df['Recommended'].map({'Yes':1, 'No':0})
    #### Normalizing reviews
    df['normalized_review'] = df.extract.apply(normalizer)
    df = df[df['normalized_review'].map(len) > 0] # removing rows with normalized reviews of length 0
    print("Printing top 5 rows of dataframe showing original and cleaned reviews....")
    print(df[['extract','normalized_review']].head())
    df.drop(['source', 'domain', 'score_max', 'extract','product'], axis=1, inplace=True)
    #### Saving cleaned reviews to csv
    df.to_csv('Cleaned_UserReviews.csv', encoding='utf-8', index=False)
    #### Reading cleaned reviews as dataframe
    cleaned_data = pd.read_csv("Cleaned_UserReviews.csv", encoding = "ISO-8859-1")
    pd.set_option('display.max_colwidth', -1)
    data = cleaned_data.normalized_review
    targets = cleaned_data.Recommended
    tfidf = TfidfVectorizer(min_df=10, ngram_range=(1,1)) 
    tfidf.fit(data) #Learning vocabulary - creating variables
    data = tfidf.transform(data) # Generating tfidf matrix
    pd.DataFrame.from_dict(data=dict([word, i] for i, word in enumerate(tfidf.get_feature_names())), orient='index').to_csv('vocabulary.csv', header=False)
    print("Shape of tfidf matrix: ", data.shape)
    #### Implementing Oversampling to balance the dataset; SMOTE stands for Synthetic Minority Oversampling TEchnique
    print("Number of observations in each class before oversampling: \n", pd.Series(targets).value_counts())

    smote = SMOTE(random_state = 101)
    data,targets = smote.fit_sample(data,targets)

    print("Number of observations in each class after oversampling: \n", pd.Series(targets).value_counts())
    
    abc = AdaBoostClassifier(random_state=1)
    grid_param = {'n_estimators': [310,320,330,340,350,360,370,380,390,400]}
    gd_sr = GridSearchCV(estimator=abc, param_grid=grid_param, scoring='accuracy', cv=10)
    gd_sr.fit(data, targets)
    best_parameters = gd_sr.best_params_
    print(best_parameters)
    best_result = gd_sr.best_score_ 
    print(best_result)

if __name__ == "__main__":
    main()


# In[ ]:




