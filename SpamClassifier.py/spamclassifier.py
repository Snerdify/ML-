#how should we read the dataset
#use pandas to read the dataset
import pandas as pd 

#pd.read_csv= reads the dataset file
#seperator \t sepeates the dependent feature 'ham' and rest of the sentence
# 'ham/spam' and the sentences are sepearted by tab and npt space

#after u seperate provide 2 column names, 'label'='ham/spam;
#and 'msg' = sentence
messages=pd.read_csv('dataset\SMSSpamCollection',sep='\t',
                            names=['label','message'])




#cleaning and preprocessing the data

import re #regular expression lib
import nltk
#download stopwords
nltk.download('stopwords')
#import stopwords
from nltk.corpus import stopwords
#import the stemming library -PorterStemmer
from nltk.stem.porter import PorterStemmer

#create a stemmer object

ps = PorterStemmer()

#create an empty list called corpus to store all the 
#cleaned up sentences 

corpus = []

#iterate over all the messages 

for i in range(len(messages)):
    #use re to remove all the commas,punctuations
    # and only keep the characters a to A and z to Z
    review= re.sub('[^a-zA-Z]', messages['message'][i])

    #connvert all the words to lowercase
    review = review.lower()

    #split up the sentences to get list of words
    review = review.split()

    #use the stemming object to do stemming if the words are
    # not the stopwords(unwanted words-he,she,it,etc)- dont
    # help in understanding positive and negative sentiments
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] 
    
    # join all the imp words and make imp sentences
    #that can used to do analysis

    review = ' '.join(review)

    #append these imp sentences to corpus list 
    corpus.append(review)



# prepare the bag of words model

from sklearn.feature_extraction.text import CountVectorizer


#create a Countervectorizer objcet which accpets max features as 2500
#these will be top 2500 most frequent words
cv = Countervectorizer(max_features =2500)
X = cv.fit_transform(corpus).toarray()


# output data -  label column
# convert ham and spam into dummy variables(1,0)

y = pd.get_dummies(messages['label'])
# but we need only one column to undersatnd which msg is
#spam and which is not 
#bcoz when spam is 0 , ham is 1
# and when ham is 0, spam is 1.
# so only one of these columns will tell us the complete info
# use iloc to get one column
y = y.iloc[:,1].values


#now x(independent feature) and y(dependant feature )
# both are ready
# so now do the train , test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=0)


# train the model using naive bayes classifier


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)
y_pred = spam_detect_model.predict(X_test)




#confusion matrix
from sklearn import confusion_matrix
con_m = confusion_matrix(y_test,y_pred)




#accuracy

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred) 






