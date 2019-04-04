
# coding: utf-8

# In[1]:


# # from google.colab import files

# # uploaded = files.upload()

# !mkvirtualenv keras_tf -p python3
# !workon keras_tf
# !pip install --upgrade tensorflow


# In[2]:


# import tensorflow as tf
# print("GPU Enabled in TensorFlow Backend @",tf.test.gpu_device_name())
# def authenticate():
#   !pip install -U -q PyDrive
#   from pydrive.auth import GoogleAuth
#   from pydrive.drive import GoogleDrive
#   from google.colab import auth
#   from oauth2client.client import GoogleCredentials
#   auth.authenticate_user()
#   gauth = GoogleAuth()
#   gauth.credentials = GoogleCredentials.get_application_default()
#   drive = GoogleDrive(gauth)
#   return drive


# In[3]:


# def download(folderID, fname, can_print_contents=False):
#   !pip install -U -q pprint
#   import pprint
#   drive = authenticate()
#   files = (drive.ListFile({'q': "'"+ folderID +"' in parents and trashed=false"}).GetList())
#   fdict = {file['title']:file['id'] for file in files}
#   if can_print_contents:
#     pprint.PrettyPrinter(indent=4).pprint(fdict)
#   train_downloaded = drive.CreateFile({'id': fdict[fname]})
#   train_downloaded.GetContentFile(fname)


# In[4]:


#FOLDERID =  '1VAx868PjZxNybXyOEG18Ka9cXy5hZiet'

#download(FOLDERID, 'sentclf_train.csv',True)
#download(FOLDERID, 'sentclf_test_data.csv')


# In[5]:


from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd 
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


import pandas as pd
data=pd.read_csv('train.csv',encoding='utf-8')
test_file=pd.read_csv("test_data.csv")


# In[7]:



data.drop(data.index[273514],axis=0,inplace=True)
data=data.reset_index()


# In[8]:


IDs=test_file['ID']
test_file.drop('ID',axis=1,inplace=True)


# In[9]:



test_file


# In[10]:


# from nltk.tokenize import sent_tokenize, word_tokenize
# import pandas as pd 
# from nltk.corpus import stopwords
# from nltk import pos_tag
# from sklearn.feature_extraction.text import CountVectorizer
# import nltk


# In[11]:


import nltk
nltk.download()


# In[12]:


documents=[]
for indx in range(len(data)):
    documents.append(word_tokenize(str(data['text'][indx])))
    if indx%1000==0:
      print(indx)
for indx in range(len(test_file)):
  documents.append(word_tokenize(str(test_file['text'][indx])))


# In[13]:


len(documents)


# In[14]:


# from nltk.corpus import wordnet
# def get_simple_pos(tag):
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.NOUN


# In[15]:


# from nltk.stem import WordNetLemmatizer
# lemmatizer=WordNetLemmatizer()


# In[16]:


# from nltk.corpus import stopwords
# import string
# stops=set(stopwords.words('english'))
# punctuation=list(string.punctuation)
# stops.update(punctuation)
# len(stops)


# In[17]:


# stops


# In[18]:


# def clean_review(words):
#     output_words=[]
# #     passing words as it is important ex. capital means proper noun
#     for w in words:
#         if w.lower() not in stops:
#             pos=pos_tag((w))
#             clean_word=lemmatizer.lemmatize(w,pos=get_simple_pos(pos[0][1]))
#             output_words.append(clean_word.lower())
#     return output_words
  


# In[19]:


# data.shape


# In[20]:


# # documents=[ (clean_review(documents[indx]),data.label[indx]) for indx in range(len(documents))]
# documentts=[]
# for indx in range(len(documents)):
#     documentts.append((clean_review(documents[indx])))
#     if indx%1000==0:
#       print(indx)


# In[21]:


len(documents)


# In[22]:


# len(documentts)


# In[23]:


category=[data['label'][indx] for indx in range(len(data))]
len(category)


# In[24]:


text_documents=[ " ".join(document) for document in documents ]


# In[25]:


len(text_documents)


# In[26]:


#  RUN TILL HERE


# In[27]:


# import pickle


# In[28]:


# pickle_out=open("sentimentall.pickle","wb")

# pickle.dump(text_documents,pickle_out)   
# pickle_out.close()


# In[29]:


# pickle_in=open("sentimentall.pickle","rb")
# text_documents= pickle.load(pickle_in)  


# In[30]:


# text_documents


# In[31]:


len(text_documents)


# In[32]:


category


# In[33]:


# x_train=text_documents[0:400000]


# In[34]:


# x_test=text_documents[400000:]


# In[35]:


# len(x_train)


# In[36]:


# len(x_test)


# In[37]:


# y_train=category


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train, x_test,y_train, y_test=train_test_split(text_documents[0:400000], category,random_state=1)


# In[40]:


count_vec=CountVectorizer(max_features=3000,ngram_range=(1,2),max_df=0.97,min_df=0.05)
x_train_features=count_vec.fit_transform(x_train)
x_train_features.todense()


# In[41]:


len(count_vec.get_feature_names())


# In[42]:


x_test_features=count_vec.transform(x_test)


# In[43]:


x_test_features.todense()


# In[44]:


# Naive Bayes


# In[45]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train_features, y_train)


# In[46]:


clf.score( x_test_features, y_test)


# In[47]:


# import numpy as np

# sentim_NB = pd.DataFrame({'ID':IDs , "label": clf.predict(x_test_features)})
# sentim_NB.to_csv('senti_NB.csv', index=False)


# In[48]:


#svc


# In[49]:


from sklearn.svm import SVC


# In[ ]:


svc=SVC(C=100)
svc.fit(x_train_features, y_train)


# In[ ]:


svc.score(x_test_features, y_test)


# In[ ]:


# import numpy as np

# senti_svc = pd.DataFrame({'ID':IDs , "label": svc.predict(x_test_features)})
# senti_svc.to_csv('senti_svc.csv', index=False)


# In[ ]:


# 


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# clf = RandomForestClassifier()


# In[ ]:


# clf.fit(x_train_features, y_train)
# clf.score(x_test_features, y_test)

