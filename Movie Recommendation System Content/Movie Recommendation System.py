#!/usr/bin/env python
# coding: utf-8

# # <font color="orange">Movie Recommendation System </font>

# ## Import Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")


# ## Loading Required Dataset

# In[2]:


df1 = pd.read_csv("tmdb_movies.csv")


# In[3]:


df2=pd.read_csv("tmdb_credits.csv")


# ### Top 5 rows of df1

# In[4]:


df1.head(5)


# ### Top 5 rows of df1

# In[5]:


df2.head(5)


# ### Features in df1

# In[6]:


df1.columns


# ### Features in df2

# In[7]:


df2.columns


# ### Finding common column in both datasets

# In[8]:


for i in df1:
    for j in df2:
        if i==j:
            print("the common column in both datasets is {0} so have to merge on {0}".format(i))


# ### Merging Datasets

# In[9]:


movie_df=df1.merge(df2,on="title")


# In[10]:


movie_df.head()


# ### shape of final dataset

# In[11]:


movie_df.shape


# ### The number of unique values in each feature

# In[12]:


movie_df.nunique()


# ### Inspection on dataset using info methods

# In[13]:


movie_df.info()


# # Content Based Filtering

# ### Selection of required fetures for Content Based Filtering
#     movie_id
#     title
#     overview
#     genres
#     keywords
#     cast
#     crew

# In[14]:


movie_df=movie_df[["movie_id","title","overview","genres","keywords","cast","crew"]]


# ### Data cleaning

# In[15]:


# Checking null values
movie_df.isnull().sum()


# In[16]:


# Percentage of Null values
a=(((movie_df.isnull().sum())/len(movie_df))*100).round(2)
a


# In[17]:


# Only feature overview has 0.06% of null values
# better to drop null values
movie_df.dropna(inplace=True)


# In[18]:


movie_df.isnull().sum()


# In[19]:


movie_df.isnull().sum().sum()


# ### Checking Duplicate Values

# In[20]:


movie_df.duplicated().sum()


# ### Data cleaning feature wise

# #### column genres

# In[21]:


movie_df.genres[0]


# In[22]:


# selection of name data like ["Action","Adventure","Fantasy","Science Fiction"]


# In[23]:


import ast
def change(object):
    L = []
    for i in ast.literal_eval(object):
        L.append(i["name"])
    return L


# In[24]:


movie_df["genres"]=movie_df["genres"].apply(change)
movie_df["genres"]


# In[25]:


#Space removel between two words
movie_df["genres"].apply(lambda x: [i.replace(" ","") for i in x])


# #### column keywords

# In[26]:


movie_df["keywords"] = movie_df["keywords"].apply(change)
movie_df["keywords"]


# In[27]:


#Space removel between two words
movie_df["keywords"].apply(lambda x: [i.replace(" ","") for i in x])


# #### column cast

# In[28]:


movie_df.cast[0]


# In[29]:


def changeA(object):
    L=[]
    count = 0
    for i in ast.literal_eval(object):
        if count != 3:
            L.append(i["name"])
            count+=1
        else:
            break
    return L


# In[30]:


movie_df["cast"] = movie_df["cast"].apply(changeA)


# In[31]:


movie_df["cast"]


# In[32]:


# Space removel between two words
movie_df["cast"].apply(lambda x: [i.replace(" ","") for i in x])


# #### column crew

# In[33]:


movie_df["crew"][0]


# In[34]:


def director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L


# In[35]:


movie_df["crew"]=movie_df["crew"].apply(director)
movie_df["crew"]


# In[36]:


#Space removel between two words
movie_df["crew"].apply(lambda x: [i.replace(" ","") for i in x])


# #### column overview

# In[37]:


movie_df["overview"][0]


# In[38]:


movie_df["overview"]=movie_df["overview"].apply(lambda x:x.split())


# In[39]:


movie_df["overview"]


# ### Combining of the features into single feature as Tags

# In[40]:


movie_df["Tags"]=movie_df["overview"]+movie_df["genres"]+movie_df["keywords"]+                 movie_df["cast"]+movie_df["crew"]


# In[41]:


movie_df["Tags"]=movie_df["Tags"].apply(lambda x:" ".join(x))
movie_df["Tags"]


# In[42]:


# Converting all the letters into lower case using string method
movie_df["Tags"]=movie_df["Tags"].apply(lambda x:x.lower())
movie_df["Tags"][0]


# ## Selection of features for final dataset

# In[43]:


final_df=movie_df[["movie_id","title","Tags"]]


# In[44]:


final_df["title"]=final_df["title"].apply(lambda x: x.lower())
final_df["title"]


# In[45]:


final_df.head()


# ### Convertion of words into root from 

# In[46]:


import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[47]:


ps.stem("eating")


# In[48]:


ps.stem("running")


# In[49]:


ps.stem("dispatched")


# In[50]:


ps.stem('civilization')


# In[51]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[52]:


final_df["Tags"].apply(stem)


# ### Convertion of words into vectores

# In[53]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")
vectors=cv.fit_transform(final_df["Tags"]).toarray()


# In[54]:


vectors[1]


# In[55]:


vectors[5]


# ### Calculating the similarity between movies

# In[56]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)


# In[57]:


def recommend(movie):
    movie_index=final_df[final_df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    print(" ")
    print("****Similar Movies****")
    print(" ")
    for i in movies_list:
        print(final_df.iloc[i[0]].title)


# In[61]:


recommend(input("Enter a Movie Name : "))


# #### <font color="red">Conclusions:</font>
# <font color="green"></font>
# * <font color="green">Here The Recommendation System is Developed for 5000 List of Movies from TMDB</font>
# * <font color="green">Movie Recommendation System is Developed Based on Content Based Recommendation</font>
# * <font color="green">We Have to Give Movie Name (Keyword) as Input and it Recommends Top 5 Movies as Output</font>
# * <font color="green">These Top 5 Movies Recommended Using Content Based Filtering Technique and Cosine Similarity</font>
# 

# In[ ]:




