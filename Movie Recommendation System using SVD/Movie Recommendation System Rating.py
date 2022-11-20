#!/usr/bin/env python
# coding: utf-8

# # <font color="orange">Movie Recommendation System </font>

# ### Importing the basic libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing & Parsing the dataset as ratings and movies details

# In[2]:


ratingData = pd.read_table('ratings.dat', 
names=['user_id', 'movie_id', 'rating', 'time'],engine='python', delimiter='::',encoding="ISO-8859-1")
movieData = pd.read_table('movies.dat',names=['movie_id', 'title', 'genre'],engine='python',
                          delimiter='::',encoding="ISO-8859-1")


# ### Basic Inspection on datasets

# In[3]:


# Top 5 rows of movie data
movieData.head()


# In[4]:


# Top 5 rows of rating data
ratingData.head()


# In[5]:


r,c=ratingData.shape
print("rating data having {} rows {} columns".format(r,c))


# In[6]:


r,c=movieData.shape
print("movie data having {} rows {} columns".format(r,c))


# In[7]:


movieData.size


# In[8]:


ratingData.size


# In[9]:


print('columns in the movie data: ',list(movieData.columns))


# In[10]:


print('columns in the rating data: ',list(ratingData.columns))


# In[11]:


len(movieData.movie_id.unique())


# In[12]:


len(ratingData.movie_id.unique())


# In[13]:


ratingData.info()


# In[14]:


movieData.info()


# In[15]:


# Checking null values
def checknull(obj):
    return obj.isnull().sum()


# In[16]:


movieData.apply(checknull)


# In[17]:


ratingData.apply(checknull)


# In[18]:


# Checking duplicate values
def checkduplicate(obj):
    return obj.duplicated().sum()


# In[19]:


movieData.apply(checkduplicate)


# In[20]:


ratingData.apply(checkduplicate)


# ### Create the ratings matrix of shape (m×u)

# In[27]:


ratingData.movie_id.values


# In[28]:


np.max(ratingData.movie_id.values)


# In[29]:


ratingData.user_id.values


# In[30]:


np.max(ratingData.user_id.values)


# In[31]:


ratingMatrix = np.ndarray(
    shape=(np.max(ratingData.movie_id.values), np.max(ratingData.user_id.values)),
    dtype=np.uint8)


# In[34]:


ratingData.movie_id.values-1


# In[36]:


ratingData.user_id.values-1


# In[37]:


ratingData.rating.values


# In[38]:


ratingMatrix[ratingData.movie_id.values-1, ratingData.user_id.values-1] = ratingData.rating.values


# In[39]:


print(ratingMatrix)


# ### Subtract Mean off - Normalization

# In[44]:


np.mean(ratingMatrix)


# In[43]:


np.mean(ratingMatrix, 1)


# In[45]:


np.mean(ratingMatrix, 1).shape


# In[48]:


np.asarray(np.mean(ratingMatrix, 1))


# In[49]:


np.asarray(np.mean(ratingMatrix, 1)).shape


# In[50]:


normalizedMatrix = ratingMatrix - np.asarray([(np.mean(ratingMatrix, 1))]).T


# In[51]:


print(normalizedMatrix)


# ### Computing SVD

# In[53]:


normalizedMatrix.T


# In[57]:


ratingMatrix.shape[0] - 1


# In[58]:


np.sqrt(ratingMatrix.shape[0] - 1)


# In[59]:


A = normalizedMatrix.T / np.sqrt(ratingMatrix.shape[0] - 1)
A


# In[61]:


U, S, V = np.linalg.svd(A)


# ### Calculate cosine similarity, sort by most similar and return the top N

# In[62]:


def similar(ratingData, movie_id, top_n):
    index = movie_id - 1 # Movie id starts from 1
    movie_row = ratingData[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', ratingData, ratingData)) #Einstein summation |  traditional matrix multiplication and is equivalent to np.matmul(a,b)
    similarity = np.dot(movie_row, ratingData.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity) #Perform an indirect sort along the given axis (Last axis)
    return sort_indexes[:top_n]


# ### Select k principal components to represent the movies, a movie_id to find recommendations and print the top_n results

# In[64]:


k = int(input("enter the total number of movies: "))
movie_id = int(input("enter the movie id: "))
top_n = int(input("ton n movies: "))

sliced = V.T[:, :k] # representative data
indexes = similar(sliced, movie_id, top_n)

print(" ")
print('Recommendations for Movie {0}: \n'.format(
movieData[movieData.movie_id == movie_id].title.values[0]))
for id in indexes + 1:
    print(movieData[movieData.movie_id == id].title.values[0])


# #### <font color="red">Conclusions:</font>
# <font color="green"></font>
# * <font color="green">Here The Recommendation System is Developed for List of N Movies</font>
# * <font color="green">Movie Recommendation System is Developed Based on Collabarating Based Recommendation</font>
# * <font color="green">We Have to Give K Number of Features, Movie Id,Top N as Input and it Recommends Top N Movies as Output</font>
# * <font color="green">These Top N Movies Recommended Using Collabarating Based Filtering Technique with Cosine Similarity and SVD</font>

# In[ ]:





# In[ ]:




