#Book recommendation systems for Arabic and English books
import openai
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
# Configure page
st.title("Book Recommeder System For Arabic and English")
env_path = os.path.join('.env')
load_dotenv(env_path)

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


#The idea is to convert the books' overview and the favourited books which the users prefer.

def get_embedding(text,model="text-embedding-ada-002"):
  """
   It takes the textual content and converts into embedded verison 
   via pre-trained models or LLMs.
  """
  text = text.replace("\n","Empty")
  return client.embeddings.create(input=[text],model=model).data[0].embedding
  
ar_book_path = st.secrets["AR_BOOK_PATH"]
en_book = st.secrets["EN_BOOKS"]
en_ratings= st.secrets["EN_RATINGS"]
en_to_read = st.secrets["EN_TO_READ"]
en_tags =  st.secrets["EN_TAGS"]
arabic_books_pd=pd.read_csv(ar_book_path)

filtered_arabic_data=arabic_books_pd[ arabic_books_pd['Description'].notna() &  # Remove NaN/None
    (arabic_books_pd['Description'].astype(str).str.strip() != '') &  # Remove empty strings
    (arabic_books_pd['Description'].astype(str).str.strip() != 'None') &  # Remove 'None'
    (arabic_books_pd['Description'].astype(str).str.strip() != ' None')]
en_book_pd = pd.read_csv(en_book)

filtered_en_data=en_book_pd[ en_book_pd['description'].notna() &  # Remove NaN/None
    (en_book_pd['description'].astype(str).str.strip() != '') &  # Remove empty strings
    (en_book_pd['description'].astype(str).str.strip() != 'None') &  # Remove 'None'
    (en_book_pd['description'].astype(str).str.strip() != ' None')]
en_ratings_pd = pd.read_csv(en_ratings)
en_to_read_pd = pd.read_csv(en_to_read)
en_tags_pd = pd.read_csv(en_tags)
#st.sidebar.dataframe(en_to_read_pd)
              
def LLM_descriptor(query):
  """ 
     It takes the name of elements and describes the fields.
  """
  response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful search assistant.with only json"},
            {"role": "user", "content": f"Search for information about books overwiews add one for title and another section detailed overview arabic only for each book: {query} include all books in the dataframe"}
        ], response_format={"type": "json_object"}
        
    )
  return response.choices[0].message.content
  
#json_string=LLM_descriptor(arabic_books_pd["Title"].to_list()[:1000])
# Download button
#st.download_button( label="Download JSON File", data=json_string, file_name="data.json", mime="application/json")



def calculate_books_features(metadata,column_overview):
  """
    It converts the textual content into embedding numbers.
  """
  embeddings = []
  for description in metadata[column_overview].to_list():
        try:
            embeddings.append(get_embedding(description))
        except Exception as e:
             st.sidebar.write(f"Error {e} in books arabic description:",description)    
  return np.array(embeddings)           
#The best version is to make 1000 api calls or accmulates the text and make one call by 1000 books overviews

def get_embedding_enhanced(texts,model="text-embedding-ada-002"):
  """
   It takes the textual content and converts into embedded verison 
   via pre-trained models or LLMs.
  """
  embeddings=np.array([v.embedding for v in client.embeddings.create(input=texts,model=model).data])
  return embeddings
def calculate_books_features_enhanced(metadata,column_overview,batch_sample_size):
  """
    It converts the textual content into embedding numbers.
  """
  embeddings = []
  books_overviews=metadata[column_overview][0:batch_sample_size].values.tolist()
  embeddings=get_embedding_enhanced(books_overviews)      
  return embeddings
#This part is offline why because we will build up the embedding for all books 
# then used this offline data to query the online data 
#arabic_desc_embd=calculate_books_features_enhanced(filtered_arabic_data.drop_duplicates(),"Description",1000)
#en_desc_embd = calculate_books_features_enhanced(filtered_en_data.drop_duplicates(),"description",1000)
#np.save('arabic_desc_embd.npy', arabic_desc_embd)
#np.save('en_desc_embd.npy', en_desc_embd)

def calculate_users_features(read_books,metadata,col_title,col_desc,embd_len=100, embd_res_type="All"):
   """
    It creates embeddings for books' description which the user watched 
    then caculate similarity    
   """
   read_books_feat = ""
   read_embeddings = []
   for book in read_books:
      if book in metadata[col_title].values:
              #st.sidebar.write(metadata.loc[metadata['original_title']== book,col_desc].iloc[0])
              description= metadata.loc[metadata[col_title]== book,col_desc].iloc[0]
              read_books_feat+=description
   if len(read_books_feat)==0:
        return None,False 
   else: 
        read_books_embd=get_embedding_enhanced([read_books_feat])
        if embd_res_type == "All":
             return read_books_embd,True
        elif embd_res_type == "mean":
             return np.mean(read_books_embd,axis=0)
 

"""
 To Design get recommendation:
 1- Read-books.
 2- Books_overiew_embeddings.
 3- Data.
 4- Cosine similarity.
 5- Sort.
 6- Exclude the all read-books and recommend non-readbooks.
 7- recommend top k. 
 8- add in the structure to view.
"""

def get_recommendations(read_books,col_title,col_desc,books_desc_embd,metadata):
    """
    It takes the readbooks which users prompted and embddings and extract top k
    from embeddings that are very similar to the read-books from the metadata.
    """
    embd_len = books_desc_embd.shape[1]
    user_profile,status=calculate_users_features(read_books,metadata,col_title,col_desc,"mean") 
    #cosine similarity and sort 
    #st.sidebar.write(user_profile)
    #st.sidebar.write(books_desc_ar_embd)
    cosine_sim=cosine_similarity(user_profile,books_desc_embd)
    #sort
    sim_scores = list(enumerate(cosine_sim[0]))
    #Sorted Descending order
    sim_scores = sorted(sim_scores,key=lambda x: x[1], reverse=True)
    #Exclude books already read
    title_repeated = []
    read_indices = []
    for sim_score in sim_scores:
          if (metadata[col_title].iloc[sim_score[0]] not in read_books) and (metadata[col_title].iloc[sim_score[0]] not in title_repeated):
                      read_indices.append(sim_score[0])
                      title_repeated.append(metadata[col_title].iloc[sim_score[0]])
    #Recommend top 5 books
    top_recommendations = metadata[col_title].iloc[read_indices][:5]
    # prepare dataframe to view the recomended books
    recommendations = [{'title':title,"score":score[1]} for title, score in zip(top_recommendations,sim_scores)]
    recommendation_df= pd.DataFrame(recommendations)
    return recommendation_df

user_input=st.text_area("Please write your read books as Json file:")
options = ['Arabic Dataset', 'English Dataset']
selected_option = st.selectbox('Select Dataset Language:', options)
if st.button("get recommendation"):
      read_books=eval(user_input)
      if selected_option=='Arabic Dataset':
          col_title = "Title"
          col_desc = "Description"
          books_desc_ar_embd = np.load("/home/abdo/Downloads/LLMs_apps_gihub/recommender_systems/ar_books_data/arabic_desc_embd.npy")
          recommendation_df =get_recommendations(read_books,col_title,col_desc,books_desc_ar_embd,filtered_arabic_data)
      else:
          col_title = "original_title"
          col_desc = "description"
          books_desc_en_embd = np.load("/home/abdo/Downloads/LLMs_apps_gihub/recommender_systems/en_books_data/en_desc_embd.npy")  
          recommendation_df =get_recommendations(read_books,col_title,col_desc,books_desc_en_embd,filtered_en_data)
               
      st.dataframe(recommendation_df)