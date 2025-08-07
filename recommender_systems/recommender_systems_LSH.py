#Book recommendation systems for Arabic and English books
import openai
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

# Configure page
st.title("Arabic and English Book Recommeder System collborative filterting and LLM-based and Vector search and database")
env_path = os.path.join('.env')
load_dotenv(env_path)

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

#here we will use vector database to keep the data in the vector stores and not to scroll or iterate over all samples of data between query 
# and to get the embeddings and check which one in each sample is similar then sort. Alternatively, we will 
#The idea is to convert the books' overview and the favourited books which the users prefer.

def create_vector_database_faiss_by_feature_type(metadata,featureName,embedding_model=OpenAIEmbeddings(),batch_size=900):
     """
      create locally Faiss vector Database.
     """
    
     if featureName.strip()=='Title' or featureName.strip()=='original_title':
          texts = metadata[featureName].tolist()
          vec_db=FAISS.from_texts(texts,embedding_model)
     else:
          docs=[Document(page_content=description, metadata= dict(index=idx)) for idx,description in enumerate(metadata[featureName])]
          vec_db= FAISS.from_documents(docs[:batch_size],embedding_model)     
     return vec_db

  
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
            
  
#json_string=LLM_descriptor(arabic_books_pd["Title"].to_list()[:1000])
# Download button
#st.download_button( label="Download JSON File", data=json_string, file_name="data.json", mime="application/json")



#This part is offline why because we will build up the embedding for all books 
# then used this offline data to query the online data 
#arabic_desc_embd=calculate_books_features_enhanced(filtered_arabic_data.drop_duplicates(),"Description",1000)
#en_desc_embd = calculate_books_features_enhanced(filtered_en_data.drop_duplicates(),"description",1000)
#np.save('arabic_desc_embd.npy', arabic_desc_embd)
#np.save('en_desc_embd.npy', en_desc_embd)


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

def get_recommendations_vector_search(read_books,featureName,vec_db,metadata,k=10,featureName2=None):
    """
    It takes the readbooks which users prompted and vector_d and extract top k
    from embeddings that are very similar to the read-books from the metadata.
    """
    #Get the embeddings from the title or descritption embeddings of books
    #via vector database such as FAISS, etc.
    seen = set() 
    relevant_books_with_scores=vec_db.similarity_search_with_score(read_books,k=k*2)
    relevant_books_with_scores=[(doc, score) for doc, score in relevant_books_with_scores 
            if doc.page_content[:50] not in seen and not seen.add(doc.page_content[:50])][:k]
    # Retrieve the books' title or description based on feat name with scores
    recommended_books = []
    
    for book, score in relevant_books_with_scores:
          #st.write(f"{book} and {score}")
          if featureName.strip()=='Title' or featureName.strip()=='original_title':
              recommended_books.append({'title':book.page_content,'score':score})
          else:
              #here we should knows the index of book overview to support to retrieve the right
              # book when create recomended list
              recommended_books.append({'title':metadata[featureName2].iloc[book.metadata['index']].strip(),featureName:book.page_content,'score':score})     
    # Exclude readbooks title from the recommended books 
    # to show the similar to the query of the reader.     
    recommended_books = [book for book in recommended_books if book['title'] not in read_books.strip()]
    # prepare dataframe to view the recomended books
    recommendation_df= pd.DataFrame(recommended_books)
    #Reranking descending order from high to low score.
    recommendation_df = recommendation_df.sort_values(by='score',ascending=False)
    return recommendation_df

user_input=st.text_area("Please write your read books as Json file:")
options = ['Arabic Dataset', 'English Dataset']
selected_option = st.selectbox('Select Dataset Language:', options)
options2 = ['by title', 'by description']
selected_option2 = st.selectbox('Sort and Recommend by', options2)
if st.button("get recommendation"):
      read_books=eval(user_input)
      if selected_option=='Arabic Dataset':
          col_title = "Title"
          col_desc = "Description"
          books_desc_ar_embd = np.load("/home/abdo/Downloads/LLMs_apps_gihub/recommender_systems/ar_books_data/arabic_desc_embd.npy")
          if selected_option2 == 'by title':
               vec_db=create_vector_database_faiss_by_feature_type(filtered_arabic_data,col_title,embedding_model=OpenAIEmbeddings())
               recommendation_df =get_recommendations_vector_search(read_books,col_title,vec_db,filtered_arabic_data.drop_duplicates(col_title))
          else: 
               vec_db=create_vector_database_faiss_by_feature_type(filtered_arabic_data,col_desc,embedding_model=OpenAIEmbeddings())    
               recommendation_df =get_recommendations_vector_search(read_books,col_desc,vec_db,filtered_arabic_data.drop_duplicates(col_title),k=10,featureName2=col_title)
      else:
          col_title = "original_title"
          col_desc = "description"
          if selected_option2 == 'by title':
               vec_db=create_vector_database_faiss_by_feature_type(filtered_en_data,col_title,embedding_model=OpenAIEmbeddings())
               recommendation_df =get_recommendations_vector_search(read_books,col_title,vec_db,filtered_en_data.drop_duplicates(col_title))
          else:     
               vec_db=create_vector_database_faiss_by_feature_type(filtered_en_data,col_desc,embedding_model=OpenAIEmbeddings())  
               recommendation_df =get_recommendations_vector_search(read_books,col_desc,vec_db,filtered_en_data.drop_duplicates(col_title),k=10,featureName2=col_title)
               
      st.dataframe(recommendation_df)