
import streamlit as st
import json
import openai
import os
from dotenv import load_dotenv
import pandas as pd



st.title("Arabic and English Book Recommeder System pure-LLM-Based")
env_path = os.path.join('.env')
load_dotenv(env_path)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
system_msg = """
              #Task 
              Act as a Book Recommender system. I will give you json like theat:
              # Input
              {  "user_read_books": list of strings representing all books that user has read,
                 "all books": list of strings representing all the available books in the datasset,
              
              }
              #Output:
               I want you to return to me a json file like that:
               {
                    'recommended_books': [ 
                    
                    {
                                  
                                'title': string representing the title of the recommended book.
                                'score': string representing the scotr of the recomeneded book. from 0 to 1, ranked based on the relevance to the user. 
                                'justification': string representing the reason provided by the LLM why this book matches the user's book interest. 
                                It will help debugging and improve the prompt. limit your justification of why you picked this book into 2 sentences maximum. 
                    },                
                                  ]
               }
 
               
             #Example Input: 
             { 
                 
                 'user_read_books': ['Book X','Book Y'],
                 'all_books': ['Book 1'....,'Book N'],               
             
             }

             #Example Output: 

              {
              
                'recommended_books':
                [
                            {
                                         
                            'title':'Book A',
                            'score': 0.8,
                            'justification': Reason provided by the LLM why this book matches the user's book interest. 
                                It will help debugging and improve the prompt
                            
                            
                            },
                            {
                            
                            'title':'Book P',
                            'score': 0.6,
                            'justification': Reason provided by the LLM why this book matches the user's book interest. 
                                It will help debugging and improve the prompt
                            
                            },
                
                ]


                 
              }


             #RULES
              - Be specific with your recommendation.
              - If the user is already read a book don't recommend it, or give it score = 0.
              - Pick recommendation from the given set of all available books 'all_books'.
              - Books that are sequels and related contextually to the user has read should be recommended with a higher score than other books.
              - Provide your answer in the json fromat only with no extra unformatted text so that I can parse it in code.
              - Don't enclose your answer in ``` json quotes it will hinder the parse processing in code.             
             """

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


def get_recommendation_based_gen_LLM(system_msg,read_books,metadata_json,model_name="gpt-4.1"):
    """
      this will recommend the books based on the description and title of books.
    """
     # create a request to the model such open ai request
    response= client.chat.completions.create(model=model_name,
                                       messages= [{'role':'system','content':system_msg},{'role':'user','content':json.dumps(metadata_json)}],      
                                       temperature = 0.1,
                                       top_p=0.1,
                                       response_format={"type":"json_object"},
                                       )
    recommendations = json.loads(response.choices[0].message.content)['recommended_books']
    print("Recommendations from LLMs",recommendations)
    #exclude the read books from the users and the recommended books which is less than 0.5 score
    recommendations=[book for book in recommendations if book['title'] not in read_books and book['score'] > 0.5]
    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df

user_input=st.text_area("Please write your read books as Json file:")
options = ['Arabic Dataset', 'English Dataset']
selected_option = st.selectbox('Select Dataset Language:', options)

batch_size = 50
if st.button("get recommendation"):
    read_books=eval(user_input)
    if selected_option=='Arabic Dataset':
        metadata_json = {'user_read_books':read_books,'all_books':filtered_arabic_data.values.tolist()[:batch_size]}
        recommendations_df=get_recommendation_based_gen_LLM(system_msg,read_books,metadata_json,model_name="gpt-4.1")
    else:
        metadata_json = {'user_read_books':read_books,'all_books':filtered_en_data.values.tolist()[:batch_size]}
        recommendations_df=get_recommendation_based_gen_LLM(system_msg,read_books,metadata_json,model_name="gpt-4.1")
    st.dataframe(recommendations_df)         