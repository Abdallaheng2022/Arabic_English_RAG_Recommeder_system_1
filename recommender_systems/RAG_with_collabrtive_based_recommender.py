import streamlit as st
import json
import openai
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

st.title("Corrected Sequential Pipeline Arabic and English Book Recommender System")
st.subheader("Stage 1: Vector Search (k=1000) → Stage 2: LLM Refinement")

env_path = os.path.join('.env')
load_dotenv(env_path)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==================== ORIGINAL CODE - METHODOLOGY 1: PURE LLM-BASED ====================
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

def get_recommendation_based_gen_LLM(system_msg, read_books, metadata_json, model_name="gpt-4.1"):
    """
    This will recommend the books based on the description and title of books.
    """
    # create a request to the model such open ai request
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': json.dumps(metadata_json)}
        ],      
        temperature=0.1,
        top_p=0.1,
        response_format={"type": "json_object"},
    )
    recommendations = json.loads(response.choices[0].message.content)['recommended_books']
    print("Recommendations from LLMs", recommendations)
    # exclude the read books from the users and the recommended books which is less than 0.5 score
    recommendations = [book for book in recommendations if book['title'] not in read_books and book['score'] > 0.5]
    recommendations_df = pd.DataFrame(recommendations)
    return recommendations_df

# ==================== ORIGINAL CODE - METHODOLOGY 2: VECTOR SEARCH ====================

def create_vector_database_faiss_by_feature_type(metadata, featureName, embedding_model=OpenAIEmbeddings(), batch_size=900):
    """
    create locally Faiss vector Database.
    """
    try:
        if featureName.strip() == 'Title' or featureName.strip() == 'original_title':
            texts = metadata[featureName].fillna('').astype(str).tolist()[:batch_size]
            # Filter out empty texts
            texts = [text for text in texts if text.strip()]
            if not texts:
                raise ValueError(f"No valid texts found in {featureName} column")
            vec_db = FAISS.from_texts(texts, embedding_model)
        else:
            # Ensure descriptions are strings and not empty
            descriptions = metadata[featureName].fillna('').astype(str)
            docs = [Document(page_content=str(description), metadata=dict(index=idx)) 
                    for idx, description in enumerate(descriptions) 
                    if str(description).strip() and str(description).strip() != 'nan']
            
            if not docs:
                raise ValueError(f"No valid documents found in {featureName} column")
                
            docs = docs[:batch_size]
            vec_db = FAISS.from_documents(docs, embedding_model)
        
        return vec_db
    except Exception as e:
        st.error(f"Error creating vector database: {str(e)}")
        raise e

def get_recommendations_vector_search(read_books, featureName, vec_db, metadata, k=10, featureName2=None):
    """
    It takes the readbooks which users prompted and vector_d and extract top k
    from embeddings that are very similar to the read-books from the metadata.
    """
    # Get the embeddings from the title or description embeddings of books
    # via vector database such as FAISS, etc.
    
    # Convert read_books to string if it's a list
    if isinstance(read_books, list):
        query_text = " ".join(read_books)  # Join list into single string
    else:
        query_text = str(read_books)  # Ensure it's a string
    
    seen = set() 
    relevant_books_with_scores = vec_db.similarity_search_with_score(query_text, k=k*2)
    relevant_books_with_scores = [(doc, score) for doc, score in relevant_books_with_scores 
            if doc.page_content[:50] not in seen and not seen.add(doc.page_content[:50])][:k]
    # Retrieve the books' title or description based on feat name with scores
    recommended_books = []
    
    for book, score in relevant_books_with_scores:
        # st.write(f"{book} and {score}")
        if featureName.strip() == 'Title' or featureName.strip() == 'original_title':
            recommended_books.append({'title': book.page_content, 'score': score})
        else:
            # here we should knows the index of book overview to support to retrieve the right
            # book when create recommended list
            recommended_books.append({
                'title': metadata[featureName2].iloc[book.metadata['index']].strip(),
                featureName: book.page_content,
                'score': score
            })     
    # Exclude readbooks title from the recommended books 
    # to show the similar to the query of the reader.     
    recommended_books = [book for book in recommended_books 
                        if not any(read_book.strip().lower() in book['title'].lower() 
                                 for read_book in (read_books if isinstance(read_books, list) else [read_books]))]
    # prepare dataframe to view the recommended books
    recommendation_df = pd.DataFrame(recommended_books)
    # Reranking descending order from high to low score.
    recommendation_df = recommendation_df.sort_values(by='score', ascending=False)
    return recommendation_df

# ==================== DATA LOADING ====================
ar_book_path = st.secrets["AR_BOOK_PATH"]
en_book = st.secrets["EN_BOOKS"]
en_ratings = st.secrets["EN_RATINGS"]
en_to_read = st.secrets["EN_TO_READ"]
en_tags = st.secrets["EN_TAGS"]

arabic_books_pd = pd.read_csv(ar_book_path)
filtered_arabic_data = arabic_books_pd[
    arabic_books_pd['Description'].notna() &  # Remove NaN/None
    (arabic_books_pd['Description'].astype(str).str.strip() != '') &  # Remove empty strings
    (arabic_books_pd['Description'].astype(str).str.strip() != 'None') &  # Remove 'None'
    (arabic_books_pd['Description'].astype(str).str.strip() != ' None')
]

en_book_pd = pd.read_csv(en_book)
filtered_en_data = en_book_pd[
    en_book_pd['description'].notna() &  # Remove NaN/None
    (en_book_pd['description'].astype(str).str.strip() != '') &  # Remove empty strings
    (en_book_pd['description'].astype(str).str.strip() != 'None') &  # Remove 'None'
    (en_book_pd['description'].astype(str).str.strip() != ' None')
]

en_ratings_pd = pd.read_csv(en_ratings)
en_to_read_pd = pd.read_csv(en_to_read)
en_tags_pd = pd.read_csv(en_tags)

# ==================== CORRECTED SEQUENTIAL PIPELINE FUNCTION ====================
def get_corrected_sequential_pipeline_recommendations(read_books, selected_option, selected_option2, vector_k=1000, final_llm_candidates=100):
    """
    CORRECTED Sequential Pipeline:
    Stage 1: Vector Search gets top 1000 candidates from ALL books (fast & efficient)
    Stage 2: LLM refines these 1000 candidates to final recommendations (intelligent analysis)
    """
    
    # Determine dataset and columns
    if selected_option == 'Arabic Dataset':
        full_metadata = filtered_arabic_data
        col_title = "Title"
        col_desc = "Description"
    else:
        full_metadata = filtered_en_data
        col_title = "original_title"
        col_desc = "description"
    
    st.info(f"🚀 Starting CORRECTED Sequential Pipeline...")
    
    # ===== STAGE 1: Vector Search (Get top 1000 candidates from ALL books) =====
    st.info(f"🔍 Stage 1: Vector search analyzing ALL {len(full_metadata)} books, selecting top {vector_k}...")
    
    try:
        # Apply vector search on the FULL dataset to get top candidates
        if selected_option2 == 'by title':
            vec_db = create_vector_database_faiss_by_feature_type(
                full_metadata, col_title, embedding_model=OpenAIEmbeddings(), batch_size=min(len(full_metadata), 5000)
            )
            stage1_results = get_recommendations_vector_search(
                read_books, col_title, vec_db, 
                full_metadata.drop_duplicates(col_title), k=vector_k
            )
        else:
            vec_db = create_vector_database_faiss_by_feature_type(
                full_metadata, col_desc, embedding_model=OpenAIEmbeddings(), batch_size=min(len(full_metadata), 5000)
            )
            stage1_results = get_recommendations_vector_search(
                read_books, col_desc, vec_db, 
                full_metadata.drop_duplicates(col_title), 
                k=vector_k, featureName2=col_title
            )
        
        stage1_count = len(stage1_results)
        st.success(f"✅ Stage 1 Complete: Vector search selected {stage1_count} candidate books from {len(full_metadata)} total books")
        
        if stage1_count == 0:
            st.warning("No candidates from Stage 1. Ending pipeline.")
            return pd.DataFrame(), pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error in Stage 1: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    # ===== STAGE 2: LLM Refinement (Analyze vector candidates intelligently) =====
    st.info(f"🤖 Stage 2: LLM intelligent analysis of top {stage1_count} vector candidates...")
    
    try:
        # Prepare the vector candidates for LLM analysis
        # We need to convert the stage1_results back to the format expected by LLM
        
        # Get the full book information for LLM candidates
        stage1_titles = stage1_results['title'].tolist()
        
        # Create metadata subset for LLM analysis
        stage1_metadata_subset = full_metadata[full_metadata[col_title].isin(stage1_titles)]
        
        # Prepare data for Stage 2 - send vector candidates to LLM for intelligent refinement
        metadata_json = {
            'user_read_books': read_books,
            'all_books': stage1_metadata_subset.values.tolist()  # Send vector candidates to LLM
        }
        
        # Get LLM refinement of vector candidates
        stage2_results = get_recommendation_based_gen_LLM(
            system_msg, read_books, metadata_json, model_name="gpt-4.1"
        )
        
        stage2_count = len(stage2_results)
        st.success(f"✅ Stage 2 Complete: LLM refined {stage1_count} candidates to {stage2_count} final recommendations")
        
        # Add vector similarity scores to final results for comparison
        if not stage2_results.empty and not stage1_results.empty:
            # Create mapping from stage1 vector scores
            vector_score_map = dict(zip(stage1_results['title'], stage1_results['score']))
            
            # Add vector similarity scores to final results
            stage2_results['vector_similarity_score'] = stage2_results['title'].map(vector_score_map)
            
            # Reorder columns
            cols = ['title', 'score', 'vector_similarity_score', 'justification']
            stage2_results = stage2_results[[col for col in cols if col in stage2_results.columns]]
        
    except Exception as e:
        st.error(f"Error in Stage 2: {e}")
        return stage1_results, pd.DataFrame()
    
    return stage1_results, stage2_results

# ==================== STREAMLIT UI ====================
st.markdown("---")

# Input section
user_input = st.text_area(
    "📚 Please write your read books as Json file:",
    value='["The Great Gatsby", "1984", "To Kill a Mockingbird"]'
)

# Configuration
col1, col2, col3, col4 = st.columns(4)

with col1:
    options = ['Arabic Dataset', 'English Dataset']
    selected_option = st.selectbox('Select Dataset Language:', options)

with col2:
    options2 = ['by title', 'by description']
    selected_option2 = st.selectbox('Vector Search Method:', options2)

with col3:
    vector_k = st.selectbox(
        'Stage 1 Vector K:', 
        [500, 1000, 1500, 2000],
        index=1,  # Default to 1000
        help="Number of candidates from vector search to send to LLM"
    )

with col4:
    pipeline_type = st.selectbox(
        'Pipeline Type:',
        ['Corrected (Vector→LLM)', 'Original (LLM→Vector)'],
        help="Choose between corrected efficient pipeline or original method"
    )

# Pipeline explanation based on selection
if pipeline_type == 'Corrected (Vector→LLM)':
    st.success("""
    ✅ **CORRECTED Sequential Pipeline Process:**
    1. **Stage 1 (Vector Search)**: Fast scan of ALL books, select top {} most similar
    2. **Stage 2 (LLM Refinement)**: Intelligent analysis of {} candidates for final recommendations
    3. **Result**: Efficient + Intelligent = Best recommendations
    """.format(vector_k, vector_k))
else:
    st.warning("""
    ⚠️ **ORIGINAL Sequential Pipeline Process:**
    1. **Stage 1 (LLM Analysis)**: Slow analysis of {} books
    2. **Stage 2 (Vector Search)**: Apply vector search on LLM results
    3. **Result**: Less efficient but shows original approach
    """.format(vector_k))

# Main button
if st.button("🚀 Run Sequential Pipeline", type="primary"):
    try:
        read_books = eval(user_input)
        
        with st.spinner("Running sequential pipeline..."):
            
            if pipeline_type == 'Corrected (Vector→LLM)':
                # Run CORRECTED pipeline: Vector first, then LLM
                stage1_results, stage2_results = get_corrected_sequential_pipeline_recommendations(
                    read_books, selected_option, selected_option2, vector_k
                )
                stage1_name = "Vector Search Candidates"
                stage2_name = "LLM Refined Results"
            else:
                # Run ORIGINAL pipeline: LLM first, then Vector (from your original code)
                # This is kept for comparison purposes
                st.warning("Running original (less efficient) pipeline for comparison...")
                
                # Determine dataset and columns
                if selected_option == 'Arabic Dataset':
                    full_metadata = filtered_arabic_data
                    col_title = "Title"
                    col_desc = "Description"
                else:
                    full_metadata = filtered_en_data
                    col_title = "original_title"
                    col_desc = "description"
                
                # Stage 1: LLM first
                metadata_json = {
                    'user_read_books': read_books,
                    'all_books': full_metadata.values.tolist()[:vector_k]
                }
                stage1_results = get_recommendation_based_gen_LLM(
                    system_msg, read_books, metadata_json, model_name="gpt-4.1"
                )
                
                # Stage 2: Vector search on LLM results
                if not stage1_results.empty:
                    stage1_titles = stage1_results['title'].tolist()
                    stage2_metadata = full_metadata[full_metadata[col_title].isin(stage1_titles)]
                    
                    if selected_option2 == 'by title':
                        vec_db = create_vector_database_faiss_by_feature_type(
                            stage2_metadata, col_title, embedding_model=OpenAIEmbeddings()
                        )
                        stage2_results = get_recommendations_vector_search(
                            read_books, col_title, vec_db, 
                            stage2_metadata.drop_duplicates(col_title), k=len(stage2_metadata)
                        )
                    else:
                        vec_db = create_vector_database_faiss_by_feature_type(
                            stage2_metadata, col_desc, embedding_model=OpenAIEmbeddings()
                        )
                        stage2_results = get_recommendations_vector_search(
                            read_books, col_desc, vec_db, 
                            stage2_metadata.drop_duplicates(col_title), 
                            k=len(stage2_metadata), featureName2=col_title
                        )
                else:
                    stage2_results = pd.DataFrame()
                
                stage1_name = "LLM Analysis Results"
                stage2_name = "Vector Refined Results"
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs([
                "🎯 Final Results (Stage 2)", 
                f"📊 Stage 1 ({stage1_name})",
                "📈 Pipeline Analysis"
            ])
            
            with tab1:
                st.subheader("🎯 Final Refined Recommendations")
                if not stage2_results.empty:
                    st.success(f"Found {len(stage2_results)} final recommendations")
                    st.dataframe(stage2_results, use_container_width=True)
                    
                    # Download final results
                    csv = stage2_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Final Recommendations",
                        data=csv,
                        file_name=f"pipeline_final_recommendations_{pipeline_type.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('→', '_to_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No final recommendations generated from the pipeline.")
            
            with tab2:
                st.subheader(f"📊 Stage 1: {stage1_name}")
                if not stage1_results.empty:
                    st.info(f"Stage 1 generated {len(stage1_results)} candidates")
                    st.dataframe(stage1_results, use_container_width=True)
                    
                    # Download stage 1 results
                    csv = stage1_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Stage 1 Results",
                        data=csv,
                        file_name=f"pipeline_stage1_{pipeline_type.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('→', '_to_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No candidates generated from Stage 1.")
            
            with tab3:
                st.subheader("📈 Pipeline Performance Analysis")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Books Read", len(read_books))
                with col2:
                    st.metric("Stage 1 Candidates", len(stage1_results))
                with col3:
                    st.metric("Final Recommendations", len(stage2_results))
                with col4:
                    if selected_option == 'Arabic Dataset':
                        total_books = len(filtered_arabic_data)
                    else:
                        total_books = len(filtered_en_data)
                    st.metric("Total Books in Dataset", total_books)
                
                # Pipeline efficiency analysis
                st.markdown("---")
                st.markdown("### 🔄 Pipeline Efficiency")
                
                if pipeline_type == 'Corrected (Vector→LLM)':
                    st.success("""
                    **✅ Corrected Pipeline Benefits:**
                    - **Fast Stage 1**: Vector search efficiently scans ALL books
                    - **Smart Stage 2**: LLM provides intelligent analysis of best candidates
                    - **Cost Efficient**: LLM only processes pre-filtered high-quality candidates
                    - **Comprehensive**: Considers entire dataset in initial screening
                    """)
                else:
                    st.warning("""
                    **⚠️ Original Pipeline Limitations:**
                    - **Slow Stage 1**: LLM processes limited batch, may miss good books
                    - **Limited Coverage**: Only analyzes subset of total books
                    - **Higher Cost**: LLM processes many irrelevant books
                    - **Inefficient**: Vector search applied to small pre-filtered set
                    """)
                
                # Performance comparison
                if len(stage1_results) > 0:
                    total_books = len(filtered_arabic_data) if selected_option == 'Arabic Dataset' else len(filtered_en_data)
                    coverage_rate = (vector_k / total_books) * 100 if pipeline_type == 'Original (LLM→Vector)' else 100
                    
                    st.markdown(f"""
                    **📊 Performance Metrics:**
                    - **Dataset Coverage**: {coverage_rate:.1f}% of books analyzed
                    - **Stage 1 → Stage 2 Efficiency**: {len(stage2_results)}/{len(stage1_results)} = {(len(stage2_results)/len(stage1_results)*100) if len(stage1_results) > 0 else 0:.1f}%
                    - **Pipeline Type**: {pipeline_type}
                    """)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your input format and try again.")

# Information section
st.markdown("---")
st.subheader("ℹ️ Pipeline Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ **CORRECTED Pipeline (Recommended)**
    **🔍 Stage 1: Vector Search (Fast)**
    - Scans ALL books in dataset
    - Uses embeddings for semantic similarity
    - Selects top 1000 most relevant candidates
    - Fast and comprehensive coverage
    
    **🤖 Stage 2: LLM Analysis (Smart)**  
    - Analyzes only the top vector candidates
    - Provides intelligent reasoning and scoring
    - Cost-effective (processes fewer books)
    - High-quality final recommendations
    """)

with col2:
    st.markdown("""
    ### ⚠️ **Original Pipeline (Less Efficient)**
    **🤖 Stage 1: LLM Analysis (Slow)**
    - Limited to small batch (1000 books max)
    - May miss relevant books not in batch
    - Expensive to process many books
    - Sequential processing bottleneck
    
    **🔍 Stage 2: Vector Search (Limited)**
    - Only searches within LLM results
    - Reduced candidate pool
    - Less comprehensive coverage
    - Redundant similarity analysis
    """)

# Dataset info
with st.expander("📊 Dataset Information"):
    st.write(f"**Arabic Books**: {len(filtered_arabic_data):,} books with descriptions")
    st.write(f"**English Books**: {len(filtered_en_data):,} books with descriptions")
    st.write("**Recommendation**: Use Vector→LLM pipeline for better coverage and efficiency")