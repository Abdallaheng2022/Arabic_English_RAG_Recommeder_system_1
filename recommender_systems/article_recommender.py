import streamlit as st
import os 
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import openai
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,manhattan_distances # pyright: ignore[reportMissingModuleSource]
import numpy as np
import pandas as pd
st.title("Article Recommeder System For Arabic and English")
env_path = os.path.join('.env')
load_dotenv(env_path)

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
articles_pd=pd.read_csv("articles.csv",encoding="latin1")
#offline part 


def distances_from_embeddings(
    query_embedding: list[float] | np.ndarray,
    embeddings: list[list[float]] | np.ndarray,
    distance_metric: str = "cosine"
) -> np.ndarray:
    """
    Calculate distances between a query embedding and a list of embeddings.
    
    Args:
        query_embedding: The source embedding vector
        embeddings: List of embedding vectors to compare against
        distance_metric: Distance metric to use ("cosine", "euclidean", "manhattan", "dot_product")
    
    Returns:
        Array of distances between query_embedding and each embedding in embeddings
    """
    # Convert to numpy arrays for easier manipulation
    query_embedding=query_embedding.reshape(1,-1)   
    embeddings=embeddings.reshape(embeddings.shape[0],1)      
    print(query_embedding.shape)
    if distance_metric == "cosine":
        # Method 1: Using scipy.spatial.distance.cosine (returns cosine distance)
        distance = cosine_similarity(query_embedding, embeddings)
          
    
    elif distance_metric == "euclidean":
         distance = euclidean_distances(query_embedding, embeddings)
          
    
    elif distance_metric == "manhattan":
        # Manhattan distance (L1 norm)
        distance = manhattan_distances(query_embedding, embeddings)
    
    elif distance_metric == "dot_product":
        # Negative dot product (to make smaller values = more similar)
        distance = -np.dot(query_embedding, embeddings)
        
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    return distance


def indices_of_nearest_neighbors_excluding_self(
    distances: np.ndarray,
    query_index: int = None,
    k: int = None
) -> np.ndarray:
    """
    Return indices sorted by distance, optionally excluding the query index.
    
    Args:
        distances: Array of distances
        query_index: Index to exclude (usually the source embedding)
        k: Number of nearest neighbors to return
    
    Returns:
        Array of indices sorted by distance, excluding query_index if specified
    """
    sorted_indices = np.argsort(distances)
    
    # Remove query_index if specified
    if query_index is not None:
        sorted_indices = sorted_indices[sorted_indices != query_index]
    
    # Return top k if specified
    if k is not None:
        return sorted_indices[:k]
    
    return sorted_indices


def get_embedding_enhanced(texts,model="text-embedding-ada-002"):
  """
   It takes the textual content and converts into embedded verison 
   via pre-trained models or LLMs.
  """
  embeddings=np.array([v.embedding for v in client.embeddings.create(input=texts,model=model).data])
  return embeddings



def print_recommendations_from_strings(
    strings: list[str],
    index_of_source_string: int,
    k_nearest_neighbors: int = 1,
    model="text-embedding-ada-002",
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""
    # get embeddings for all strings
    new_strings=" ".join(str(strings.values))
    embeddings = get_embedding_enhanced(new_strings) 
    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]
    # get distances between the source embedding and other embeddings (function from embeddings_utils)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    st.write(distances)
    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_excluding_self(distances,index_of_source_string)
    print()
    # print out source 
    query_string = strings.iloc[index_of_source_string].values[0]
    #print(query_string)
    st.write(f"Source string: {query_string}")
    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any string that are identical matches to the starting string
        if query_string == strings.iloc[i].values[0]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        st.write(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors

text_input = st.text_area("Please write the number of article do you like we have 16 article")
get_recommendation  = st.button("Get Recommendation")
if get_recommendation:
    print_recommendations_from_strings(articles_pd[:15],int(text_input)-1)