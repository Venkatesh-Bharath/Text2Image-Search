import streamlit as st
from PIL import Image
import numpy as np
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
import faiss
from sentence_transformers import SentenceTransformer

# Set Google API key
os.environ['GOOGLE_API_KEY'] = "Enter your Key Here"

# Load pre-trained BERT-based model for text embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize GeminiPro model
gemini_pro = GeminiMultiModal(model_name="models/gemini-pro-vision")

# Placeholder database to store image features and descriptions
vector_database = {}

# Placeholder function to store image features and descriptions in the vector database
def store_image_data(image_data, descriptions):
    global vector_database
    
    # Assuming each image_data and description pair corresponds to one image
    for idx, (img_data, desc) in enumerate(zip(image_data, descriptions)):
        # Store image features and description in the database
        vector_database[idx] = {'image_data': img_data, 'description': desc}

# Placeholder function to perform similarity search in the vector database
def search_images_by_text(query_keyword):
    global vector_database
    
    relevant_images = []
    
    # Iterate over the stored images and their descriptions
    for idx, data in vector_database.items():
        desc = data['description']
        if query_keyword.lower() in desc.lower():
            relevant_images.append((desc, f'./stored_images/image{idx}.jpg'))
    
    return relevant_images

# Streamlit UI
st.title("Search images by text")

# Initialize ChatGoogleGenerativeAI instance
llm = ChatGoogleGenerativeAI(model='gemini-pro-vision')

# Multi-file uploader for images
uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Display already stored images in a sidebar
st.sidebar.subheader("Stored Images")
stored_images_path = "./stored_images"
# Display all images and their descriptions from the vector database
def display():
    for idx, data in vector_database.items():
        st.sidebar.subheader(f"Image {idx + 1}")
        st.sidebar.image(data['image_data'],use_column_width=True)

    
def retrieve_image_descriptions():
    global vector_database
    descriptions = []
    for data in vector_database.items():
        descriptions.append(data['description'])
    return descriptions

stored_images = os.listdir(stored_images_path)
image_descriptions = retrieve_image_descriptions()
for img_path, desc in zip(stored_images, image_descriptions):
    img = Image.open(os.path.join(stored_images_path, img_path))
    st.sidebar.image(img, caption=desc, use_column_width=True)
if uploaded_files:
    image_data = []
    descriptions = []
    
    # Display uploaded images and collect descriptions
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        # Generate description for the uploaded image using GeminiPro
        message = HumanMessage(content=[{'type':'text', 'text':"Generate description of Image"}, {'type':'image_url', 'image_url':image}])
        response = llm.invoke([message])
        image_data.append(np.array(image))
        descriptions.append(response.content)
        # Save the image
        image.save(f'./stored_images/image{idx}.jpg')
    
    # Store image data and descriptions in the vector database
    if image_data and descriptions:
        store_image_data(image_data, descriptions)
        display()
        st.success("Image data stored successfully!")

# Text input for keyword search
query_keyword = st.text_input("Enter keyword to search")

# Process the keyword search
if query_keyword:
    # Perform keyword search in image descriptions
    relevant_images = search_images_by_text(query_keyword)
    
    # Display relevant images and descriptions
    if relevant_images:
        st.subheader("Relevant Images:")
        for description, image_path in relevant_images:
            st.image(Image.open(image_path), caption=description, use_column_width=True)
    else:
        st.write("No relevant images found.")
