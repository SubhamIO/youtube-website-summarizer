import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from yt_dlp import YoutubeDL
from dotenv import load_dotenv
from langchain.schema import Document

# Load API key from .env file
load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = st.secrets["GROQ_API_KEY"]


# Streamlit app configuration
st.set_page_config(page_title="LangChain Enhanced Summarizer", page_icon="🌟")
st.title("YouTube or Website Summarizer")
st.write("Welcome! Summarize content from YouTube videos or websites in a more detailed manner.")

# Sidebar content
st.sidebar.title("About This App")
st.sidebar.info(
    "This app uses LangChain and the Llama model from Groq API to provide detailed summaries. "
    "Simply enter a URL (YouTube or website) and get a concise summary!"
)

# Instructions
st.header("How to Use:")
st.write("1. Enter the URL of a YouTube video or website you wish to summarize.")
st.write("2. Click **Summarize** to get a detailed summary.")
st.write("3. Enjoy the results!")

# Text input for URL
st.subheader("Enter the URL:")
generic_url = st.text_input("URL", label_visibility="collapsed", placeholder="https://example.com")

# Gemma Model using Groq API
llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a detailed summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def load_youtube_content(url):
    """Extract YouTube content as text using yt_dlp."""
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Video")
        description = info.get("description", "No description available.")
        return f"{title}\n\n{description}"

# Summarize button and output
if st.button("Summarize"):
    if not generic_url.strip():
        st.error("Please provide a URL to proceed.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("Processing..."):
                # Load content from URL
                if "youtube.com" in generic_url:
                    # Load YouTube content as a string
                    text_content = load_youtube_content(generic_url)
                    # Wrap the string into a Document for compatibility with LangChain
                    docs = [Document(page_content=text_content)]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                        )
                    docs = loader.load()

                # Summarize using LangChain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.subheader("Detailed Summary:")
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception occurred: {e}")
