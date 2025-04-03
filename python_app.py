import validators
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from yt_dlp import YoutubeDL
from langchain.schema import Document

# Load environment variables
load_dotenv()

def load_youtube_content(url):
    """Extract YouTube content as text using yt_dlp."""
    ydl_opts = {'format': 'bestaudio/best', 'quiet': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Video")
        description = info.get("description", "No description available.")
        return f"{title}\n\n{description}"

def main():
    print("🦜 LangChain: Summarize Text From YT or Website")
    
    groq_api_key = input("Enter your Groq API Key: ")
    if not groq_api_key.strip():
        print("API Key is required!")
        return
    
    generic_url = input("Enter the URL (YouTube or Website): ")
    if not generic_url.strip() or not validators.url(generic_url):
        print("Please enter a valid URL (YouTube or Website).")
        return
    
    llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)

    prompt_template = """
    Provide a detailed summary of the following content in 300 words:
    Content: {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    try:
        print("Processing, please wait...")
        
        # Load content from URL
        if "youtube.com" in generic_url or "youtu.be" in generic_url:
            text_content = load_youtube_content(generic_url)
            docs = [Document(page_content=text_content)]
        else:
            loader = UnstructuredURLLoader(
                urls=[generic_url],
                ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
            )
            docs = loader.load()

        # Summarization Chain
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        output_summary = chain.invoke(docs)
        
        print("\nSummary:")
        print(output_summary)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
