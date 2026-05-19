from flask import Flask, render_template, request, jsonify
import os

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq

app = Flask(__name__)

# -----------------------------------
# Load Embeddings
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# Load Existing FAISS DB (Optional)
# -----------------------------------

try:

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

except:

    db = None

# -----------------------------------
# Load Groq LLM
# -----------------------------------

import os

llm = ChatGroq(

    groq_api_key=os.getenv("GROQ_API_KEY"),

    model_name="llama-3.1-8b-instant",

    temperature=0.1
)

# -----------------------------------
# Prompt Template
# -----------------------------------

prompt = PromptTemplate(

    template="""
You are a helpful AI assistant.

Answer ONLY from the provided context.

You MUST answer in the EXACT format below.

what is :
<write 2 concise sentences>

Key Points:
• point 1
• point 2
• point 3
• point 4

Summary:
<write 2 concise sentences>
defination give only when there is it some definition in the context otherwise leave it blank

Rules:
- Keep formatting clean
- Use bullet points
- Do not repeat information
- Keep answers concise but informative
- If answer is not found in context say:
  "I don't know from the provided context."

Context:
{context}

Question:
{question}

Answer:
""",

    input_variables=["context", "question"]
)

# -----------------------------------
# Create Vector DB from YouTube
# -----------------------------------

def create_vector_db_from_youtube(video_url):

    global db

    try:

        # Handle different URL formats

        if "watch?v=" in video_url:

            video_id = video_url.split("watch?v=")[1].split("&")[0]

        elif "youtu.be/" in video_url:

            video_id = video_url.split("youtu.be/")[1].split("?")[0]

        else:

            return False

        print("Video ID:", video_id)

        # Fetch transcript

        ytt_api = YouTubeTranscriptApi()

        transcript = ytt_api.fetch(video_id)

        # Convert transcript to text

        full_text = " ".join(
        [item.text for item in transcript]
       )

        print("Transcript fetched successfully")

        # Split text

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(full_text)

        print("Chunks created:", len(chunks))

        # Create FAISS DB

        db = FAISS.from_texts(
            chunks,
            embeddings
        )

        # Save DB

        db.save_local("faiss_index")

        print("FAISS database created")

        return True

    except Exception as e:

     import traceback

     print("\n========== FULL ERROR ==========")

     traceback.print_exc()

     print("================================\n")

     return False

# -----------------------------------
# Home Route
# -----------------------------------

@app.route('/')
def home():

    return render_template('index.html')

# -----------------------------------
# Load YouTube Video
# -----------------------------------

@app.route('/load_video', methods=['POST'])
def load_video():

    video_url = request.form['video_url']

    success = create_vector_db_from_youtube(
        video_url
    )

    if success:

        return jsonify({
            "status": "YouTube video loaded successfully!"
        })

    else:

        return jsonify({
            "status": "Failed to load YouTube video."
        })

# -----------------------------------
# Chat Route
# -----------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    global db

    if db is None:

        return jsonify({
            "response": "Please load a YouTube video first."
        })

    user_message = request.form['message']

    print("Question:", user_message)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=db.as_retriever(
            search_kwargs={"k": 4}
        ),

        chain_type_kwargs={
            "prompt": prompt
        }
    )

    # Generate answer
    result = qa_chain.invoke({
        "query": user_message
    })

    print("Answer generated")

    return jsonify({
        "response": result["result"]
    })

# -----------------------------------
# Run App
# -----------------------------------

import os

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port
    )