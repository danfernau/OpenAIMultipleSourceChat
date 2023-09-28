#Chat with Multiple PDFs
This code creates a conversational AI that can answer questions based on the content of multiple uploaded PDF files. The program utilizes Streamlit for the user interface, various NLP libraries for text processing and embeddings, and language models for generating responses.
Usage
 

#Install the required dependencies.
Run the Streamlit application using the command: streamlit run your_file_name.py
Upload your PDFs using the file uploader in the sidebar.
Click "Process" to process the PDFs and initialize the conversational AI.
Ask questions related to your documents in the text input field.
Dependencies
 

#Streamlit
python-dotenv
PyPDF2
langchain
htmlTemplates (custom library)
openai

#Functions
get_pdf_text(pdf_docs): Extracts the text from the uploaded PDF files.
get_text_chunks(text): Splits the extracted text into smaller chunks.
get_vectorstore(text_chunks): Creates a FAISS vector store from the text chunks using HuggingFace Instruct embeddings.
get_conversation_chain(vectorstore): Initializes a ConversationalRetrievalChain with the vector store and a large language model.
handle_userinput(user_question): Handles user input and generates a response using the ConversationalRetrievalChain.
main(): Main function that runs the Streamlit application.

#Components
ConversationalRetrievalChain: A class that combines the language model, retriever, and memory for conversational retrieval tasks.
HuggingFaceInstructEmbeddings: A class that provides HuggingFace Instruct embeddings for given text.
HuggingFaceHub: A class for working with HuggingFace Hub models.
AzureOpenAI: A class for working with Azure OpenAI models (currently unused in the code).
OpenAIEmbeddings: A class for working with OpenAI embeddings (currently unused in the code).
FAISS: A class that provides a fast similarity search and clustering of dense vectors.
CharacterTextSplitter: A class that splits text into chunks based on a specified separator and chunk size.

#Notes
The current code uses the HuggingFace Instruct embeddings and Google's Flan-T5-XXL model. You can switch to other embeddings or models by modifying the relevant lines in the get_vectorstore and get_conversation_chain functions.
The conversation history is stored in the Streamlit session state, allowing users to continue their conversation with the AI as they ask more questions.