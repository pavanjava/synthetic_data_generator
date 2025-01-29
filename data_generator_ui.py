import streamlit as st
import tempfile
import os
from dotenv import load_dotenv, find_dotenv
from data_generator import DataGenerator
from models import GeneratorParams

load_dotenv(find_dotenv())

# Initialize session state at the start
if 'generator' not in st.session_state:
    st.session_state['generator'] = DataGenerator()


def create_qa_interface():
    # Set page title and layout
    st.set_page_config(page_title="Document QA Interface", layout="wide")

    # Create two columns for layout
    left_col, right_col = st.columns([1, 2])

    with left_col:
        # File upload section
        uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt", "docx"])

        # LLM selection
        st.write("Choose your LLM")
        llm_choice = st.radio(
            "Select LLM",
            ["OpenAI GPT-4o", "Anthropic Sonnet-3.5", "Ollama - Llama-3.1", "Ollama - Deepseek-r1"],
            label_visibility="collapsed"
        )

        # API Key input
        api_key = st.text_input("API Key:", type="password")

    with right_col:
        # Parameters section
        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.text_input("chunk size")
            questions_per_chunk = st.text_input("number of questions per chunk")

        with col2:
            chunk_overlap = st.text_input("chunk overlap")

        # Vectordb checkbox
        use_vectordb = st.checkbox("index chunks in vectordb?", value=True)

        # Generate button
        if st.button("generate synthetic data", use_container_width=False):
            if uploaded_file is None:
                st.error("Please upload a file first!")
                return

            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                file_path = tmp_file.name

            try:
                # Collect all parameters
                params = GeneratorParams(
                    file_path=file_path,
                    llm_choice=llm_choice,
                    api_key=api_key,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    questions_per_chunk=int(questions_per_chunk),
                    use_vectordb=use_vectordb
                )

                # Generate data
                st.session_state.generator.generate_data(params)
                st.success("Data generation completed!")

            except ValueError as e:
                st.error(f"Invalid input parameters: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(file_path)


if __name__ == "__main__":
    create_qa_interface()
