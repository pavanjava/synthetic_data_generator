from models import GeneratorParams
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from typing import List
from vector_store_indexer import VectorStoreIndexer
import pandas as pd
import json


class DataGenerator:
    def __init__(self):
        self.params = None
        self.vector_store_indexer = VectorStoreIndexer()
        self.df = pd.DataFrame(columns=['questions', 'context'])

    def generate_data(self, params: GeneratorParams):
        """
        Process the input parameters and generate synthetic data.
        This is where you would implement your data generation logic.
        """
        # Print parameters for demonstration
        print(f"Generating data with parameters:")
        print(f"File: {params.file_path}")
        print(f"LLM: {params.llm_choice}")
        print(f"Chunk size: {params.chunk_size}")
        print(f"Chunk overlap: {params.chunk_overlap}")
        print(f"Questions per chunk: {params.questions_per_chunk}")
        print(f"Using vectordb: {params.use_vectordb}")

        if params.llm_choice == "OpenAI GPT-4o":
            Settings.llm = OpenAI(model="gpt-4o")
            Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
        elif params.llm_choice == "Anthropic Sonnet-3.5":
            Settings.llm = Anthropic(model="claude-3-5-sonnet-20241022")
            Settings.embed_model = FastEmbedEmbedding()
        elif params.llm_choice == "Ollama - Llama-3.1":
            Settings.llm = Ollama(model="llama3.1:latest", request_timeout=300, base_url="http://localhost:11434")
            Settings.embed_model = OllamaEmbedding(model_name="llama3.1:latest", base_url="http://localhost:11434")
        elif params.llm_choice == "Ollama - Deepseek-r1":
            Settings.llm = Ollama(model="deepseek-r1:8b", request_timeout=300, base_url="http://localhost:11434")
            Settings.embed_model = OllamaEmbedding(model_name="deepseek-r1:8b", base_url="http://localhost:11434")

        Settings.chunk_size = int(params.chunk_size)
        Settings.chunk_overlap = int(params.chunk_overlap)

        # chunker
        base_splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        documents: List[Document] = SimpleDirectoryReader(input_files=[params.file_path]).load_data()
        nodes = base_splitter.get_nodes_from_documents(documents=documents)

        print(f"length of nodes: {len(nodes)}")
        print("=" * 100)

        for node in nodes:
            # print(node.get_content())
            prompt = self.generate_question_prompt(node.get_content(), int(params.questions_per_chunk))
            questions = self.chat_with_llm(user_message=prompt)
            print(f"questions: {questions}")
            print("=" * 100)

            if params.use_vectordb:
                self.vector_store_indexer.index_data(chunk=node.get_content())

            self.df = self.validate_json_questions_and_create_df(json_str=questions.message.content,
                                                                 chunk=node.get_content(),
                                                                 expected_count=params.questions_per_chunk, df=self.df)

    def generate_question_prompt(self, chunk: str, num_questions: int) -> str:
        prompt = f"""
                You are an AI assistant tasked with generating a single, realistic question-answer pair based on a given document. 
                The question should be something a user might naturally ask when seeking information contained in the document.
                
                Given: {chunk}
                
                Instructions:
                1. Analyze the key topics, facts, and concepts in the given document, only focus on the chosen topic, concepts.
                2. Generate {num_questions} similar questions that a user might ask to find the information in this document that does NOT contain any company name.
                3. Use natural language and occasionally include typos or colloquialisms to mimic real user behavior in the question.
                4. Ensure the question is semantically related to the document content WITHOUT directly copying phrases.
                5. Make sure that all of the questions are similar to each other. I.E. All asking about a similar topic/requesting the same information.
                
                Output Format:
                Return a JSON object with the following structure without back ticks:
                
                {{
                  "question_1": "Generated question text",
                  "question_2": "Generated question text",
                  ...
                }}
                
                Be creative, think like a curious user, and generate your {num_questions} similar questions that would 
                naturally lead to the given document in a semantic search. Ensure your response is a valid JSON object 
                containing only the questions nothing else no explanation, no reasons nor extra text.
                
                """

        return prompt

    def chat_with_llm(self, user_message: str):
        messages = [
            ChatMessage(
                role="system", content="You are a helpful assistant following the instruction from user."
            ),
            ChatMessage(role="user", content=user_message),
        ]
        resp = Settings.llm.chat(messages)
        return resp

    def validate_json_questions_and_create_df(self, json_str: str, chunk: str, expected_count: int,
                                              df: pd.DataFrame) -> pd.DataFrame:

        try:
            questions = json.loads(json_str)

            # Check if it's a dict and has correct number of questions
            if not isinstance(questions, dict) or len(questions) != expected_count:
                return None

            # Check if keys are correctly formatted
            expected_keys = {f"question_{i + 1}" for i in range(expected_count)}
            if set(questions.keys()) != expected_keys:
                return None

            # create df from each question
            rows = []
            for key, question in questions.items():
                rows.append({'questions': question, 'context': chunk})

            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            print(df.head())
            df.to_csv("synthetic_data.csv", index=False)
            return df

        except json.JSONDecodeError as e:
            print(e)
