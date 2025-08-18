class LLMConfig:
    """
    Configuration class for LLM (Language Learning Model).
    This class holds all the parameters for the model configuration.
    """

    TEMPERATURE = 0.0

    MODEL_NAME = "qwen3"
    TOP_P = 0.7
    TOP_K = 50
    REPETITION_PENALTY = 1

    SYSTEM_PROMPT = """If user asks to get a current date execute get_current_date tool.
Otherwise, return a text summary of user input - short representation of the text.

# ** Summary format to use **:
[summary]: ..."""


class APIConfig:
    """
    Configuration class for the API.
    This class holds the parameters for the API configuration.
    """

    HOST = "http://localhost:8000"


class SemanticSearchConfig:
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    PRODUCTS_CSV_PATH = "./data/products.csv"
    FAISS_INDEX_PATH = "./data/products.index"
