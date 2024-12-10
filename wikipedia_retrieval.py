from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper=WikipediaAPIWrapper(top_k_results=5,doc_content_chars_max=100)
tool_wikipedia=WikipediaQueryRun(api_wrapper=api_wrapper)