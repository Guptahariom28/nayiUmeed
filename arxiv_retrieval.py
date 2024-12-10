from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

arxiv_wrapper=ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=1000)
tool_arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
