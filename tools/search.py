from langchain_tavily import TavilySearch


def get_search_tool(max_results: int = 5) -> TavilySearch:
    return TavilySearch(max_results=max_results)
