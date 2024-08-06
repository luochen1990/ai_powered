import os
from typing import List
import requests
from typing_extensions import Literal, TypedDict, NotRequired
from ai_powered.tool_call import make_tool

class SearchOption(TypedDict):
    siteSearch: NotRequired[str]
    dateRestrict: NotRequired[Literal["d[1]", "w[1]", "m[1]"]]

class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "mock_google_pse_api_key")
GOOGLE_ENGINE_ID = os.getenv("GOOGLE_ENGINE_ID", "mock_google_pse_engine_id")

@make_tool
def google_search(keywords: str, options: SearchOption) -> List[SearchResult]:
    ''' 用Google搜索互联网,对于专业问题请用英文关键词搜索 '''
    url = "https://www.googleapis.com/customsearch/v1"
    params : dict[str, str] = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_ENGINE_ID,
        "q": keywords,
        **options #type: ignore
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # 如果请求失败，抛出异常

    search_results = response.json()
    results : list[SearchResult] = []

    for item in search_results.get("items", []):
        results.append(SearchResult(
            title=item["title"],
            link=item["link"],
            snippet=item["snippet"]
        ))

    return results