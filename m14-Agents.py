from langchain.agents import create_react_agent, AgentExecutor
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_tavily import TavilySearch
from langchain import hub
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import logging
from dotenv import load_dotenv
import re
load_dotenv()

logging.basicConfig(level=logging.INFO)

class YoutubeTranscriptSummarizerToolInput(BaseModel):
    url: str = Field(description="The URL of the YouTube video to get the Transcript of.")

class YoutubeTranscriptSummarizerTool(BaseTool):
    name: str = 'YouTube Transcript Summarizer Tool'
    description: str = 'A tool used to get the Transcript and then summarize it. Only works when given a URL.'
    args_schema: Type[BaseModel] = YoutubeTranscriptSummarizerToolInput

    def _run(self, URL: str):
        logging.info(f"Found URL: {URL}")
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", URL)
        if match:
            logging.info("Generating Transcript")
            content = " ".join(chunk.text for chunk in YouTubeTranscriptApi().fetch(video_id=match.group(1), languages=["en"]).snippets)
            logging.info("Transcript successfully generated")
            prompt = PromptTemplate(template="""
Summarize the following text clearly, and in extreme detail focusing on the most important announcements, breakthroughs, and controversies.

Requirements:
Use plain language understandable to a general audience.
If there are dates, numbers, or statistics, include them.


CONTENT: {content}
""", input_variables=["content"])
            parser = StrOutputParser()
            chain = prompt | ChatOpenAI(model='gpt-4o') | parser
            logging.info("Generating Summary")
            return chain.invoke({'content': content})
        else:
            return None

tavily_search_tool = TavilySearch(
    max_results=1,
    topic="general",
)

youtube_transcript_tool = YoutubeTranscriptSummarizerTool()
    
prompt = hub.pull("hwchase17/react")
tools = [tavily_search_tool, youtube_transcript_tool]
agent = create_react_agent(llm=ChatOpenAI(model = 'gpt-4o'), tools=tools, prompt=prompt)
agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

result = agent_exec.invoke({"input":
f"""
Follow this process:
Search YouTube (AND ONLY YouTube) for AI news 3 days before {datetime.now().strftime("%d-%mmm-%Y")} using the tavily search tool.
Review titles, publish dates, and view counts to determine which video is the most recent and popular.
Use the YouTube video link, the YouTube Transcript Tool to get the entire video transcript.
Then use the Summarizer tool by using the entire video transcript to generate and return the summary.
Just return the summary as is. Dont modify it and dont say anything extra.

Constraints:
Prioritize videos from reputable AI or tech channels.
Ignore videos older than 7 days.
Ensure the content is specifically about AI news, not tutorials or unrelated tech.
If multiple videos qualify, choose the one with the highest engagement.
"""})
print(result['output'])