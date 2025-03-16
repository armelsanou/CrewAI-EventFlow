import asyncio
import logging
from crewai import Agent, Crew, Task, LLM
from crewai.flow.flow import Flow, listen, start, router, or_, and_
from pydantic import BaseModel, Field
from typing import List, Union
from litellm import completion
from crewai.tools import BaseTool
from typing import List
import sys
import agentops
sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ["AGENTOPS_API_KEY"] = "991c0f1a-e025-4382-a189-2acea228ea59"

from dotenv import load_dotenv
load_dotenv()

agentops.init(os.getenv("AGENTOPS_API_KEY"), auto_start_session=True)


# Configuration du logging pour le monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# État du flow avec une persistance simulée
class BlogState(BaseModel):
    topic: str = ""
    research_notes: List[str] = []
    fact_checked_notes: List[str] = []
    draft_content: str = ""
    edited_content: str = ""
    final_content: str = ""
    validation_status: str = "Pending"

class BlogContentFlow(Flow[BlogState]):
    
    model = LLM(
        model='ollama/llama3.2:3b',  #you can use any other model downloaded using ollama pull model-name
        base_url='http://localhost:11434',
    )

    def __init__(self):
        super().__init__()

        # Définition des agents
        self.researcher = Agent(
            name="Researcher",
            role="researcher",
            goal="Conduct thorough research on tech topics",
            backstory="Expert tech researcher with years of experience",
            llm=self.model,
            verbose=True
        )

        self.trend_analyst = Agent(
            name="Trend Analyst",
            role="analyst",
            goal="Identify current tech trends",
            backstory="Analyzes industry trends to find relevant topics",
            llm=self.model,
            verbose=True,
            tools=[]
        )

        self.fact_checker = Agent(
            name="Fact Checker",
            role="fact_checker",
            goal="Verify the accuracy of research data",
            backstory="Ensures all research is factually correct",
            llm=self.model,
            verbose=True
        )

        self.writer = Agent(
            name="Writer",
            role="writer",
            goal="Create engaging tech content",
            backstory="Experienced tech writer and blogger",
            llm=self.model,
            verbose=True
        )

        self.editor = Agent(
            name="Editor",
            role="editor",
            goal="Ensure content quality and accuracy",
            backstory="Senior content editor with technical expertise",
            llm=self.model,
            verbose=True
        )

    @start()
    def generate_topic(self):
        """ Génération dynamique du sujet via un analyste de tendances """
        response = completion(
            model=self.model.model,
            messages=[{"role": "user", "content": "Generate a trending tech blog post topic for 2024."}]
        )
        topic = response["choices"][0]["message"]["content"]
        self.state.topic = topic
        logging.info(f"Generated Topic: {topic}")
        return topic

    @listen(generate_topic)
    def conduct_research(self, topic):
        """ Recherche et collecte d'informations par le chercheur """
        research_crew = Crew(
            agents=[self.researcher],
            tasks=[
                Task(
                    description=f"Research key points about: {topic}",
                    agent=self.researcher,
                    expected_output="A list of key points with sources on the topic"
                ),
                Task(
                    description="Identify relevant statistics and examples",
                    agent=self.researcher,
                    expected_output="A list of statistics and real-world examples related to the topic"
                )
            ]
        )
        
        research_results = research_crew.kickoff()
        self.state.research_notes = research_results

        logging.info(f"Research Notes: {research_results}")

        if not research_results:  # Vérification si la recherche est vide
            logging.warning("Research failed. Rerouting to trend analyst for a better topic.")
            return self.generate_topic()  # Revenir à la génération du sujet si la recherche échoue
        
        return research_results

    @listen(conduct_research)
    def fact_check_research(self, research):
        """ Vérification des faits avant d'envoyer à l'écriture """
        fact_check_crew = Crew(
            agents=[self.fact_checker],
            tasks=[
                Task(
                    description=f"Fact-check the following research notes: {research}",
                    agent=self.fact_checker,
                    expected_output="A corrected and validated set of research notes"
                )
            ]
        )
        
        validated_research = fact_check_crew.kickoff()
        self.state.fact_checked_notes = validated_research

        logging.info(f"Fact-checked Notes: {validated_research}")

        if not validated_research:
            logging.error("Fact-checking failed. Returning to researcher.")
            return self.conduct_research(self.state.topic)  # Relancer la recherche en cas d'échec

        return validated_research

    @listen(and_(fact_check_research, conduct_research))
    def write_content(self, research):
        """ Rédaction du contenu après validation des données """
        writing_crew = Crew(
            agents=[self.writer],
            tasks=[
                Task(
                    description=f"Write a blog post about {self.state.topic} using this research: {research}",
                    agent=self.writer,
                    expected_output="A well-structured blog post"
                )
            ]
        )
        
        draft = writing_crew.kickoff()
        self.state.draft_content = draft

        logging.info(f"Draft Content: {draft}")

        return draft

    @listen(write_content)
    def edit_content(self, draft):
        """ Édition et correction du brouillon """
        editing_crew = Crew(
            agents=[self.editor],
            tasks=[
                Task(
                    description=f"Edit and improve this blog post: {draft}",
                    agent=self.editor,
                    expected_output="A polished and well-structured blog post"
                )
            ]
        )
        
        edited_content = editing_crew.kickoff()
        self.state.edited_content = edited_content

        logging.info(f"Edited Content: {edited_content}")

        return edited_content

    @router(edit_content)
    def validate_content(self, edited_content):
        """ Validation finale avec contrôle qualité """
        if "error" in edited_content.lower():
            logging.warning("Editing flagged an issue. Sending back to writer.")
            return self.write_content(self.state.fact_checked_notes)  # Rerouter en cas de problème

        self.state.final_content = edited_content
        self.state.validation_status = "Approved"

        logging.info("Content successfully validated and finalized.")
        return edited_content

async def main():
    flow = BlogContentFlow()
    flow.plot()  # Visualisation du workflow
    result = await flow.kickoff_async()

    logging.info(f"Final Blog Post:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
