"""
CrewAI crew definition for the AI Study Assistant.

Uses:
- Pinecone for document retrieval
- Sentence Transformers for embeddings
- Ollama + llama3.2 for local LLM
- CrewAI for Research, Analysis and Review agents
"""

import os

from crewai import Agent, Task, Crew, Process, LLM

from vector_store import search_documents


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)


def get_llm():
    """
    Create a CrewAI-compatible LLM using local Ollama.
    """
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
    )


class StudyAssistantCrew:

    def __init__(self):

        llm = get_llm()

        self.research_agent = Agent(
            role="Research Agent",
            goal=(
                "Find and summarize the most relevant passages from "
                "the knowledge base for the user's question."
            ),
            backstory=(
                "An expert researcher who locates and summarizes "
                "relevant information from uploaded study documents. "
                "Never invent facts."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        self.analysis_agent = Agent(
            role="Analysis Agent",
            goal=(
                "Analyze the retrieved information and prepare a "
                "clear and accurate answer."
            ),
            backstory=(
                "A careful analyst who combines retrieved evidence "
                "into a simple and understandable answer."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

        self.review_agent = Agent(
            role="Review Agent",
            goal=(
                "Check the answer against the retrieved document "
                "content and produce the final verified answer."
            ),
            backstory=(
                "A strict reviewer who checks that every important "
                "claim is supported by the uploaded documents."
            ),
            llm=llm,
            verbose=True,
            allow_delegation=False,
        )

    def run(self, question: str, top_k: int = 5) -> dict:

        retrieved_chunks = search_documents(
            question,
            top_k=top_k
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "I couldn't find anything relevant in the "
                    "uploaded documents to answer that question."
                ),
                "sources": [],
            }

        context_text = "\n\n".join(
            f"[Source {i + 1}] {chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        )

        research_task = Task(
            description=(
                f"The user asked: '{question}'.\n\n"
                f"Retrieved content:\n{context_text}\n\n"
                "Summarize only the information relevant to the "
                "question. Do not add outside information."
            ),
            agent=self.research_agent,
            expected_output=(
                "A concise summary of the relevant information."
            ),
        )

        analysis_task = Task(
            description=(
                f"Using the research summary, answer this question:\n"
                f"'{question}'\n\n"
                "Use only information supported by the retrieved "
                "content. If information is missing, say so instead "
                "of guessing."
            ),
            agent=self.analysis_agent,
            expected_output=(
                "A clear and accurate draft answer."
            ),
            context=[research_task],
        )

        review_task = Task(
            description=(
                "Review the draft answer against the retrieved "
                "content. Remove unsupported claims and correct "
                "mistakes. Output only the final verified answer."
            ),
            agent=self.review_agent,
            expected_output=(
                "The final fact-checked answer."
            ),
            context=[analysis_task],
        )

        crew = Crew(
            agents=[
                self.research_agent,
                self.analysis_agent,
                self.review_agent,
            ],
            tasks=[
                research_task,
                analysis_task,
                review_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        return {
            "answer": str(result),
            "sources": retrieved_chunks,
        }