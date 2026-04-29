"""LangChain tool for SalienceGate evaluation in Ada assistant integration."""

from langchain.tools import BaseTool
from typing import Optional, Type, Any, Dict
from pydantic import BaseModel, Field
import json


class SalienceInput(BaseModel):
    """Input schema for SalienceGate evaluation."""
    user_query: str = Field(description="The user's current query or message")
    conversation_history: Optional[str] = Field(
        default="",
        description="Previous conversation context as a string"
    )


class SalienceGateTool(BaseTool):
    """Tool that wraps SalienceGate evaluation for filtering irrelevant queries."""

    name: str = "salience_gate"
    description: str = (
        "Evaluates whether a user query is salient (relevant) to the assistant's capabilities. "
        "Returns a salience score and recommendation. Use this before processing any user query."
    )
    args_schema: Type[BaseModel] = SalienceInput

    def _evaluate_salience(self, user_query: str, conversation_history: str = "") -> Dict[str, Any]:
        """
        Mock implementation of SalienceGate evaluation.
        Replace with actual API call to SalienceGate service.
        """
        # Heuristic-based mock logic
        low_salience_keywords = ["hello", "hi", "thanks", "ok", "test", "ignore"]
        query_lower = user_query.lower()

        if any(kw in query_lower for kw in low_salience_keywords) and len(user_query.split()) < 4:
            score = 0.2
            decision = "ignore"
            reason = "Query appears to be greeting or very low information content"
        elif len(user_query.split()) < 2:
            score = 0.1
            decision = "ignore"
            reason = "Query too short to be meaningful"
        else:
            score = 0.85
            decision = "process"
            reason = "Query contains substantive content"

        # If conversation history adds context, boost score slightly
        if conversation_history and len(conversation_history) > 50:
            score = min(score + 0.1, 1.0)
            if score > 0.7:
                decision = "process"

        return {
            "salience_score": score,
            "decision": decision,
            "reason": reason,
            "query": user_query
        }

    def _run(
        self,
        user_query: str,
        conversation_history: Optional[str] = ""
    ) -> str:
        """Run SalienceGate evaluation synchronously."""
        result = self._evaluate_salience(user_query, conversation_history)
        return json.dumps(result, indent=2)

    async def _arun(
        self,
        user_query: str,
        conversation_history: Optional[str] = ""
    ) -> str:
        """Run SalienceGate evaluation asynchronously."""
        # For simplicity, call synchronous version
        return self._run(user_query, conversation_history)


def get_salience_tool() -> SalienceGateTool:
    """Factory function to get the SalienceGate tool."""
    return SalienceGateTool()
