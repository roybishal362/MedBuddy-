"""
MedBuddy — Conversation Memory
Session-scoped conversational Q&A for Mode 2 follow-up questions.
Uses LangChain ConversationBufferWindowMemory (k=5).
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import primary_llm, CONVERSATION_FOLLOWUP_PROMPT


class ConversationMemory:
    """Manages conversational context for Mode 2 report Q&A."""

    def __init__(self):
        self.history = []  # List of (role, message) tuples
        self.max_turns = 5
        self.report_context = None
        self.narrative_summary = ""
        self.entities_context = ""

    def set_report_context(self, narrative_summary: str, entities_with_flags: list):
        """
        Initialize context with a new report.
        Clears previous conversation history.
        """
        self.history = []
        self.narrative_summary = narrative_summary

        # Format entities for context
        entity_lines = []
        for e in entities_with_flags:
            line = f"- {e.get('term', 'Unknown')}: {e.get('patient_value', 'N/A')} {e.get('unit', '')} "
            line += f"(Ref: {e.get('reference_range', 'N/A')}) — {e.get('flag', 'UNKNOWN')}"
            if e.get('explanation'):
                line += f"\n  Explanation: {e['explanation']}"
            entity_lines.append(line)
        self.entities_context = "\n".join(entity_lines)

        self.report_context = {
            "summary": narrative_summary,
            "entities": self.entities_context
        }

    def ask(self, user_question: str, language: str = "English") -> str:
        """
        Process a follow-up question about the uploaded report.

        Args:
            user_question: The patient's question
            language: Response language

        Returns:
            str: The assistant's response
        """
        if not self.report_context:
            if language == "Hindi":
                return "कृपया पहले एक रिपोर्ट अपलोड करें।"
            return "Please upload a report first before asking questions."

        # Build chat history string
        chat_history_str = ""
        for role, msg in self.history[-self.max_turns:]:
            if role == "patient":
                chat_history_str += f"Patient: {msg}\n"
            else:
                chat_history_str += f"MedBuddy: {msg}\n"

        try:
            prompt = CONVERSATION_FOLLOWUP_PROMPT.format(
                narrative_summary=self.narrative_summary,
                entities_with_flags_and_explanations=self.entities_context,
                chat_history=chat_history_str if chat_history_str else "No previous conversation.",
                user_question=user_question,
                language=language
            )

            response = primary_llm.invoke(prompt)
            answer = response.content.strip()

            # Add to history
            self.history.append(("patient", user_question))
            self.history.append(("assistant", answer))

            # Trim to max turns
            if len(self.history) > self.max_turns * 2:
                self.history = self.history[-(self.max_turns * 2):]

            return answer

        except Exception as e:
            print(f"[ConversationMemory] Error: {e}")
            if language == "Hindi":
                return "क्षमा करें, एक त्रुटि हुई। कृपया पुनः प्रयास करें।"
            return "Sorry, an error occurred. Please try again."

    def clear(self):
        """Clear all conversation history and context."""
        self.history = []
        self.report_context = None
        self.narrative_summary = ""
        self.entities_context = ""

    def get_history(self) -> list:
        """Return conversation history for display."""
        return self.history.copy()
