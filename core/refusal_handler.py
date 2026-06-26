"""
MedBuddy — Refusal Handler
Blocks and redirects out-of-scope queries professionally.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import REFUSAL_RESPONSE


def handle_refusal(language: str = "English") -> str:
    """
    Return the appropriate refusal response based on language.

    Args:
        language: "English" or "Hindi"

    Returns:
        str: Refusal message
    """
    return REFUSAL_RESPONSE.get(language, REFUSAL_RESPONSE["English"])


def handle_report_redirect(language: str = "English") -> str:
    """
    Redirect user to Mode 2 when they ask about a report without uploading.

    Args:
        language: "English" or "Hindi"

    Returns:
        str: Redirect message
    """
    messages = {
        "English": (
            "It looks like you're asking about a medical report. Please switch to the "
            "'Upload Report' tab and upload your PDF to get a detailed, personalized analysis. "
            "I can then help you understand every result in your report."
        ),
        "Hindi": (
            "ऐसा लगता है कि आप एक मेडिकल रिपोर्ट के बारे में पूछ रहे हैं। कृपया "
            "'रिपोर्ट अपलोड करें' टैब पर जाएं और विस्तृत विश्लेषण के लिए अपनी PDF अपलोड करें। "
            "उसके बाद मैं आपकी रिपोर्ट का हर परिणाम समझाने में आपकी मदद कर सकता हूँ।"
        )
    }
    return messages.get(language, messages["English"])
