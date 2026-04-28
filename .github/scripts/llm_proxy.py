import os
from google import genai

def get_ai_response(prompt):
    """
    ارتباط با مدل جمینای با استفاده از آخرین نسخه SDK (v2.0)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")

    client = genai.Client(api_key=api_key)
    
    try:
        # استفاده از مدل gemini-1.5-pro که پایدارترین نسخه است
        response = client.models.generate_content(
            model='gemini-1.5-pro',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error in AI Call: {str(e)}"
