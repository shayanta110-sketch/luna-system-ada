import os
from google import genai

class LLMProxy:
    def __init__(self, model_name='gemini-1.5-pro'):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")
        
        # استفاده از کلاینت جدید گوگل
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def ask(self, prompt):
        """متد اصلی برای دریافت پاسخ از هوش مصنوعی"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"AI Call Failed: {str(e)}"

    # برای اطمینان از سازگاری با کدهای قدیمی، اگر متد شما نام دیگری داشت:
    def get_completion(self, prompt):
        return self.ask(prompt)
