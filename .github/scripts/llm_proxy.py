"""
llm_proxy.py
لایه انتزاعی برای ارتباط با مدل‌های Gemini و DeepSeek (از طریق API رسمی)
با قابلیت تلاش مجدد و پشتیبانی از پروکسی سفارشی
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
import google.generativeai as genai


class LLMProxy:
    """
    یک کلاینت واحد برای فراخوانی مدل‌های مختلف
    """
    def __init__(
        self,
        gemini_api_key: str,
        deepseek_api_key: str,
        proxy_base_url: Optional[str] = None,   # اگر می‌خواهید از پروکسی واسط استفاده کنید
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.gemini_key = gemini_api_key
        self.deepseek_key = deepseek_api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # تنظیم Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")

        # تنظیم DeepSeek (سازگار با OpenAI)
        base_url = proxy_base_url if proxy_base_url else "https://api.deepseek.com"
        self.deepseek_client = OpenAI(
            api_key=deepseek_api_key,
            base_url=base_url
        )

    def _call_deepseek(self, messages: List[Dict[str, str]], temperature: float = 0.7, force_json: bool = False) -> str:
        """فراخوانی DeepSeek با مدیریت خطا و تلاش مجدد"""
        for attempt in range(self.max_retries):
            try:
                params = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "temperature": temperature,
                }
                if force_json:
                    params["response_format"] = {"type": "json_object"}

                response = self.deepseek_client.chat.completions.create(**params)
                return response.choices[0].message.content

            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                wait = self.retry_delay * (2 ** attempt)
                time.sleep(wait)

    def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        """تبدیل messages به یک پرامپت ساده برای Gemini (چون Gemini فرمت OpenAI را مستقیماً نمی‌فهمد)"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        full_prompt = "\n".join(prompt_parts)

        for attempt in range(self.max_retries):
            try:
                response = self.gemini_model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay * (2 ** attempt))

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek",   # "deepseek" یا "gemini"
        temperature: float = 0.7,
        force_json: bool = False
    ) -> str:
        """
        ارسال یک مکالمه به مدل انتخاب شده و برگرداندن پاسخ متنی
        - model: 'deepseek' یا 'gemini'
        - force_json: فقط برای deepseek معنی دارد (output را مجبور به JSON می‌کند)
        """
        if model == "gemini":
            return self._call_gemini(messages)
        else:  # deepseek
            return self._call_deepseek(messages, temperature, force_json)


# نمونه استفاده (برای تست در محیط محلی)
if __name__ == "__main__":
    # توجه: این بخش فقط برای تست است و در GitHub Actions از طریق secrets تأمین می‌شود
    proxy = LLMProxy(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        proxy_base_url=None   # در صورت داشتن پروکسی، آدرس آن را وارد کنید
    )

    test_messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply in valid JSON."},
        {"role": "user", "content": "Say hello in JSON format: {\"greeting\": \"...\"}"}
    ]

    try:
        response = proxy.chat(test_messages, model="deepseek", force_json=True)
        print("DeepSeek response:", response)
    except Exception as e:
        print("Error:", e)
