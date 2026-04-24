"""
main_agent.py
عامل خودمختار که در GitHub Actions اجرا می‌شود.
ورودی: ISSUE_BODY (متن Issue) و AI_TYPE (برچسب)
خروجی: ایجاد/تغییر فایل‌ها در مخزن و سپس یک Pull Request
"""

import os
import json
import sys
from typing import Dict, Any, List, Optional

from llm_proxy import LLMProxy
from file_manager import FileManager
from prompts import get_system_prompt   # در سیکل ۴ این فایل را می‌سازیم


class AutonomousAgent:
    def __init__(self):
        # خواندن متغیرهای محیطی (تأمین شده در GitHub Actions)
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.issue_body = os.getenv("ISSUE_BODY", "No task provided")
        self.ai_type = os.getenv("AI_TYPE", "ai-deepseek")   # ai-gemini یا ai-deepseek

        if not self.deepseek_key and "deepseek" in self.ai_type:
            raise ValueError("DEEPSEEK_API_KEY not set in secrets")
        if not self.gemini_key and "gemini" in self.ai_type:
            raise ValueError("GEMINI_API_KEY not set in secrets")

        # مدل انتخابی
        self.model = "gemini" if "gemini" in self.ai_type else "deepseek"

        # راه‌اندازی پروکسی (در صورت نیاز می‌توانید proxy_base_url را از env بخوانید)
        # مثال: proxy_url = os.getenv("DEEPSEEK_PROXY_URL", None)
        self.llm = LLMProxy(
            gemini_api_key=self.gemini_key,
            deepseek_api_key=self.deepseek_key,
            proxy_base_url=None   # در صورت داشتن پروکسی، آدرس را وارد کنید
        )
        self.fm = FileManager(root_dir=".")

    def run(self):
        """نقطه ورود اصلی"""
        print(f"[Agent] Starting with model: {self.model}")
        print(f"[Agent] Task: {self.issue_body[:200]}...")

        # 1. دریافت درخت پروژه
        project_tree = self.fm.get_project_tree()
        print(f"[Agent] Project tree has {len(project_tree)} files.")

        # 2. ساخت پرامپت سیستمی + درخواست کاربر
        system_prompt = get_system_prompt(project_tree)
        full_prompt = f"{system_prompt}\n\n## User Request:\n{self.issue_body}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.issue_body}
        ]

        # 3. فراخوانی هوش مصنوعی (با format JSON)
        try:
            response_text = self.llm.chat(
                messages=messages,
                model=self.model,
                force_json=True   # مهم: DeepSeek می‌تواند JSON اجباری، Gemini نیز تلاش می‌کند
            )
            print("[Agent] Raw AI response:\n", response_text[:500])
        except Exception as e:
            print(f"[Agent] AI call failed: {e}")
            sys.exit(1)

        # 4. پردازش پاسخ JSON و اعمال تغییرات
        self._process_ai_response(response_text)

    def _process_ai_response(self, response_text: str):
        """پارس JSON و اعمال عملیات (با خودترمیمی)"""
        # پاک کردن可能的 backticks
        clean = response_text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"[Agent] Invalid JSON: {e}")
            print(f"Raw response: {response_text}")
            sys.exit(1)

        thoughts = data.get("thoughts", "No thoughts provided")
        print(f"\n[Agent Thoughts]: {thoughts}\n")

        operations = data.get("operations", [])
        if not operations:
            print("[Agent] No operations to perform.")
            return

        for op in operations:
            filepath = op.get("filepath")
            content = op.get("content")
            if not filepath or content is None:
                print(f"[Agent] Skipping invalid operation: {op}")
                continue

            print(f"[Agent] Applying change to {filepath} ...")
            success, msg = self.fm.apply_changes(filepath, content, run_runtime_check=True)

            if success:
                print(f"[Agent] ✅ {msg}")
            else:
                print(f"[Agent] ❌ Error: {msg}")
                # خودترمیمی (Self-Healing) با همان مدل
                fixed_content = self._heal_code(content, msg, filepath)
                if fixed_content:
                    retry_success, retry_msg = self.fm.apply_changes(filepath, fixed_content, run_runtime_check=True)
                    if retry_success:
                        print(f"[Agent] 🔧 Self-healing succeeded: {retry_msg}")
                    else:
                        print(f"[Agent] 💥 Self-healing failed: {retry_msg}")
                else:
                    print(f"[Agent] 🚫 Could not heal the code. Skipping {filepath}")
def _heal_code(self, bad_code: str, error_msg: str, filepath: str) -> Optional[str]:
    """درخواست اصلاح کد از هوش مصنوعی"""
    print(f"[Agent] Attempting self-healing for {filepath}...")
    heal_prompt = f"""
You wrote Python code that has an error.

File: {filepath}
Error Details:
{error_msg}

Bad Code:
{bad_code}

Please fix the error and return ONLY valid JSON in this format:
{{"content": "the fully fixed Python code here"}}

Make sure to escape double quotes and newlines properly.
"""
    messages = [
        {"role": "system", "content": "You are a code repair assistant. Return only valid JSON."},
        {"role": "user", "content": heal_prompt}
    ]
    try:
        response = self.llm.chat(messages, model=self.model, force_json=True)
        clean_resp = response.strip()
        if clean_resp.startswith("
```json"):
clean_resp = clean_resp[7:]
if clean_resp.startswith("
```"):
            clean_resp = clean_resp[3:]
        if clean_resp.endswith("
```"):
clean_resp = clean_resp[:-3]
data = json.loads(clean_resp.strip())
fixed = data.get("content")
if fixed:
return fixed
else:
print("[Agent] No 'content' field in healing response.")
return None
except Exception as e:
print(f"[Agent] Healing failed: {e}")
return None
