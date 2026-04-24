"""
file_manager.py
مدیریت فایل‌ها با قابلیت:
- syntax check (AST)
- runtime test در محیط ایزوله (subprocess با timeout)
- بکاپ‌گیری قبل از تغییر
"""

import os
import shutil
import ast
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Tuple, List

class FileManager:
    def __init__(self, root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)

    def get_project_tree(self) -> List[str]:
        """بازگشت لیست تمام مسیرهای فایل‌های پروژه (به جز .git و __pycache__)"""
        tree = []
        for root, dirs, files in os.walk(self.root_dir):
            # رد کردن دایرکتوری‌های نامرتبط
            dirs[:] = [d for d in dirs if d not in ('.git', '__pycache__', '.venv', 'env', 'node_modules')]
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.root_dir)
                tree.append(rel_path)
        return tree

    def _backup_file(self, filepath: str) -> str:
        """ایجاد بکاپ با زمان در کنار فایل اصلی (مثلاً file.py.bak_20250221_153022)"""
        if not os.path.exists(filepath):
            return ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.bak_{timestamp}"
        shutil.copy(filepath, backup_path)
        return backup_path

    def check_python_syntax(self, code_string: str) -> Tuple[bool, str]:
        """بررسی صحت نحوی کد پایتون"""
        try:
            ast.parse(code_string)
            return True, "Syntax OK"
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def check_python_runtime(self, code_string: str, timeout_sec: int = 5) -> Tuple[bool, str]:
        """
        اجرای کد در یک محیط موقت ایزوله برای تشخیص خطاهای زمان اجرا.
        بازگشت: (success, error_message)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code_string)
            temp_path = f.name

        try:
            # اجرای کد با محدودیت زمان
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=tempfile.gettempdir()  # اجرا در دایرکتوری موقت برای کاهش عوارض جانبی
            )
            if result.returncode != 0:
                return False, f"Runtime Error:\n{result.stderr.strip()}"
            return True, "Runtime OK"
        except subprocess.TimeoutExpired:
            return False, f"Runtime Timeout (>{timeout_sec} seconds)"
        except Exception as e:
            return False, f"Unexpected error during execution: {str(e)}"
        finally:
            # پاک کردن فایل موقت
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def apply_changes(self, filepath: str, new_content: str, run_runtime_check: bool = True) -> Tuple[bool, str]:
        """
        ذخیره تغییرات در فایل، پس از بررسی‌های امنیتی.
        اگر run_runtime_check=True باشد، علاوه بر سینتکس، کد را اجرا می‌کند.
        بازگشت: (success, message)
        """
        full_path = os.path.join(self.root_dir, filepath)
        # ایجاد دایرکتوری والد اگر وجود نداشت
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # بررسی سینتکس برای فایل‌های پایتون
        if filepath.endswith('.py'):
            syntax_ok, syntax_msg = self.check_python_syntax(new_content)
            if not syntax_ok:
                return False, syntax_msg

            if run_runtime_check:
                runtime_ok, runtime_msg = self.check_python_runtime(new_content)
                if not runtime_ok:
                    return False, runtime_msg

        # ایجاد بکاپ از فایل موجود (در صورت وجود)
        if os.path.exists(full_path):
            backup_path = self._backup_file(full_path)
            print(f"[FileManager] Backup created: {backup_path}")

        # نوشتن محتوای جدید
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            return False, f"Write error: {str(e)}"

        return True, f"File {filepath} updated successfully."

    def read_file(self, filepath: str) -> str:
        """خواندن محتوای فعلی یک فایل"""
        full_path = os.path.join(self.root_dir, filepath)
        if not os.path.exists(full_path):
            return ""
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()


# نمونه تست (برای اجرای محلی)
if __name__ == "__main__":
    fm = FileManager(".")
    # تست ایجاد فایل موقت
    success, msg = fm.apply_changes("test_output.py", "print('Hello from safe execution')", run_runtime_check=True)
    print(f"Test result: {success}, {msg}")
