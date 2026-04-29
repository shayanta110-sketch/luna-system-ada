# ada_guardian.py
import os
import hashlib
import re
from typing import Union, List, Dict, Any

class Guardian:
    """
    کلاس Guardian مسئول بررسی مسیرها، اعتبارسنجی داده‌ها و اسکن کدها
    برای پروژه ADA می‌باشد.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        مقداردهی اولیه کلاس Guardian با تنظیمات اختیاری.

        Args:
            config (dict, optional): تنظیمات پیکربندی مانند حداکثر حجم فایل،
                                     الگوهای مجاز و غیره.
        """
        self.config = config or {}
        self.max_file_size = self.config.get("max_file_size", 10 * 1024 * 1024)  # 10 MB
        self.allowed_extensions = self.config.get("allowed_extensions", [".py", ".txt", ".json", ".yaml"])
        self.forbidden_patterns = self.config.get("forbidden_patterns", [
            r"eval\s*\(", r"exec\s*\(", r"__import__\s*\(", r"subprocess\.Popen"
        ])

    def check_path(self, path: str) -> Dict[str, Any]:
        """
        بررسی می‌کند که آیا مسیر داده شده وجود دارد، قابل دسترس و معتبر است یا خیر.

        Args:
            path (str): مسیر فایل یا دایرکتوری جهت بررسی.

        Returns:
            dict: نتیجه بررسی شامل کلیدهای 'valid' (bool)، 'message' (str) و
                  در صورت موفقیت 'size' (int) و 'extension' (str).
        """
        result = {
            "valid": False,
            "message": "",
            "path": path
        }

        # بررسی وجود مسیر
        if not os.path.exists(path):
            result["message"] = f"Path does not exist: {path}"
            return result

        # اگر مسیر یک دایرکتوری باشد
        if os.path.isdir(path):
            result["valid"] = True
            result["message"] = "Directory is valid and accessible."
            result["is_dir"] = True
            return result

        # اگر فایل است، بررسی های اضافی انجام بده
        if os.path.isfile(path):
            # بررسی حجم فایل
            file_size = os.path.getsize(path)
            if file_size > self.max_file_size:
                result["message"] = f"File too large: {file_size} bytes (max {self.max_file_size})"
                return result

            # بررسی پسوند مجاز
            _, ext = os.path.splitext(path)
            if ext not in self.allowed_extensions:
                result["message"] = f"File extension '{ext}' not allowed. Allowed: {self.allowed_extensions}"
                return result

            # بررسی قابلیت خواندن
            if not os.access(path, os.R_OK):
                result["message"] = f"File is not readable: {path}"
                return result

            result["valid"] = True
            result["message"] = "File is valid and accessible."
            result["size"] = file_size
            result["extension"] = ext
            result["is_dir"] = False
            return result

        result["message"] = f"Path is neither a file nor a directory: {path}"
        return result

    def validate_data(self, data: Union[str, bytes], expected_hash: str = None) -> Dict[str, Any]:
        """
        اعتبارسنجی داده‌ها از نظر یکپارچگی (با هش) و محتوای مخرب.

        Args:
            data (str or bytes): داده ورودی جهت بررسی.
            expected_hash (str, optional): هش مورد انتظار (SHA-256) برای تطابق.

        Returns:
            dict: نتیجه شامل 'valid' (bool) و 'message' (str) و در صورت درخواست 'hash' (str).
        """
        result = {
            "valid": False,
            "message": ""
        }

        # تبدیل داده به bytes برای یکسان‌سازی
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data

        # محاسبه هش داده
        sha256_hash = hashlib.sha256(data_bytes).hexdigest()
        result["hash"] = sha256_hash

        # بررسی هش مورد انتظار
        if expected_hash and sha256_hash != expected_hash:
            result["message"] = f"Hash mismatch. Expected {expected_hash}, got {sha256_hash}"
            return result

        # جستجوی الگوهای ممنوعه در داده (برای داده متنی)
        if isinstance(data, str) or (isinstance(data, bytes) and data_bytes.decode('utf-8', errors='ignore')):
            text = data if isinstance(data, str) else data_bytes.decode('utf-8', errors='ignore')
            for pattern in self.forbidden_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    result["message"] = f"Forbidden pattern detected: {pattern}"
                    return result

        result["valid"] = True
        result["message"] = "Data validation passed."
        return result

    def scan_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        اسکن کد برای یافابی مشکلات امنیتی، خطاهای احتمالی یا انحراف از استانداردها.

        Args:
            code (str): کد منبع به صورت رشته.
            language (str): زبان برنامه‌نویسی (پیش‌فرض 'python').

        Returns:
            dict: شامل 'issues' (list)، 'summary' (str) و 'score' (int از 0 تا 100).
        """
        if language != "python":
            return {
                "issues": ["Only Python code scanning is supported currently."],
                "summary": "Unsupported language.",
                "score": 0
            }

        issues = []
        score = 100  # شروع با نمره کامل

        # چک کردن الگوهای ممنوعه
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Potentially dangerous pattern found: {pattern}")
                score -= 20  # هر کدام 20 امتیاز کم کن

        # چک کردن توابع یا کلمات کلیدی خطرناک دیگر (اختیاری)
        dangerous_keywords = ["os.system", "subprocess.call", "pickle.loads", "__reduce__"]
        for kw in dangerous_keywords:
            if kw in code:
                issues.append(f"Dangerous function/module usage: {kw}")
                score -= 15

        # محدودیت امتیاز
        if score < 0:
            score = 0

        # ایجاد خلاصه
        if len(issues) == 0:
            summary = "No security issues detected in code."
        else:
            summary = f"Found {len(issues)} potential issue(s)."

        return {
            "issues": issues,
            "summary": summary,
            "score": score
        }
