# ada/agent/graph.py
"""
این فایل گراف اجرایی عامل Ada را با استفاده از LangGraph تعریف می‌کند.
یک گره پیش‌نیاز (pre_model_check) برای بررسی سلامت سیستم و انتخاب هوشمند مدل
با استفاده از nexus-router اضافه شده است.
"""

import logging
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from nexus_router import ResourceGuard, ModelRouter

logger = logging.getLogger(__name__)

# تعریف ساختار state سراسری عامل
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    force_model: Optional[str]        # مدل اجباری (در صورت کمبود منابع)
    system_warning: Optional[str]     # هشدار سیستم
    next_step: str                    # گام بعدی در گراف

# نمونه‌های یکتا از نگهبان و مسیریاب
_guard = ResourceGuard()
_router = ModelRouter()

def pre_model_check(state: AgentState) -> AgentState:
    """
    این گره پیش از هر بار فراخوانی مدل اجرا می‌شود.
    وظیفه:
    1. بررسی وضعیت RAM / VRAM با ResourceGuard
    2. در صورت ناامن بودن، مدل اجباری سبک‌تری تعیین می‌کند
    3. هشدار لازم را در state ثبت می‌نماید
    """
    health = _guard.check_health(force=True)
    if not health["safe_to_proceed"]:
        logger.warning(
            "منابع سیستم بحرانی است: %s. مدل fallback انتخاب می‌شود.",
            health["reasons"]
        )
        # انتخاب کوچک‌ترین مدل موجود برای جلوگیری از OOM
        fallback_model = _router.select_model(
            task="general",
            complexity="simple",
            prefer_small=True  # حتماً مدل کوچک
        )
        state["force_model"] = fallback_model
        state["system_warning"] = (
            f"حالت منابع محدود: {'؛ '.join(health['reasons'])}"
        )
    else:
        # در صورت سلامت سیستم، مدل توسط مسیریاب عادی در model_loader انتخاب می‌شود
        state["force_model"] = None
        state["system_warning"] = None

    state["next_step"] = "call_model"
    return state

# فرض می‌کنیم بقیه گره‌ها (call_model, tool_executor و ...) قبلاً تعریف شده‌اند.
# آن‌ها را از ماژول‌های دیگر وارد می‌کنیم.
from ada.agent.nodes import call_model_node, tool_node, should_continue

# ساخت گراف
workflow = StateGraph(AgentState)

# افزودن گره‌ها
workflow.add_node("pre_check", pre_model_check)
workflow.add_node("call_model", call_model_node)
workflow.add_node("tools", tool_node)

# تنظیم یال‌ها (ترتیب اجرا)
workflow.set_entry_point("pre_check")
workflow.add_edge("pre_check", "call_model")
workflow.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "tools",   # اگر ابزاری فراخوانی شده باشد
        "end": END             # در غیر این صورت
    }
)
workflow.add_edge("tools", "call_model")  # بازگشت به مدل پس از اجرای ابزار

# کامپایل و آماده‌سازی
agent_graph = workflow.compile()
