# graph.py
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, END

# ============================================================
# 1. تعریف ساختار state با TypedDict (استاندارد LangGraph)
# ============================================================
class AgentState(TypedDict):
    """وضعیت جاری عامل (agent) برای تصمیم‌گیری بر اساس منابع سخت‌افزاری"""
    resource_requirement: Literal["gpu", "cpu"]   # نیاز محاسباتی: GPU یا CPU
    gpu_available: bool                            # آیا GPU در دسترس است؟
    memory_mb: int                                 # حافظه (RAM) مورد نیاز (مگابایت)
    available_memory_mb: int                       # حافظه موجود (مگابایت)
    execution_device: str                          # خروجی: "gpu" یا "cpu"
    memory_mode: str                               # خروجی: "high_memory" یا "low_memory"
    chunked_processing: bool                       #是否需要 پردازش تکه‌تکه


# ============================================================
# 2. توابع تصمیم‌گیرنده (مسیریاب‌ها)
# ============================================================
def should_use_gpu(state: AgentState) -> Literal["gpu_node", "cpu_node"]:
    """
    تصمیم می‌گیرد که از GPU استفاده کند یا CPU.
    معیارها: نیاز محاسباتی = "gpu" و وجود GPU در دسترس
    """
    req = state.get("resource_requirement", "cpu")
    gpu_ok = state.get("gpu_available", False)
    if req == "gpu" and gpu_ok:
        return "gpu_node"
    return "cpu_node"


def should_limit_memory(state: AgentState) -> Literal["high_memory", "low_memory"]:
    """
    تصمیم می‌گیرد که حافظه کافی است یا نیاز به محدودیت (پردازش تکه‌تکه) وجود دارد.
    """
    needed = state.get("memory_mb", 0)
    available = state.get("available_memory_mb", 8192)   # پیش‌فرض 8 گیگابایت
    if needed > available:
        return "low_memory"
    return "high_memory"


# ============================================================
# 3. نودهای اجرایی (فعلاً فقط وضعیت را به‌روز می‌کنند)
# ============================================================
def gpu_node(state: AgentState) -> AgentState:
    """نود مختص پردازش با GPU"""
    state["execution_device"] = "gpu"
    # در آینده می‌توانید کد واقعی GPU را اینجا قرار دهید
    return state


def cpu_node(state: AgentState) -> AgentState:
    """نود مختص پردازش با CPU"""
    state["execution_device"] = "cpu"
    return state


def high_memory_node(state: AgentState) -> AgentState:
    """نود برای حالتی که حافظه کافی وجود دارد"""
    state["memory_mode"] = "high_memory"
    state["chunked_processing"] = False
    return state


def low_memory_node(state: AgentState) -> AgentState:
    """نود برای حالتی که حافظه محدود است – فعال‌سازی پردازش تکه‌تکه"""
    state["memory_mode"] = "low_memory"
    state["chunked_processing"] = True
    return state


# ============================================================
# 4. ساخت گراف اصلی (با نقطه ورود صحیح)
# ============================================================
def build_agent_graph() -> StateGraph:
    """
    ایجاد و پیکربندی گراف تصمیم‌گیری LangGraph.
    مسیر: router -> تصمیم GPU/CPU -> تصمیم حافظه -> END
    """
    workflow = StateGraph(AgentState)

    # اضافه کردن همه نودها
    workflow.add_node("router", lambda state: state)          # نود عبوری ساده
    workflow.add_node("gpu_node", gpu_node)
    workflow.add_node("cpu_node", cpu_node)
    workflow.add_node("high_memory_node", high_memory_node)
    workflow.add_node("low_memory_node", low_memory_node)

    # نقطه ورود: نود router
    workflow.set_entry_point("router")

    # یال شرطی از router: تصمیم‌گیری GPU یا CPU
    workflow.add_conditional_edges(
        "router",
        should_use_gpu,
        {
            "gpu_node": "gpu_node",
            "cpu_node": "cpu_node"
        }
    )

    # یال شرطی از هر دو نود gpu_node و cpu_node به سمت مدیریت حافظه
    workflow.add_conditional_edges(
        "gpu_node",
        should_limit_memory,
        {
            "high_memory": "high_memory_node",
            "low_memory": "low_memory_node"
        }
    )
    workflow.add_conditional_edges(
        "cpu_node",
        should_limit_memory,
        {
            "high_memory": "high_memory_node",
            "low_memory": "low_memory_node"
        }
    )

    # پس از نودهای حافظه، گراف پایان می‌یابد
    workflow.add_edge("high_memory_node", END)
    workflow.add_edge("low_memory_node", END)

    return workflow


# ============================================================
# 5. (اختیاری) تابع کمکی برای اجرای آسان گراف
# ============================================================
def run_agent(
    resource_requirement: Literal["gpu", "cpu"] = "cpu",
    gpu_available: bool = False,
    memory_mb: int = 0,
    available_memory_mb: int = 8192
) -> AgentState:
    """
    اجرای گراف با مقادیر ورودی مشخص و بازگرداندن وضعیت نهایی.
    (برای تست و نمونه استفاده)
    """
    initial_state: AgentState = {
        "resource_requirement": resource_requirement,
        "gpu_available": gpu_available,
        "memory_mb": memory_mb,
        "available_memory_mb": available_memory_mb,
        "execution_device": "",
        "memory_mode": "",
        "chunked_processing": False
    }
    graph = build_agent_graph()
    app = graph.compile()
    final_state = app.invoke(initial_state)
    return final_state


# ============================================================
# 6. مثال اجرا (در صورت اجرای مستقیم فایل)
# ============================================================
if __name__ == "__main__":
    # تست سناریو 1: نیاز به GPU، GPU موجود نیست ← باید CPU انتخاب شود
    result1 = run_agent(resource_requirement="gpu", gpu_available=False, memory_mb=1024)
    print("Scenario 1 (no GPU):", result1)

    # تست سناریو 2: نیاز به GPU، GPU موجود است و حافظه کافی
    result2 = run_agent(resource_requirement="gpu", gpu_available=True, memory_mb=512, available_memory_mb=4096)
    print("Scenario 2 (GPU, enough memory):", result2)

    # تست سناریو 3: نیاز به GPU، GPU موجود اما حافظه ناکافی
    result3 = run_agent(resource_requirement="gpu", gpu_available=True, memory_mb=5000, available_memory_mb=4096)
    print("Scenario 3 (GPU, low memory):", result3)
