import tkinter as tk
import customtkinter as ctk
import threading
import queue
from typing import Dict, Any, Optional

class AgentGUI(ctk.CTk):
    def __init__(self, model_orchestrator):
        super().__init__()
        self.model_orchestrator = model_orchestrator
        self.lock = threading.RLock()
        self.task_queue = queue.Queue()
        self.current_task_id = None
        self.current_task_thread = None
        self.cancel_flag = threading.Event()
        self.setup_ui()
        self.process_task_queue()

    def setup_ui(self):
        self.title("Agent System")
        self.geometry("800x600")

        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Model status display
        self.model_status_label = ctk.CTkLabel(self.main_frame, text="Models: Loading...")
        self.model_status_label.pack(pady=5)

        # Action button frame
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        self.run_button = ctk.CTkButton(self.button_frame, text="Run Long Task", command=self.start_long_task)
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ctk.CTkButton(self.button_frame, text="Cancel", command=self.cancel_long_task, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Output text area
        self.output_text = ctk.CTkTextbox(self.main_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True, pady=5)

        self.update_model_status()

    def update_model_status(self):
        """Thread-safe update of model status display"""
        with self.lock:
            if hasattr(self.model_orchestrator, 'loaded_models'):
                models = self.model_orchestrator.loaded_models
                status = f"Loaded models: {', '.join(models.keys()) if models else 'None'}"
            else:
                status = "Model status unavailable"
        self.model_status_label.configure(text=status)
        self.after(2000, self.update_model_status)

    def start_long_task(self):
        """Start a long-running task in a background thread"""
        with self.lock:
            if self.current_task_thread and self.current_task_thread.is_alive():
                self.log_message("A task is already running. Cancel it first.")
                return

        self.cancel_flag.clear()
        self.current_task_id = id(threading.current_thread())
        self.run_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self.log_message("Starting long task...")

        self.current_task_thread = threading.Thread(target=self.long_running_operation, daemon=True)
        self.current_task_thread.start()

    def long_running_operation(self):
        """Example long operation that accesses shared resources"""
        try:
            # Simulate work with cancellation check
            for step in range(10):
                if self.cancel_flag.is_set():
                    self.queue_log_message("Task cancelled by user.")
                    return

                # Access shared resource with lock
                with self.lock:
                    # Simulate reading loaded_models
                    if hasattr(self.model_orchestrator, 'loaded_models'):
                        model_count = len(self.model_orchestrator.loaded_models)
                    else:
                        model_count = 0
                    self.queue_log_message(f"Step {step+1}/10 - Models available: {model_count}")

                # Simulate heavy work
                for _ in range(5):
                    if self.cancel_flag.is_set():
                        return
                    import time
                    time.sleep(0.2)

            self.queue_log_message("Long task completed successfully.")
        except Exception as e:
            self.queue_log_message(f"Error in long task: {str(e)}")
        finally:
            self.queue_task_finished()

    def cancel_long_task(self):
        """Cancel the currently running long task"""
        with self.lock:
            if not self.current_task_thread or not self.current_task_thread.is_alive():
                self.log_message("No running task to cancel.")
                self.cancel_button.configure(state=tk.DISABLED)
                return

        self.log_message("Cancelling task...")
        self.cancel_flag.set()
        # Note: The thread will check cancel_flag and exit
        self.cancel_button.configure(state=tk.DISABLED)

    def queue_log_message(self, message: str):
        """Queue a message to be displayed from the worker thread"""
        self.task_queue.put(('log', message))

    def queue_task_finished(self):
        """Queue notification that the task has finished"""
        self.task_queue.put(('finish', None))

    def process_task_queue(self):
        """Process queued GUI updates from background threads (thread-safe)"""
        try:
            while True:
                msg_type, data = self.task_queue.get_nowait()
                if msg_type == 'log':
                    self.log_message(data)
                elif msg_type == 'finish':
                    with self.lock:
                        self.current_task_thread = None
                        self.current_task_id = None
                    self.run_button.configure(state=tk.NORMAL)
                    self.cancel_button.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_task_queue)

    def log_message(self, message: str):
        """Log a message to the output text widget"""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)

    # Example of a deprecated method warning fix: In customtkinter, CTkButton's 'state' parameter
    # is preferred over 'disabled' keyword in older versions. We use 'state' consistently.
    # Additionally, use CTkTextbox instead of deprecated CTkTextbox (if any).

    # Thread-safe method to access loaded_models from outside
    def get_loaded_models_safe(self):
        """Return a thread-safe copy of loaded_models"""
        with self.lock:
            if hasattr(self.model_orchestrator, 'loaded_models'):
                return dict(self.model_orchestrator.loaded_models)
            return {}

    # Thread-safe method to update loaded_models
    def update_loaded_models_safe(self, model_id: str, model_data: Any):
        """Thread-safe update of loaded_models"""
        with self.lock:
            if hasattr(self.model_orchestrator, 'loaded_models'):
                self.model_orchestrator.loaded_models[model_id] = model_data

if __name__ == "__main__":
    # Example placeholder orchestrator
    class MockOrchestrator:
        def __init__(self):
            self.loaded_models = {}

    orchestrator = MockOrchestrator()
    app = AgentGUI(orchestrator)
    app.mainloop()