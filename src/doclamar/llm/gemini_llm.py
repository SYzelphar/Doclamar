import time
import threading
from google import genai
from doclamar.llm.base import BaseLLM

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        # --- Multi-Agent Synchronization ---
        self._lock = threading.Lock()  # Prevents agents from stepping on each other
        self.last_call_time = 0.0
        self.min_gap = 6.0  # 6 seconds = 10 requests per minute (very safe for 4 agents)

    def generate(self, prompt: str) -> str:
        # 'with self._lock' makes the agents wait their turn in a queue
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call_time
            
            if elapsed < self.min_gap:
                wait_time = self.min_gap - elapsed
                print(f"🤖 Agent Queue: Waiting {wait_time:.2f}s before next request...")
                time.sleep(wait_time)

            # Actual API Call
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            # Update timestamp ONLY after a successful call
            self.last_call_time = time.time()
            return response.text.strip()