import os
from dotenv import load_dotenv
import httpx
import asyncio
import json
from typing import List, Dict
from pathlib import Path
load_dotenv()
APPLICATION_MODEL = os.getenv("APPLICATION_MODEL", "ai/qwen3:0.6B-Q4_0")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:12434/engines/llama.cpp/v1/chat/completions")


class ContextManager:
    def __init__(self, storage_path: str | None = None):
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self.HISTORY_MAX_CHARS = 6000
        self.SUMMARIZE_THRESHOLD = 4000

        # persistence
        self.storage_path = Path(
            storage_path or os.getenv(
                "CONTEXT_STORAGE_PATH", "data/context.json"
            )
        )
        self._ensure_storage_dir()
        # load existing histories (best-effort)
        try:
            self._load_from_disk()
        except Exception:
            # ignore loading errors to avoid crashing server on malformed file
            pass

    def _ensure_storage_dir(self):
        if not self.storage_path.parent.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def get_history(self, user_id: str) -> List[Dict[str, str]]:
        return self._histories.setdefault(user_id, [])

    async def append_message(self, user_id: str, role: str, content: str):
        lock = self._get_lock(user_id)
        async with lock:
            hist = self.get_history(user_id)
            hist.append({"role": role, "content": content})
            await self._save_to_disk_async()

    def _history_size(self, history: List[Dict[str, str]]) -> int:
        return sum(len(m.get("content", "")) for m in history)

    def slice_for_context(
            self, history: List[Dict[str, str]], max_chars: int
    ) -> List[Dict[str, str]]:
        """
        Return the most recent messages 
        that fit into max_chars (approximation).
        """
        out: List[Dict[str, str]] = []
        total = 0
        for m in reversed(history):
            c = len(m.get("content", ""))
            if total + c > max_chars and out:
                break
            out.insert(0, m)
            total += c
        return out

    async def ensure_summary(self, user_id: str):
        """
        If user's history exceeds SUMMARIZE_THRESHOLD, summarize older messages
        and replace them with a compact summary message.
        """
        lock = self._get_lock(user_id)
        async with lock:
            history = self.get_history(user_id)
            size = self._history_size(history)
            if size < self.SUMMARIZE_THRESHOLD:
                return      
            recent_window = self.slice_for_context(
                history, int(self.HISTORY_MAX_CHARS * 0.4)
            )

            idx = max(0, len(history) - len(recent_window))
            to_summarize = history[:idx]
            if not to_summarize:
                return

            # build summarization prompt
            convo_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in to_summarize])
            summary_prompt = (
                "You are a concise summarizer. Produce a short summary"
                "(2-6 sentences or bullet points) "
                "of the conversation below, preserving relevant facts,"
                "user preferences and topic context. "
                "Do not add new information.\n\n"
                "Conversation:\n" + convo_text
            )

            # call LLaMA to summarize
            async with httpx.AsyncClient(timeout=60.0) as client:
                req_json = {
                    "model": APPLICATION_MODEL,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You summarize conversations concisely."
                        },
                        {
                            "role": "user",
                            "content": summary_prompt
                        }
                    ]
                }
                resp = await client.post(
                    LLAMA_API_URL, json=req_json, headers={
                        "Content-Type": "application/json"
                    }
                )
                resp.raise_for_status()
                resp_json = resp.json()

            summary_text = ""
            try:
                summary_text = resp_json.get("choices", [])[0].get("message", {}).get("content", "")
            except Exception:
                # fallback: convert top-level fields if different shape
                summary_text = str(resp_json)

            # replace older messages with a single summary message
            new_history = [{"role": "system", "content": "Conversation summary: " + summary_text}] + recent_window
            self._histories[user_id] = new_history
            # persist after summarization
            await self._save_to_disk_async()

    # Persistence helpers
    def _load_from_disk(self):
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # validate shape and assign
            if isinstance(data, dict):
                # ensure values are lists of dicts with role/content
                cleaned: Dict[str, List[Dict[str, str]]] = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        cleaned[k] = [m for m in v if isinstance(m, dict) and "role" in m and "content" in m]
                self._histories = cleaned
        except Exception:
            # ignore errors (corrupt file) to avoid startup failure
            return

    def _save_to_disk_sync(self):
        tmp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._histories, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.storage_path)

    async def _save_to_disk_async(self):
        loop = asyncio.get_running_loop()
        # perform blocking write in threadpool
        await loop.run_in_executor(None, self._save_to_disk_sync)
