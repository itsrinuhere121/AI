import httpx
from model.ChatRequest import QueryRequest
from model.context_manager import ContextManager, APPLICATION_MODEL, LLAMA_API_URL
context_manager = ContextManager()


async def chatRequest(req: QueryRequest):
    """
    Handle a QueryRequest by building context, optionally summarizing old history,
    forwarding to llama.cpp endpoint, and storing the assistant reply.
    Expects QueryRequest to have at least `.prompt` and optionally `.user_id`.
    """
    user_id = getattr(req, "user_id", None) or "default"
    await context_manager.append_message(user_id, "user", req.prompt)
    await context_manager.ensure_summary(user_id)

    # build messages to send: keep system + recent messages (approx by chars)
    history = context_manager.get_history(user_id)
    recent_msgs = context_manager.slice_for_context(
        history, context_manager.HISTORY_MAX_CHARS
    )

    # always include a default system instruction at the front
    system_msg = {"role": "system", "content": "You are a helpful assistant. Use the conversation context to answer."}
    outgoing_messages = [system_msg] + recent_msgs

    chat_req = {
        "model": APPLICATION_MODEL,
        "messages": outgoing_messages
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            LLAMA_API_URL,
            json=chat_req,
            headers={
                "Content-Type": "application/json"
            }
        )
        resp.raise_for_status()
        resp_json = resp.json()

    # extract assistant reply (best-effort)
    assistant_content = ""
    try:
        assistant_content = resp_json.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception:
        assistant_content = str(resp_json)

    # append assistant reply to history
    await context_manager.append_message(
        user_id, "assistant", assistant_content
    )

    return resp_json
