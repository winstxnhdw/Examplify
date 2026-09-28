from pathlib import Path


def replace_exact(source: Path, original: str, replacement: str, count: int = 1) -> None:
    text = source.read_text()
    if text.count(original) != count:
        raise RuntimeError(f"{source}: expected {count} patch location(s); review upstream changes")
    source.write_text(text.replace(original, replacement))


source = Path("/app/lightrag/llm/openai.py")
original = """    # Prepare messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
"""
replacement = """    # Keep all system instructions in one leading message. Some chat templates
    # reject a second system message in conversation history.
    messages: list[dict[str, Any]] = []
    system_parts = [system_prompt] if system_prompt else []
    system_parts.extend(
        message["content"]
        for message in history_messages
        if message.get("role") == "system"
    )
    if system_parts:
        messages.append({"role": "system", "content": "\\n\\n".join(system_parts)})
    messages.extend(
        message for message in history_messages if message.get("role") != "system"
    )
"""

replace_exact(source, original, replacement)
