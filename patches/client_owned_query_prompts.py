from pathlib import Path
import sys


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/lightrag")


def patch(source: Path, changes: list[tuple[str, str, int]]) -> None:
    text = source.read_text()
    for original, replacement, expected in changes:
        found = text.count(original)
        if found != expected:
            raise RuntimeError(
                f"{source}: expected {expected} patch location(s), found {found}"
            )
        text = text.replace(original, replacement)
    source.write_text(text)


patch(
    root / "operate.py",
    [
        (
            '_ANSWER_CACHE_POLICY_VERSION = "query-answer-cache-v2"',
            '_ANSWER_CACHE_POLICY_VERSION = "query-answer-cache-v3-client-system"',
            1,
        ),
        (
            '''    # Build system prompt
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        response_type=response_type,
        user_prompt=effective_user_prompt.slot,
        context_data=context_result.context,
    )

    user_query = query
''',
            '''    # Retrieved material is user content; only the caller supplies system text.
    sys_prompt = system_prompt or ""
    user_query = "\\n\\n".join(
        part
        for part in (
            context_result.context,
            effective_user_prompt.text,
            "---User Query---\\n" + query,
        )
        if part
    )
''',
            1,
        ),
        (
            'prompt_content = "\\n\\n".join([sys_prompt, "---User Query---", user_query])',
            'prompt_content = "\\n\\n".join(part for part in (sys_prompt, user_query) if part)',
            2,
        ),
        (
            'query_prompt_tokens = await acount_tokens(tokenizer, query)',
            'query_prompt_tokens = await acount_tokens(tokenizer, user_query)',
            1,
        ),
        (
            'len_of_prompts = await acount_tokens(tokenizer, query + sys_prompt)',
            'len_of_prompts = await acount_tokens(tokenizer, user_query + sys_prompt)',
            1,
        ),
        (
            '''        # The COMPOSED instructions, so changing the server-side prefix
        # invalidates entries generated under the old one. With no prefix
        # configured this is byte-identical to the previous
        # `query_param.user_prompt or ""`, so existing entries keep hitting --
        # which is why _ANSWER_CACHE_POLICY_VERSION does not need a bump.
        # `disable_user_prompt_prefix` is deliberately NOT a separate key
        # component: it only ever acts through this value, and adding it would
        # split the cache between two requests that build identical prompts.
''',
            '''        # Client instructions and conversation history affect the answer.
''',
            2,
        ),
        (
            '''        effective_user_prompt.text,
        query_param.enable_rerank,
''',
            '''        effective_user_prompt.text,
        system_prompt or "",
        query_param.conversation_history,
        query_param.enable_rerank,
''',
            2,
        ),
        (
            '''        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt,
''',
            '''        response = await use_model_func(
            user_query,
            system_prompt=sys_prompt or None,
''',
            2,
        ),
        (
            '''        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

''',
            "",
            1,
        ),
        (
            '''        if len(response) > len(sys_prompt):
            response = (
                response[len(sys_prompt) :]
                .replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

''',
            "",
            1,
        ),
        (
            '''    # Budget against the template that will ACTUALLY be rendered. `kg_query`
    # picks its template after this function returns, so without the forwarded
    # `system_prompt` the estimate silently used the default one -- charging a
    # caller's custom template at the wrong size, and charging the prefix even
    # when that template has no {user_prompt} placeholder to render it into.
    # `system_prompt_template` is kept as a lower-priority fallback: nothing in
    # this repo writes it, but a downstream user may set it on their own config.
    sys_prompt_template = (
        system_prompt
        or global_config.get("system_prompt_template")
        or PROMPTS["rag_response"]
    )
''',
            '''    # Only client-supplied system text contributes to the token budget.
    sys_prompt_template = system_prompt or ""
''',
            1,
        ),
        (
            '''    pre_sys_prompt = sys_prompt_template.format(
        context_data="",  # Empty for overhead calculation
        response_type=response_type,
        user_prompt=effective_user_prompt.text,
    )
''',
            '''    pre_sys_prompt = sys_prompt_template
''',
            1,
        ),
        (
            '''    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )
''',
            '''    sys_prompt_template = system_prompt or ""
''',
            1,
        ),
        (
            '''    pre_sys_prompt = sys_prompt_template.format(
        response_type=response_type,
        user_prompt=effective_user_prompt.slot,
        content_data="",  # Empty for overhead calculation
    )
''',
            '''    pre_sys_prompt = sys_prompt_template
''',
            1,
        ),
        (
            'query_tokens = await acount_tokens(tokenizer, query)',
            'query_tokens = await acount_tokens(tokenizer, query + effective_user_prompt.text)',
            2,
        ),
        (
            '''    sys_prompt = sys_prompt_template.format(
        response_type=query_param.response_type,
        user_prompt=effective_user_prompt.slot,
        content_data=context_content,
    )

    user_query = query
''',
            '''    sys_prompt = sys_prompt_template
    user_query = "\\n\\n".join(
        part
        for part in (
            context_content,
            effective_user_prompt.text,
            "---User Query---\\n" + query,
        )
        if part
    )
''',
            1,
        ),
    ],
)

patch(
    root / "api/routers/ollama_api.py",
    [
        (
            '''                query_param = QueryParam(**param_dict)

                if request.stream:
''',
            '''                query_param = QueryParam(**param_dict)
                if request.system and not any(
                    message["role"] == "system"
                    and message["content"] == request.system
                    for message in conversation_history
                ):
                    query_param.conversation_history = [
                        {"role": "system", "content": request.system},
                        *conversation_history,
                    ]

                if request.stream:
''',
            1,
        ),
    ],
)
