def nameof(fstring: str) -> str:
    name, _ = fstring.split('=', 1)
    return name
