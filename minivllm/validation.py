def check_prompt_fits(prompt_token_ids: list[int], max_model_len: int) -> None:
    if not prompt_token_ids:
        raise ValueError("prompt must contain at least one token")
    if len(prompt_token_ids) >= max_model_len:
        raise ValueError(
            f"prompt has {len(prompt_token_ids)} tokens; max_model_len is {max_model_len}, "
            "which leaves no room to generate"
        )
