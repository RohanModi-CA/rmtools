from __future__ import annotations

from _support import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    assert_contains,
    assert_nonempty_text,
    default_model_for_provider,
    data_path,
    dump_json,
    make_live_instance,
    print_header,
)


PROVIDER = DEFAULT_PROVIDER
MODEL = DEFAULT_MODEL or default_model_for_provider(PROVIDER)


def main() -> None:
    print_header("Live Text")
    print(f"provider={PROVIDER}")
    print(f"model={MODEL}")

    ai = make_live_instance(PROVIDER, MODEL)

    ai.attach_text("Reply with the exact token INLINE_OK.")
    inline_response = assert_nonempty_text(ai.send_message("What token should you reply with?"))
    assert_contains(inline_response.upper(), "INLINE_OK")
    dump_json("inline_response", {"text": inline_response})

    ai.attach_text(text_filepath=str(data_path("text", "attachment.txt")))
    file_response = assert_nonempty_text(ai.send_message("What exact token did the file request?"))
    assert_contains(file_response.upper(), "FILE_OK")
    dump_json("file_response", {"text": file_response})

    ai.load_prompt("basic_prompt", prompts_dir_path=str(data_path("prompts")))
    prompt_response = assert_nonempty_text(ai.send_message("What token did the prompt request?"))
    assert_contains(prompt_response.upper(), "PROMPT_OK")
    dump_json("prompt_response", {"text": prompt_response})


if __name__ == "__main__":
    main()
