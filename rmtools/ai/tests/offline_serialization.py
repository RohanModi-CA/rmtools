from __future__ import annotations

from _support import (
    data_path,
    dump_json,
    make_offline_instance,
    print_header,
)


def main() -> None:
    print_header("Offline Serialization")

    ai = make_offline_instance()
    ai.attach_text("Hello from offline serialization.")
    ai.attach_file(str(data_path("text", "attachment.txt")))
    ai.structured_output(schema_filepath=str(data_path("schemas", "basic_schema.json")))

    transcript = ai.transcript
    gemini_contents = ai._gemini_contents(transcript)
    openrouter_messages = ai._openrouter_messages(transcript)
    openrouter_response_format = ai._openrouter_response_format()

    if not gemini_contents or not openrouter_messages:
        raise AssertionError("Expected non-empty serialized outputs.")

    if openrouter_response_format is None:
        raise AssertionError("Expected an OpenRouter response format for structured output.")

    if gemini_contents[0]["parts"][0].get("text") != "Hello from offline serialization.":
        raise AssertionError("Gemini serialization did not preserve the user text.")
    if openrouter_messages[0]["content"] != "Hello from offline serialization.":
        raise AssertionError("OpenRouter serialization did not preserve the user text.")

    dump_json("gemini_contents", gemini_contents)
    dump_json("openrouter_messages", openrouter_messages)
    dump_json("openrouter_response_format", openrouter_response_format)

    ai.structured_output()
    if ai.config:
        raise AssertionError("Expected structured output config to clear when called without a schema.")


if __name__ == "__main__":
    main()
