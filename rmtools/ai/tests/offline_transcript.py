from __future__ import annotations

from pathlib import Path

from _support import (
    assert_contains,
    data_path,
    dump_json,
    make_offline_instance,
    print_header,
    temp_workspace,
)


def main() -> None:
    print_header("Offline Transcript")

    ai = make_offline_instance()
    ai.attach_text("Reply with the exact token OFFLINE_INLINE_OK.")
    ai.attach_text(text_filepath=str(data_path("text", "attachment.txt")))
    ai.attach_file(str(data_path("text", "attachment.txt")))

    if len(ai.transcript) != 3:
        raise AssertionError(f"Expected 3 transcript entries, found {len(ai.transcript)}")

    if ai.transcript[0]["role"] != "user" or ai.transcript[0]["parts"][0]["kind"] != "text":
        raise AssertionError("First transcript entry should be a user text message.")
    if ai.transcript[2]["parts"][0]["kind"] != "file":
        raise AssertionError("Third transcript entry should be a file attachment.")

    first = ai.transcript[0]["parts"][0]["text"]
    assert_contains(first.upper(), "OFFLINE_INLINE_OK")

    dump_json("transcript_before_save", ai.transcript)

    with temp_workspace() as workspace:
        context_path = workspace / "context.json"
        ai.structured_output(schema_filepath=str(data_path("schemas", "basic_schema.json")))
        ai.context_save(str(context_path))

        restored = make_offline_instance()
        restored.context_load(str(context_path))

        if restored.transcript != ai.transcript:
            raise AssertionError("Transcript did not round-trip through context_save/context_load.")
        if restored.config != ai.config:
            raise AssertionError("Structured output config did not round-trip.")
        dump_json("restored_context", {"transcript": restored.transcript, "config": restored.config})


if __name__ == "__main__":
    main()
