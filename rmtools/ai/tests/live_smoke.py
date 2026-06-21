from __future__ import annotations

import os

from _support import (
    DEFAULT_MODEL,
    assert_contains,
    assert_mapping,
    assert_nonempty_text,
    data_path,
    dump_json,
    make_live_instance,
    print_header,
    temp_workspace,
)


MODEL = DEFAULT_MODEL


def main() -> None:
    print_header("Live Smoke")
    print(f"model={MODEL}")

    ai = make_live_instance(MODEL)

    ai.attach_text("Reply with the exact token INLINE_OK.")
    inline_response = assert_nonempty_text(ai.send_message("What token should you reply with?"))
    assert_contains(inline_response.upper(), "INLINE_OK")

    ai.attach_text(text_filepath=str(data_path("text", "attachment.txt")))
    file_response = assert_nonempty_text(ai.send_message("What exact token did the file request?"))
    assert_contains(file_response.upper(), "FILE_OK")

    ai.load_prompt("basic_prompt", prompts_dir_path=str(data_path("prompts")))
    prompt_response = assert_nonempty_text(ai.send_message("What token did the prompt request?"))
    assert_contains(prompt_response.upper(), "PROMPT_OK")

    ai.structured_output(schema_filepath=str(data_path("schemas", "basic_schema.json")))
    structured = assert_mapping(ai.send_message("Return status=ok and token=SMOKE_OK."))
    assert structured["status"].strip().lower() == "ok"
    assert_contains(structured["token"].upper(), "SMOKE_OK")

    ai.structured_output()

    ai.attach_text("Remember the token CONTEXT_OK.")
    context_response = assert_nonempty_text(ai.send_message("Acknowledge with CONTEXT_OK."))
    assert_contains(context_response.upper(), "CONTEXT_OK")

    image_path = os.getenv("RMTOOLS_TEST_IMAGE_PATH", "").strip() or str(data_path("images", "image.png"))
    ai.attach_file(image_path)
    image_response = assert_nonempty_text(ai.send_message("Describe this image in one short sentence."))
    dump_json("image_response", {"text": image_response})

    with temp_workspace() as workspace:
        context_path = workspace / "context.json"
        ai.context_save(str(context_path))

        fresh = make_live_instance(MODEL)
        fresh.context_load(str(context_path))
        followup = assert_nonempty_text(fresh.send_message("What token should still be in memory?"))
        assert_contains(followup.upper(), "CONTEXT_OK")

    dump_json(
        "transcript_summary",
        {
            "messages": len(ai.transcript),
            "model": MODEL,
        },
    )


if __name__ == "__main__":
    main()
