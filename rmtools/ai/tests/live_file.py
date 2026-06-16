from __future__ import annotations

import os

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
    print_header("Live File")
    print(f"provider={PROVIDER}")
    print(f"model={MODEL}")

    ai = make_live_instance(PROVIDER, MODEL)

    text_file = data_path("text", "attachment.txt")
    ai.attach_file(str(text_file))
    file_response = assert_nonempty_text(ai.send_message("What exact token did the attached file request?"))
    assert_contains(file_response.upper(), "FILE_OK")
    dump_json("text_file_response", {"text": file_response})

    image_path = os.getenv("RMTOOLS_TEST_IMAGE_PATH", "").strip() or str(data_path("images", "image.png"))
    ai.attach_file(image_path)
    image_response = assert_nonempty_text(ai.send_message("Describe this image in one short sentence."))
    dump_json("image_response", {"text": image_response})


if __name__ == "__main__":
    main()
