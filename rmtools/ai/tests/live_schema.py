from __future__ import annotations

from _support import (
    DEFAULT_MODEL,
    assert_contains,
    assert_mapping,
    data_path,
    dump_json,
    make_live_instance,
    print_header,
)


MODEL = DEFAULT_MODEL


def main() -> None:
    print_header("Live Schema")
    print(f"model={MODEL}")

    ai = make_live_instance(MODEL)

    schema_str = data_path("schemas", "basic_schema.json").read_text()

    ai.structured_output(schema_str=schema_str)
    first = assert_mapping(ai.send_message("Return status=ok and token=SCHEMA_OK."))
    assert first["status"].strip().lower() == "ok"
    assert_contains(first["token"].upper(), "SCHEMA_OK")
    dump_json("schema_response_from_string", first)

    ai.structured_output()
    ai.structured_output(schema_filepath=str(data_path("schemas", "basic_schema.json")))
    second = assert_mapping(ai.send_message("Return status=ok and token=SCHEMA_FILE_OK."))
    assert second["status"].strip().lower() == "ok"
    assert_contains(second["token"].upper(), "SCHEMA_FILE_OK")
    dump_json("schema_response_from_file", second)


if __name__ == "__main__":
    main()
