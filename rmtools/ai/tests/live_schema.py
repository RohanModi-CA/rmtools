from __future__ import annotations

from _support import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    assert_contains,
    assert_mapping,
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
    print_header("Live Schema")
    print(f"provider={PROVIDER}")
    print(f"model={MODEL}")

    ai = make_live_instance(PROVIDER, MODEL)

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
