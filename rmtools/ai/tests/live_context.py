from __future__ import annotations

from _support import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    assert_contains,
    assert_nonempty_text,
    default_model_for_provider,
    dump_json,
    make_live_instance,
    print_header,
    temp_workspace,
)


PROVIDER = DEFAULT_PROVIDER
MODEL = DEFAULT_MODEL or default_model_for_provider(PROVIDER)


def main() -> None:
    print_header("Live Context")
    print(f"provider={PROVIDER}")
    print(f"model={MODEL}")

    ai = make_live_instance(PROVIDER, MODEL)
    ai.attach_text("Remember the token CONTEXT_OK.")
    first = assert_nonempty_text(ai.send_message("Acknowledge with the same token."))
    assert_contains(first.upper(), "CONTEXT_OK")
    dump_json("first_response", {"text": first})

    with temp_workspace() as workspace:
        context_path = workspace / "context.json"
        ai.context_save(str(context_path))

        fresh = make_live_instance(PROVIDER, MODEL)
        fresh.context_load(str(context_path))
        followup = assert_nonempty_text(fresh.send_message("What token should still be in memory?"))
        assert_contains(followup.upper(), "CONTEXT_OK")
        dump_json("followup_response", {"text": followup})


if __name__ == "__main__":
    main()
