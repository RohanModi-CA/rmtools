# rmtools Test Harness

This folder is a small executable harness, not a pytest suite.

## Live scripts

Default target:

- `OPENROUTER_API_KEY`
- `RMTOOLS_TEST_MODEL=openai/gpt-4o-mini`

Optional:

- `RMTOOLS_TEST_IMAGE_PATH=/path/to/local/image.png`

Scripts:

- `python tests/live_text.py`
- `python tests/live_file.py`
- `python tests/live_schema.py`
- `python tests/live_context.py`
- `python tests/live_smoke.py`

## Offline scripts

- `python tests/offline_transcript.py`
- `python tests/offline_serialization.py`
- `python tests/offline_auth.py`

## Data files

- `tests/data/prompts/basic_prompt.txt`
- `tests/data/text/attachment.txt`
- `tests/data/schemas/basic_schema.json`

These files are deliberately short so the scripts can show how prompt loading,
text attachment, and schema loading are supposed to work.

For image input, set `RMTOOLS_TEST_IMAGE_PATH` to a real local image file
(`png`, `jpeg`, or `webp`). The harness will skip that step if the variable is
unset.
