from google import genai
from typing import Any
import base64
import json
import google.genai.errors
from google.genai import types
import os
import warnings
from dotenv import load_dotenv, find_dotenv
from urllib import error as urllib_error
from urllib import request as urllib_request


"""
filter the warning. I think it's through logging but let's leave the warning one as well.
"""
thought_signature_warning: str = "there are non-text parts in the response: ['thought_signature']"
warnings.filterwarnings('ignore', thought_signature_warning)
import logging


class SuppressGenAINonTextWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "there are non-text parts in the response" not in record.getMessage()


logging.getLogger("google_genai.types").addFilter(SuppressGenAINonTextWarning())


class AIRequestError(RuntimeError):
    """Normalizes HTTP failures so retries do not need provider-specific parsing."""

    def __init__(self, message: str, status_code: int = 0, retry_after_ms: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_ms = retry_after_ms
        self.body = body

    def __str__(self) -> str:
        if not self.body:
            return super().__str__()

        return f"{super().__str__()}\n{self.body}"


class AI_Instance:
    def _model_selector(self, model: str = "", client_mode: str = "gemini") -> str:
        """Pick a default model without making callers think about backend details."""
        if client_mode == "openrouter":
            default_model: str = "openai/gpt-4o-mini"
        else:
            default_model: str = "gemini-2.5-flash"

        if not model:
            print(f"rmAI: No Model Specified, Defaulting to {default_model}")
            model = default_model

        return model

    def _env_truthy(self, env_var_name: str) -> bool:
        return os.getenv(env_var_name, "").lower() in ("true", "1")

    def _thinking_budget_from_level(self, thinking_level: float) -> int:
        if not 0.0 <= thinking_level <= 1.0:
            raise ValueError("rmAI: thinking must be between 0 and 1.")

        model_name = self.model.lower()

        if "flash-lite" in model_name:
            min_budget, max_budget = 0, 24576
        elif "flash" in model_name:
            min_budget, max_budget = 0, 24576
        else:
            min_budget, max_budget = 128, 32768

        if thinking_level <= 0.0:
            return min_budget

        if thinking_level >= 1.0:
            return max_budget

        return round(min_budget + (max_budget - min_budget) * thinking_level)

    def _build_config(self) -> types.GenerateContentConfig | None:
        config_kwargs = dict(self.config)

        if self.thinking is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._thinking_budget_from_level(self.thinking)
            )

        if not config_kwargs:
            return None

        return types.GenerateContentConfig(**config_kwargs)

    def _resolve_auth(self, api_key: str = "", vertex_api_key: str = "", openrouter_api_key: str = "") -> str:
        if sum(bool(val) for val in (api_key, vertex_api_key, openrouter_api_key)) > 1:
            raise ValueError("rmAI: Set only one of api_key, vertex_api_key, or openrouter_api_key.")

        if openrouter_api_key:
            self._api_key = ""
            self._vertex_api_key = ""
            self._openrouter_api_key = openrouter_api_key
            return "openrouter"

        if vertex_api_key:
            self._api_key = ""
            self._vertex_api_key = vertex_api_key
            self._openrouter_api_key = ""
            return "vertex"

        if api_key:
            self._api_key = api_key
            self._vertex_api_key = ""
            self._openrouter_api_key = ""
            return "gemini"

        load_dotenv(find_dotenv(usecwd=True))

        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self._vertex_api_key = os.getenv("GOOGLE_API_KEY") or ""
        self._openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or ""

        if self._openrouter_api_key:
            return "openrouter"

        if self._vertex_api_key:
            return "vertex"

        if self._api_key:
            return "gemini"

        raise ValueError(
            "rmAI: No API key found. Set api_key, vertex_api_key, openrouter_api_key, GEMINI_API_KEY, GOOGLE_API_KEY, or OPENROUTER_API_KEY."
        )

    def _create_client(self, api_key: str = "", vertex_api_key: str = "", openrouter_api_key: str = "") -> tuple[Any | None, str]:
        client_mode = self._resolve_auth(api_key=api_key, vertex_api_key=vertex_api_key, openrouter_api_key=openrouter_api_key)

        if client_mode == "vertex":
            return genai.Client(vertexai=True, api_key=self._vertex_api_key), client_mode

        if client_mode == "gemini":
            return genai.Client(api_key=self._api_key), client_mode

        return None, client_mode

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        vertex_api_key: str = "",
        thinking: float | None = None,
        openrouter_api_key: str = "",
    ):
        self.client, self._client_mode = self._create_client(
            api_key=api_key,
            vertex_api_key=vertex_api_key,
            openrouter_api_key=openrouter_api_key,
        )
        self.model: str = self._model_selector(model, client_mode=self._client_mode)

        self.config: dict[str, Any] = {}
        self.thinking: float | None = None
        self.transcript: list[dict[str, Any]] = []
        self._attached_file_uri_paths: dict[str, str] = {}
        self.set_thinking(thinking)

    def _text_part(self, text: str) -> dict[str, Any]:
        return {"kind": "text", "text": text}

    def _file_part(self, pathtofile: str) -> dict[str, Any]:
        with open(pathtofile, "rb") as file:
            data = file.read()

        mime_type = self._infer_mime_type(pathtofile)
        return {
            "kind": "file",
            "filename": os.path.basename(pathtofile),
            "mime_type": mime_type,
            "data_b64": base64.b64encode(data).decode("ascii"),
        }

    def _append_message(self, transcript: list[dict[str, Any]], role: str, parts: list[dict[str, Any]]) -> None:
        transcript.append({"role": role, "parts": parts})

    def _pending_transcript(self, message: str = "") -> list[dict[str, Any]]:
        transcript = [dict(entry) for entry in self.transcript]
        if message:
            self._append_message(transcript, "user", [self._text_part(message)])
        return transcript

    def _retry_after_ms_from_headers(self, headers: Any) -> int:
        retry_after = ""
        if headers:
            retry_after = headers.get("Retry-After", "") or headers.get("retry-after", "")

        if not retry_after:
            return 0

        try:
            return int(float(retry_after) * 1000)
        except ValueError:
            return 0

    def _http_json_request(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        request = urllib_request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retry_after_ms = self._retry_after_ms_from_headers(exc.headers)
            raise AIRequestError(
                f"rmAI: HTTP request failed with status {exc.code}",
                status_code=exc.code,
                retry_after_ms=retry_after_ms,
                body=body,
            ) from exc

    def _gemini_contents(self, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for entry in transcript:
            parts: list[dict[str, Any]] = []
            for part in entry["parts"]:
                if part["kind"] == "text":
                    parts.append({"text": part["text"]})
                elif part["kind"] == "file":
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": part["mime_type"],
                                "data": part["data_b64"],
                            }
                        }
                    )
                else:
                    raise ValueError("rmAI: Unknown transcript part kind.")
            role = "model" if entry["role"] == "assistant" else entry["role"]
            contents.append({"role": role, "parts": parts})
        return contents

    def _openrouter_messages(self, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for entry in transcript:
            content_parts: list[dict[str, Any]] = []
            text_chunks: list[str] = []
            for part in entry["parts"]:
                if part["kind"] == "text":
                    text_chunks.append(part["text"])
                elif part["kind"] == "file":
                    if part["mime_type"].startswith("text/"):
                        text_chunks.append(base64.b64decode(part["data_b64"]).decode("utf-8", errors="replace"))
                    elif part["mime_type"].startswith("image/"):
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{part['mime_type']};base64,{part['data_b64']}"
                                },
                            }
                        )
                    else:
                        raise NotImplementedError("rmAI: OpenRouter file attachments are only supported for text/* and image/* files.")
                else:
                    raise ValueError("rmAI: Unknown transcript part kind.")

            if content_parts:
                if text_chunks:
                    content_parts.insert(0, {"type": "text", "text": "\n".join(text_chunks)})
                content: Any = content_parts
            else:
                content = "\n".join(text_chunks)

            messages.append({"role": entry["role"], "content": content})
        return messages

    def _vertex_generation_config(self) -> types.GenerateContentConfig | None:
        config_kwargs: dict[str, Any] = {}

        if self.thinking is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._thinking_budget_from_level(self.thinking)
            )

        if self.config:
            config_kwargs["response_mime_type"] = self.config.get("response_mime_type")
            if "response_json_schema" in self.config:
                config_kwargs["response_json_schema"] = self.config["response_json_schema"]

        if not config_kwargs:
            return None

        return types.GenerateContentConfig(**config_kwargs)

    def _openrouter_response_format(self) -> dict[str, Any] | None:
        if not self.config:
            return None

        schema = self.config.get("response_json_schema")
        if not schema:
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": "rmtools",
                "strict": True,
                "schema": schema,
            },
        }

    def _extract_response_text(self, response: Any) -> str:
        if hasattr(response, "text") and getattr(response, "text") is not None:
            return response.text

        if isinstance(response, dict):
            if "choices" in response:
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                if isinstance(content, list):
                    text_bits = [part.get("text", "") for part in content if isinstance(part, dict)]
                    return "".join(text_bits)
                return content or ""

            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                text_bits = [part.get("text", "") for part in parts if isinstance(part, dict)]
                if text_bits:
                    return "".join(text_bits)

        raise ValueError("rmAI: Could not read model response text.")

    def _commit_response(self, transcript: list[dict[str, Any]], response_text: str) -> Any:
        committed = [dict(entry) for entry in transcript]
        self._append_message(committed, "assistant", [self._text_part(response_text)])
        self.transcript = committed

        if self.config:
            try:
                return dict(json.loads(response_text))
            except json.JSONDecodeError as exc:
                raise AIRequestError("rmAI: Invalid model output.", body=response_text) from exc
        return response_text

    def _send_gemini_http_message(self, transcript: list[dict[str, Any]]) -> Any:
        payload: dict[str, Any] = {"contents": self._gemini_contents(transcript)}

        generation_config = self._vertex_generation_config()
        if generation_config:
            config_payload = generation_config.model_dump(exclude_none=True)
            payload["generationConfig"] = {}
            if config_payload.get("response_mime_type"):
                payload["generationConfig"]["responseMimeType"] = config_payload["response_mime_type"]
            if config_payload.get("response_json_schema"):
                payload["generationConfig"]["responseSchema"] = config_payload["response_json_schema"]
            if config_payload.get("thinking_config"):
                payload["generationConfig"]["thinkingConfig"] = config_payload["thinking_config"]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self._api_key}"
        response = self._http_json_request(url, payload, {"Content-Type": "application/json"})
        return self._extract_response_text(response)

    def _send_vertex_sdk_message(self, transcript: list[dict[str, Any]]) -> Any:
        contents: list[Any] = []
        for entry in transcript:
            parts: list[Any] = []
            for part in entry["parts"]:
                if part["kind"] == "text":
                    parts.append(types.Part(text=part["text"]))
                elif part["kind"] == "file":
                    parts.append(types.Part.from_bytes(data=base64.b64decode(part["data_b64"]), mime_type=part["mime_type"]))
            role = "model" if entry["role"] == "assistant" else entry["role"]
            contents.append(types.Content(role=role, parts=parts))

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=self._vertex_generation_config(),
        )
        return self._extract_response_text(response)

    def _send_openrouter_message(self, transcript: list[dict[str, Any]]) -> Any:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._openrouter_messages(transcript),
        }

        response_format = self._openrouter_response_format()
        if response_format:
            payload["response_format"] = response_format

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "Content-Type": "application/json",
        }

        http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if app_title:
            headers["X-Title"] = app_title

        response = self._http_json_request("https://openrouter.ai/api/v1/chat/completions", payload, headers)
        return self._extract_response_text(response)

    def _send_message(self, message: str = "") -> Any:
        transcript = self._pending_transcript(message)

        if self._client_mode == "openrouter":
            response_text = self._send_openrouter_message(transcript)
        elif self._client_mode == "vertex":
            response_text = self._send_vertex_sdk_message(transcript)
        else:
            response_text = self._send_gemini_http_message(transcript)

        return self._commit_response(transcript, response_text)

    def send_message(self, message: str) -> Any:
        return self._send_message(message)

    def set_thinking(self, thinking: float | None = None) -> None:
        if thinking is None:
            self.thinking = None
            return

        if not 0.0 <= thinking <= 1.0:
            raise ValueError("rmAI: thinking must be between 0 and 1.")

        self.thinking = thinking

    def _infer_mime_type(self, path: str) -> str:
        import mimetypes

        mime_type, _ = mimetypes.guess_type(path)
        return mime_type or "application/octet-stream"

    def attach_file(self, pathtofile: str):
        """
        Attach a local file to the transcript.
        """
        self._append_message(self.transcript, "user", [self._file_part(pathtofile)])

    def structured_output(self, schema_filepath: str | None = None, schema_str: str | None = None) -> None:
        if schema_filepath and schema_str:
            raise ValueError("rmtools.ai: Structured output takes a filepath OR the string of the JSON, not both.")

        if schema_filepath:
            with open(schema_filepath, "r") as schema:
                json_str: str = schema.read()
        elif schema_str:
            json_str = schema_str
        else:
            self.config = {}
            return

        json_dict: dict[str, Any] = json.loads(json_str)
        self.config = {"response_mime_type": "application/json", "response_json_schema": json_dict}

    def attach_text(self, text: str | None = None, text_filepath: str | None = None) -> None:
        if (not text and not text_filepath) or (text_filepath and text):
            raise ValueError("rmAI: You must specify either text or text filepath to attach.")

        if text_filepath:
            with open(text_filepath, "r") as file:
                text = file.read()

            if not text.strip():
                raise ValueError(f"rmAI: {text_filepath} is empty.")

        self._append_message(self.transcript, "user", [self._text_part(text)])

    def load_prompt(self, prompt_name: str, prompts_dir_path: str = "prompts"):
        filepath = os.path.join(prompts_dir_path, f"{prompt_name}.txt")
        self.attach_text(text_filepath=filepath)
        return

    def context_save(self, save_filepath: str, file_preserving_path: str = "") -> None:
        data = {
            "transcript": self.transcript,
            "config": self.config,
            "thinking": self.thinking,
        }
        json_str = json.dumps(data)

        if not file_preserving_path:
            with open(save_filepath, "w") as context_file:
                context_file.write(json_str)
            return

    def context_load(self, load_filepath: str) -> None:
        with open(load_filepath, "r") as context_file:
            data = json.loads(context_file.read())

        self.transcript = data.get("transcript", [])
        self.config = data.get("config", {})
        self.thinking = data.get("thinking", None)

    def embed_text(self, text: str) -> list[float]:
        if not self.client:
            raise ValueError("rmAI: Embeddings require a Gemini or Vertex client.")

        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return response.embeddings[0].values
