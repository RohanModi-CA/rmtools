from typing import Any, Callable
import base64
import json
import os
import warnings
from urllib import error as urllib_error
from urllib import request as urllib_request


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
    def _model_selector(self, model: str = "") -> str:
        """Pick a default OpenRouter model when callers do not specify one."""
        if not model:
            model = "openai/gpt-4o-mini"
            print(f"rmAI: No Model Specified, Defaulting to {model}")
        return model

    def _resolve_openrouter_key(self, openrouter_api_key: str = "") -> str:
        """Resolve the OpenRouter API key or fail immediately."""
        key = openrouter_api_key.strip()
        if not key:
            key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise ValueError("rmAI: Set openrouter_api_key or OPENROUTER_API_KEY.")
        return key

    def _normalize_tool_defs(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only OpenAI-style function tools and make the shape explicit."""
        normalized: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                raise ValueError("rmAI: Only function tools are supported.")

            function = tool.get("function", {})
            name = function.get("name", "").strip()
            if not name:
                raise ValueError("rmAI: Each tool must have a function name.")

            normalized.append(tool)
        return normalized

    def _tool_name(self, tool: dict[str, Any]) -> str:
        """Read the function name from one tool definition."""
        function = tool.get("function", {})
        return function.get("name", "")

    def _normalize_function_aliases(self, function_aliases: list[tuple[str, Callable[..., Any]]] | None) -> dict[str, Callable[..., Any]]:
        """Turn the alias list into a small name-to-callable map."""
        aliases: dict[str, Callable[..., Any]] = {}
        for name, func in function_aliases or []:
            if name in aliases:
                raise ValueError(f"rmAI: Duplicate function alias: {name}")
            aliases[name] = func
        return aliases

    def _load_tool_schema(self, schema_filepath: str | None, schema_str: str | None) -> list[dict[str, Any]] | None:
        """Load a tool schema from disk or memory and return the tool list."""
        if schema_filepath and schema_str:
            raise ValueError("rmtools.ai: Function calling takes a filepath OR the string of the JSON, not both.")

        if schema_filepath:
            with open(schema_filepath, "r") as schema:
                json_str = schema.read()
        elif schema_str:
            json_str = schema_str
        else:
            return None

        schema = json.loads(json_str)
        if isinstance(schema, list):
            return self._normalize_tool_defs(schema)

        if isinstance(schema, dict) and "tools" in schema:
            tools = schema["tools"]
            if not isinstance(tools, list):
                raise ValueError("rmAI: tools must be a list.")
            return self._normalize_tool_defs(tools)

        raise ValueError("rmAI: Function calling schema must be a tools array or an object with a tools key.")

    def _validate_function_calling(self, tools: list[dict[str, Any]], aliases: dict[str, Callable[..., Any]], forced: list[str]) -> None:
        """Check that every advertised tool has a callable and that forced names exist."""
        tool_names = [self._tool_name(tool) for tool in tools]

        missing_aliases = [name for name in tool_names if name not in aliases]
        if missing_aliases:
            raise ValueError(f"rmAI: Missing callables for tools: {', '.join(missing_aliases)}")

        extra_aliases = [name for name in aliases if name not in tool_names]
        if extra_aliases:
            warnings.warn(
                f"rmAI: Ignoring function aliases not present in the schema: {', '.join(extra_aliases)}",
                stacklevel=2,
            )

        unknown_forced = [name for name in forced if name not in tool_names]
        if unknown_forced:
            raise ValueError(f"rmAI: Forced function names not present in the schema: {', '.join(unknown_forced)}")

    def _reset_function_calling(self) -> None:
        """Clear the runtime tool-calling state without touching structured output."""
        self._function_tools = []
        self._function_aliases = {}
        self._forced_function_names = []
        self._parallel_tool_calls: bool | None = None
        self._pause_after_tool_execution = False
        self._function_calling_enabled = False

    def _stringify_tool_result(self, result: Any) -> str:
        """Turn a tool return value into content the model can read back."""
        if isinstance(result, str):
            return result

        try:
            return json.dumps(result)
        except TypeError as exc:
            raise TypeError("rmAI: Tool results must be JSON serializable or plain strings.") from exc

    def _force_tool_choice(self) -> dict[str, Any] | str | None:
        """Convert the forced-name list into one OpenRouter tool_choice value."""
        if not self._forced_function_names:
            return None

        if len(self._forced_function_names) == 1:
            return {
                "type": "function",
                "function": {
                    "name": self._forced_function_names[0],
                },
            }

        return "required"

    def __init__(
        self,
        model: str = "",
        openrouter_api_key: str = "",
        thinking: float | None = None,
    ):
        self._openrouter_api_key = self._resolve_openrouter_key(openrouter_api_key=openrouter_api_key)
        self.model: str = self._model_selector(model)
        self.config: dict[str, Any] = {}
        self.thinking: float | None = None
        self.transcript: list[dict[str, Any]] = []
        self._reset_function_calling()
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

    def _openrouter_messages(self, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Serialize the transcript into OpenRouter's chat-completions shape."""
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

            message: dict[str, Any] = {"role": entry["role"], "content": content}
            if entry["role"] == "assistant" and entry.get("tool_calls"):
                message["tool_calls"] = entry["tool_calls"]
                if not content:
                    message["content"] = None
            if entry["role"] == "tool" and entry.get("tool_call_id"):
                message["tool_call_id"] = entry["tool_call_id"]
            messages.append(message)
        return messages

    def _function_calling_request_payload(self, allow_force: bool = True) -> dict[str, Any]:
        """Build the request fields that belong to function calling."""
        payload: dict[str, Any] = {}
        if not self._function_calling_enabled:
            return payload

        tools = self._function_tools
        if self._forced_function_names:
            forced_names = set(self._forced_function_names)
            tools = [tool for tool in tools if self._tool_name(tool) in forced_names]

        if tools:
            payload["tools"] = tools

        tool_choice = self._force_tool_choice() if allow_force else None
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        if self._parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = self._parallel_tool_calls

        return payload

    def _response_message_text(self, message: dict[str, Any]) -> str:
        """Read plain text back from an OpenRouter assistant message."""
        content = message.get("content", "")
        if isinstance(content, list):
            text_bits = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(text_bits)
        if content is None:
            return ""
        return str(content)

    def _response_message_to_transcript_entry(self, message: dict[str, Any]) -> dict[str, Any]:
        """Turn one OpenRouter response message into the transcript shape."""
        entry: dict[str, Any] = {"role": message.get("role", "assistant"), "parts": []}
        text = self._response_message_text(message)
        if text:
            entry["parts"] = [self._text_part(text)]
        if message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]
        return entry

    def _tool_result_entry(self, tool_call_id: str, result: Any) -> dict[str, Any]:
        """Store one tool result in the transcript."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "parts": [self._text_part(self._stringify_tool_result(result))],
        }

    def _execute_tool_call(self, tool_call: dict[str, Any]) -> Any:
        """Run one tool call by name and pass the JSON arguments to its callable."""
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        if tool_name not in self._function_aliases:
            raise ValueError(f"rmAI: No callable registered for tool: {tool_name}")

        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")

        if not isinstance(arguments, dict):
            raise ValueError("rmAI: Tool arguments must be a JSON object.")

        return self._function_aliases[tool_name](**arguments)

    def _append_tool_calls(self, transcript: list[dict[str, Any]], tool_calls: list[dict[str, Any]]) -> bool:
        """Append the tool results that follow one assistant tool-call message."""
        paused = False
        for tool_call in tool_calls:
            result = self._execute_tool_call(tool_call)
            transcript.append(self._tool_result_entry(tool_call.get("id", ""), result))
            paused = paused or self._pause_after_tool_execution
        return paused

    def _commit_transcript(self, transcript: list[dict[str, Any]]) -> None:
        """Publish the working transcript onto the instance."""
        self.transcript = [dict(entry) for entry in transcript]

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

    def _send_openrouter_message(self, transcript: list[dict[str, Any]], allow_force: bool = True) -> Any:
        """Send the transcript to OpenRouter and return the model text."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._openrouter_messages(transcript),
        }

        response_format = self._openrouter_response_format()
        if response_format:
            payload["response_format"] = response_format

        function_calling_payload = self._function_calling_request_payload(allow_force=allow_force)
        if function_calling_payload:
            payload.update(function_calling_payload)

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
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("rmAI: Could not read model response text.")

        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            raise ValueError("rmAI: Could not read model response text.")
        return message

    def _send_message(self, message: str = "") -> Any:
        transcript = self._pending_transcript(message)
        allow_force = True
        while True:
            message_dict = self._send_openrouter_message(transcript, allow_force=allow_force)
            allow_force = False
            tool_calls = message_dict.get("tool_calls", [])

            if tool_calls:
                transcript.append(self._response_message_to_transcript_entry(message_dict))
                self._commit_transcript(transcript)
                paused = self._append_tool_calls(transcript, tool_calls)
                self._commit_transcript(transcript)
                if paused:
                    return None
                continue

            response_text = self._response_message_text(message_dict)
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

    def function_calling(
        self,
        function_aliases: list[tuple[str, Callable[..., Any]]] | None = None,
        schema_filepath: str | None = None,
        schema_str: str | None = None,
        force_function_calling: list[str] | None = None,
        parallel_tool_calls: bool | None = None,
        pause_after_tool_execution: bool = False,
    ) -> None:
        """Set or clear the tool-calling schema used on future requests."""
        tools = self._load_tool_schema(schema_filepath=schema_filepath, schema_str=schema_str)
        if tools is None:
            if any([function_aliases, force_function_calling, parallel_tool_calls is not None, pause_after_tool_execution]):
                raise ValueError("rmAI: Function aliases and forcing require a function schema.")
            self._reset_function_calling()
            return

        aliases = self._normalize_function_aliases(function_aliases)
        forced = list(dict.fromkeys(force_function_calling or []))
        if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
            raise ValueError("rmAI: parallel_tool_calls must be True, False, or None.")
        if not isinstance(pause_after_tool_execution, bool):
            raise ValueError("rmAI: pause_after_tool_execution must be True or False.")

        self._validate_function_calling(tools=tools, aliases=aliases, forced=forced)

        self._function_tools = tools
        self._function_aliases = aliases
        self._forced_function_names = forced
        self._parallel_tool_calls = parallel_tool_calls
        self._pause_after_tool_execution = pause_after_tool_execution
        self._function_calling_enabled = True

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
