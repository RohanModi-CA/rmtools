from google import genai
from typing import Any
import google.genai.errors
import json
from dotenv import load_dotenv, find_dotenv
import os
import warnings
from typing import Optional
import tenacity


thought_signature_warning:str =  "there are non-text parts in the response: ['thought_signature']"
warnings.filterwarnings('ignore', thought_signature_warning)


class AI_Instance:
    def _model_selector(self, model:str="")-> str:
        default_model:str = "gemini-flash-latest"

        if not model:
            print(f"rmAI: No Model Specified, Defaulting to {default_model}")
            model = default_model

        return model

    def _env_truthy(self, env_var_name: str)->bool:
        return os.getenv(env_var_name, "").lower() in ("true", "1")

    def _get_env_api_key(self)->str:
        return os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY') or ""

    def _create_client(self, api_key:str="", vertex_api_key:str="")->tuple[genai.Client, str]:
        if api_key and vertex_api_key:
            raise ValueError("rmAI: Set api_key or vertex_api_key, not both.")

        if vertex_api_key:
            return genai.Client(vertexai=True, api_key=vertex_api_key), "vertex"

        if api_key:
            return genai.Client(api_key=api_key), "gemini"

        load_dotenv(find_dotenv(usecwd=True))
        env_api_key:str = self._get_env_api_key()

        if self._env_truthy('GOOGLE_GENAI_USE_VERTEXAI'):
            if env_api_key:
                return genai.Client(vertexai=True, api_key=env_api_key), "vertex"
            return genai.Client(vertexai=True), "vertex"

        if not env_api_key:
            raise ValueError("rmAI: No API key found. Set api_key, vertex_api_key, GOOGLE_API_KEY, or GEMINI_API_KEY.")

        return genai.Client(api_key=env_api_key), "gemini"

    def __init__(self, api_key:str="", model:str="", vertex_api_key:str=""):
        self.client, self._client_mode = self._create_client(api_key=api_key, vertex_api_key=vertex_api_key)
        self.model:str = self._model_selector(model)
        self.chat = self.client.chats.create(model=self.model)

        self.config:dict = {}
        self._attached_file_uri_paths:dict[str,str] = {}


    def _send_message(self, message:str="") -> Any:
        if not message:
            message = " "

        response = self.chat.send_message(message=message, config=self.config)

        if self.config:
            return dict(json.loads(response.text))
        else:
            return response.text


    
    def send_message(self, message:str)->Any:
        return self._send_message(message)


    def _infer_mime_type(self, path: str) -> str:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(path)
        return mime_type or "application/octet-stream"

    def attach_file(self, pathtofile:str):
        """
        Attaches a local file to the chat.
        - Gemini Developer API: uses client.files.upload()
        - Vertex: uses Part.from_bytes() for inline bytes
        """
        if self._client_mode == "vertex":
            with open(pathtofile, "rb") as f:
                data = f.read()
            mime_type = self._infer_mime_type(pathtofile)
            file_part = genai.types.Part.from_bytes(data=data, mime_type=mime_type)
        else:
            file = self.client.files.upload(file=pathtofile)
            if not file or not file.uri:
                raise Exception("rmAI: Failed to Upload File")
            file_part = genai.types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)

        file_content = genai.types.Content(parts=[file_part])
        self.chat._curated_history.append(file_content)


    def structured_output(self, schema_filepath:str|None=None, schema_str:str|None=None)->None:
        if schema_filepath and schema_str:
            raise(ValueError("rmtools.ai: Structured output takes a filepath OR the string of the JSON, not both."))

        if schema_filepath:
            with open(schema_filepath,'r') as schema:
                json_str:str = schema.read()
        elif schema_str:
            json_str:str = schema_str
        else:
            self.config = {}
            return

        json_dict:dict = json.loads(json_str)

        self.config = {'response_mime_type': 'application/json', 'response_json_schema': json_dict}
        return


    def attach_text(self, text:str|None=None, text_filepath:str|None=None)->None:
        if not text and not text_filepath or (text_filepath and text):
            raise ValueError("rmAI: You must specify either text or text filepath to attach.")

        if text_filepath:
            with open(text_filepath, 'r') as file:
                text = file.read()

            if not text.strip():
                raise ValueError(f"rmAI: {text_filepath} is empty.")


        text_part:genai.types.Part = genai.types.Part(text=text)
        text_content:genai.types.Content = genai.types.Content(parts=[text_part])
        
        self.chat._curated_history.append(text_content)


    def load_prompt(self, prompt_name:str, prompts_dir_path:str="prompts"):
        filepath = os.path.join(prompts_dir_path, f'{prompt_name}.txt')
        self.attach_text(text_filepath=filepath)
        return
        
    
    def context_save(self, save_filepath:str, file_preserving_path:str="")->None:
        data = [content.model_dump() for content in self.chat._curated_history]
        json_str = json.dumps(data)

        if not file_preserving_path:
            with open(save_filepath, 'w') as context_file:
                context_file.write(json_str)
            return
        
    
    def embed_text(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return response.embeddings[0].values



class rmAI_Pipeline:
    pass