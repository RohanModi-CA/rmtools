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
        """
        Here we will attempt to sanitize the model input.
        Can we get the available models?
        """
    
        default_model:str = "gemini-flash-latest"

        if not model:
            print(f"rmAI: No Model Specified, Defaulting to {default_model}")
            model = default_model

        # We will just assume a sanitized model at this point because this gets rate limited and je veux pas m'occuper de ca a ce moment.
        """
        models:list = list(self.client.models.list())

        # We need to sanitize the modelname by removing the models/ prefix.
        # If '/' is not found this will just leave unchanged since find returns -1.
        self.model_names:list = [model.name[model.name.find('/') + 1:] for model in models]

        if model in self.model_names:
            return model
        else:
            raise ValueError(f"rmAI: Model {model} not found")
        """
        return model

    def _set_api_key(self, api_key:str="")->str:
        if api_key:
            return api_key
        else:
            load_dotenv(find_dotenv(usecwd=True))
            key:str|None = os.getenv('GEMINI_API_KEY')
            
            if not key:
                raise ValueError("rmAI: No API Key or 'GEMINI_API_KEY' in .env")
            else:
                api_key = key
                return api_key

    def __init__(self, api_key:str="", model:str=""):
        api_key = self._set_api_key(api_key)

        self.client = genai.Client(api_key=api_key)
        self.model:str = self._model_selector(model)
        self.chat = self.client.chats.create(model=self.model)

        self.config:dict = {}
        self._attached_file_uri_paths:dict[str,str] = {}


    def _send_message(self, message:str) -> Any:
        """
        No retry behaviour.
        If structured_output is enabled, this will return a dict.
        Else, it returns just the string of the reply.
        """
        response = self.chat.send_message(message=message, config=self.config)

        if self.config:
            return dict(json.loads(response.text))
        else:
            return response.text


    
    def send_message(self, message:str)->Any:
        """
        Retries.
        If structured_output is enabled, this will return a dict.
        Else, it returns just the string of the reply. 
        """
        self._send_message(message)


    def attach_file(self, pathtofile:str):
        """
        We will manipulate this to add into the curated_history. This adds to _attached_file_tuples
        """

        file = self.client.files.upload(file=pathtofile)
        
        if not file or not file.uri:
            raise Exception("rmAI: Failed to Upload File")

        file_part:genai.types.Part = genai.types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
        file_content:genai.types.Content =genai.types.Content(parts=[file_part])

        # Now let's add this to the curated history to make it as if we have 'sent' it.
        self.chat._curated_history.append(file_content)

        #TODO: And let's add this to the list of attached files in case we want to save the context.
        
        #self._attached_file_uri_paths.append()


    def structured_output(self, schema_filepath:str|None=None, schema_str:str|None=None)->None:
        """
        This function sets the config of the current AI instance to include the structured output.
        Pass either the filepath to the JSON or the JSON as a string (not both), or neither to clear.
        """

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

        # We need to convert the strings to JSON.
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
        """
        Adds the contents of prompts_dir_path/prompt_name.txt to the context.
        """
        filepath = os.path.join(prompts_dir_path, f'{prompt_name}.txt')
        self.attach_text(text_filepath=filepath)
        return
        
    
    def context_save(self, save_filepath:str, file_preserving_path:str="")->None:
        #TODO: Implement
        """ Dumps the context up to now, in JSON format, to save_filepath.
            Does not preserve response schemas. Files may expire by default. 
            file_preserving_path changes behaviour completely.

            Args:
                save_filepath:str, the filepath of where to put the JSON.
                file_preserving_path: str. If this is set, this will not save the raw context. It will make a JSON with key 'context', value: raw context JSON, and key 'file_tuples'
            """
            
        data = [content.model_dump() for content in self.chat._curated_history]
        json_str = json.dumps(data)

        if not file_preserving_path:
            with open(save_filepath, 'w') as context_file:
                context_file.write(json_str)
            return
        
        # Now, to implement the file tuples. 




        



class rmAI_Pipeline:
    """
    The idea we will do here is to create a system where
    we can define a pipeline, and this will run it automatically.
    We're not particularly interested in integrations like N8N is,
    we just want a simple thing.

    A basic example:
    a function creates a string str1. str1 is then sent to AI with a prompt. 
    the AI then responds with rep1. func2 processes this, and returns str2, which is run
    into a verifier, which decides that this was a failure, and then sends us back to the first step, 
    for example.

    This will be linear. We will define a series of steps. So a step, simply put, has an input, and an out
    """
    pass
