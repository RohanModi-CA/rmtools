from google import genai
import google.genai.errors
import json
from dotenv import load_dotenv
import os
import warnings
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
            load_dotenv()
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


    def _handle_rate_limit(self, ClientError:google.genai.errors.ClientError):
        retry_delay_s: int = ClientError['e']

    """
    We need to implement two main features. One is a retry that is sensible,
    and perhaps that respects the requested time. The second is a simple management
    thing that 1.) Alerts programs that they are being rate limited, and 2.) allows for
    configurations to determine the retry behaviour.

    For the alerting programs they are being rate limited, one option is return that alongside
    the final return, but that would only happen after the function waits a while, and the 
    main function is kept in the dark. It's not ideal. Another option is perhaps a promise
    based approach but that seems very clunky and overall, I am not a fan. 

    Another option, possibly, is to have the program pass a Callable into the retry logic,
    and the retry function will call the Callable, doing whatever thing the program would
    have done with this information. This seems like by far the best option, but I'm not
    confident with the whole self thing. If we pass in a self method, that of course needs
    access to the self object, but is that object passed by reference? If it is, then I think
    this could work? Because in that case it is not a 'frozen' copy, etc. Also, there was some
    discussion of 'pickling' when we were considering self issues in mutliprocessing? We will be multiprocessing,
    so I'm not sure about this.

    We want these alerts so that we can modify the resource limits, perhaps. Or do we? What will we actually do?
    So right now, we have a limit to the active behaviours. At this current point, we may very well hit that limit.

    Perhaps a better approach that can sidestep all of this is found by not going horizontally, which suffers from
    all of the limits we have found in our case, but through a 'uniform' approach. This approach can work such that
    the function tries to keep uniform resource allocation. As in, it attempts to keep the ratio of resource utilization
    the same as that defined within the resource limits. This solves one of the two major issues with the current approach:
    rate limited processes eating the entirety of the 'overall' resource limit. However, there is a second major issue,
    which is that we will be spamming new attempts while the system has already asked us to slow down. This is a transient
    issue, since it will mostly go away once we fill the resource limit corresponding to AI, or whatever, possibly. But 
    I think we can't ignore it, since they could block our IP or something for 'spamming' them. Ultimately, I think we
    need a way to know whether or not they are accepting us or not. 

    And once we know, we probably need a way to actually handle that. If we know they're not taking new requests for the next
    24seconds, perhaps we should not send anything their way for that time. Which means that _get_legal_task will need to be made
    aware. Perhaps with a malleable 'resource_cooldown'. I suppose we should have a malleable and a user-set one. 

    So that is well and good, but the issue of knowing when we're being rate limited is highly difficult, I suppose because it requires
    communicating accross the cpu process barrier. 

    So I think maybe we have to use this manager thing. We will launch tasks with a manager, who will have a dict that can be optionally passed
    to the functions. I suppose any step will also have a manager_dict_kwarg:str. Then, we will also need to control.

    """


    @tenacity.retry
    def send_message(self, message:str) -> str|dict:
        """
        If structured_output is enabled, this will return a dict.
        Else, it returns just the string of the reply.
        """
        response = self.chat.send_message(message=message, config=self.config)

        if self.config:
            return dict(json.loads(response.text))
        else:
            return response.text

    def attach_file(self, pathtofile:str):
        """
        We will manipulate this to add into the curated_history,
        """

        file = self.client.files.upload(file=pathtofile)
        
        if not file or not file.uri:
            raise Exception("rmAI: Failed to Upload File")

        file_part:genai.types.Part = genai.types.Part.from_uri(file_uri=file.uri, mime_type=file.mime_type)
        file_content:genai.types.Content =genai.types.Content(parts=[file_part])

        # Now let's add this to the curated history to make it as if we have 'sent' it.
        self.chat._curated_history.append(file_content)

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
