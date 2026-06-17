from . import core
from .. import ai
from typing import Optional, Any
from urllib import error as urllib_error
import tenacity

class AI_Instance_PL(ai.AI_Instance):

    def __init__(self, model: str = "", openrouter_api_key: str = "", process_state_functions: Optional[core.ProcessStateFunctions]=None, rate_limit_resource_names: list[str]|None=None)->None:
        """
        Args:
            openrouter_api_key: OpenRouter API key.
            process_state_functions: optional, used to report rate limits to the resources in rate_limit_resource
            rate_limit_resource_names: optional list of resource names to report cooldowns to.
        """
        self.process_state_functions = process_state_functions
        self._set_rate_limit_resource_names(rate_limit_resource_names)
        super().__init__(model=model, openrouter_api_key=openrouter_api_key)

        
        self._EXPONENTIAL_BACKOFF = tenacity.wait_exponential(min=1, max=300, multiplier=2) 
    
        self.send_message = tenacity.retry(wait=self._get_retry_time_send_upstream_s, reraise=True)(self.send_message)

    def _set_rate_limit_resource_names(self, rate_limit_resource_names:list[str]|None)->None:
        self.rate_limit_resource_names = rate_limit_resource_names or []
        if not self.rate_limit_resource_names and self.process_state_functions:
            self.rate_limit_resource_names = self.process_state_functions.rate_limit_resource_names
        return
    
    
    def _read_error_time_ms(self, api_error: BaseException)->int:
        """
        Returns an int amount of milliseconds to wait based on a rate limit error. If we can't figure it out, return -1.
        """
        if isinstance(api_error, ai.AIRequestError):
            return api_error.retry_after_ms or -1

        if isinstance(api_error, urllib_error.HTTPError):
            retry_after = api_error.headers.get("Retry-After", "") if api_error.headers else ""
            if retry_after:
                try:
                    return int(float(retry_after) * 1000)
                except ValueError:
                    pass

        return -1
        
    def _retry_time_s(self, RetryCallState:tenacity.RetryCallState)->float:
        """ Applies rules to decide the retry time and returns it as a float.
        """
        outcome = RetryCallState.outcome
        
        if not outcome: # don't think this should happen
            return 20 

        original_error = outcome.exception()

        if not original_error:
            return self._EXPONENTIAL_BACKOFF(RetryCallState)

        # Will be -1ms if failed.
        error_time_s:float = self._read_error_time_ms(original_error)/1000.0
    
        if error_time_s < 0:
            wait_time_s:float = self._EXPONENTIAL_BACKOFF(RetryCallState)
            print(f'rmAI_PL: Rate limited, resorting to exponetial backoff, waiting {wait_time_s} seconds.')
            return wait_time_s
        else:
            print(f'rmAI_PL: Rate limited. Wait time requested: {error_time_s} seconds.')
            return error_time_s

    def _get_retry_time_send_upstream_s(self, RetryCallState: tenacity.RetryCallState)->float:
        """calls _retry_time_s to get the retry time, calls set_resource_cooldown func if set, and returns the time."""
        time = self._retry_time_s(RetryCallState)

        if self.process_state_functions:
            for resource in self.rate_limit_resource_names:
                self.process_state_functions.set_resource_cooldown_func(resource, int(1000*time))
        return time
            
        

    def send_message(self, message: str="", alert_replied:bool=True) -> Any:
        """
        Sends a message, returning a dict only if a structured output
        schema has been set. Otherwise returns a string.

        Retries limit amount of times and sets cooldowns if rate limited.
        """
        
        response = super()._send_message(message)
        if alert_replied:
            print("AI_Instance_PL: Received response.")
        return response
