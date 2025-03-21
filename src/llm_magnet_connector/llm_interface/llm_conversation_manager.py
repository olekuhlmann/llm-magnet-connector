from abc import ABC, abstractmethod
from .llm_response import LLMResponse

class LLMConversationManager(ABC):
    """
    This class is an abstract class for handling one conversation with a LLM.
    """
    
    def __init__(self, output_token_limit, max_prompts=-1):
        """
        Initializes the LLMConversationManager.
        
        Args:
            output_token_limit (int): The maximum number of tokens the model can generate. If -1, the default limit of the model will be used.
            max_prompts (int): The maximum number of prompts that can be sent to the model. If -1, there is no limit.
        """
        self._usage_input_tokens = 0
        self._usage_output_tokens = 0
        self._prompt_count = 0
        self._context = []
    
    @abstractmethod
    def prompt(self, prompt: str, images_dir: str) -> LLMResponse:
        """
        This method should take a prompt and a path to a directory of images to prompt the model with and return an LLMResponse object.
        All images in the directory should be considered when generating the response.
        Context should be maintained between calls to this method.
        
        Args:
            prompt (str): The prompt to be used for the LLM.
            images_dir (str): The directory where the images are stored.
            
        Raises:
            ValueError: If not all images could be attached to the prompt.
        """
        pass