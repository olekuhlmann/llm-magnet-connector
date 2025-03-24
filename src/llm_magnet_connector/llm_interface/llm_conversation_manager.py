from abc import ABC, abstractmethod
from .llm_response import LLMResponse


class LLMConversationManager(ABC):
    """
    This class is an abstract class for handling one conversation with a LLM.
    """
    
    def __init__(self, system_prompt = None, output_token_limit=-1, context_window_limit=-1, max_prompts=-1):
        """
        Initializes the LLMConversationManager.
        
        Args:
            system_prompt (str): The system prompt to use.
            output_token_limit (int): The maximum number of tokens the model can generate. If -1, the default limit of the model will be used.
            context_window_limit (int): The token capacity used for the context window. If -1, the default limit of the model will be used.
            max_prompts (int): The maximum number of prompts that can be sent to the model. If -1, there is no limit.
        """
        self.usage_input_tokens = 0
        self.usage_output_tokens = 0
        self._prompt_count = 0
        self._context = []
    
    @abstractmethod
    def prompt(self, prompt: str, images_dir: str | None) -> LLMResponse:
        """
        This method should take a prompt and a path to a directory of images to prompt the model with and return an LLMResponse object.
        All images in the directory should be considered when generating the response. The file name of the image should coincide with the label on the image.
        Context should be maintained between calls to this method.
        Context window should be managed in the implementation of this method.
        
        Args:
            prompt (str): The prompt to be used for the LLM.
            images_dir (str): The directory where the images are stored.
            
        Raises:
            ValueError: If not all images could be attached to the prompt.
        """
        pass