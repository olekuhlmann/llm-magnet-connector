from abc import ABC, abstractmethod
from .llm_response import LLMResponse

class LLMClient(ABC):
    """
    This class is an abstract class that defines the interface for the LLM client.
    """
    
    @abstractmethod
    def prompt(self, prompt: str, images_dir: str) -> LLMResponse:
        """
        This method should take a prompt and a path to a directory of images to prompt the model with and return an LLMResponse object.
        All images in the directory should be considered when generating the response.
        
        Args:
            prompt (str): The prompt to be used for the LLM.
            images_dir (str): The directory where the images are stored.
            
        Raises:
            ValueError: If not all images could be attached to the prompt.
        """
        pass