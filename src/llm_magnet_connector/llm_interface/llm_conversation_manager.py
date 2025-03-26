from abc import ABC, abstractmethod
from .llm_response import LLMResponse


class LLMConversationManager(ABC):
    """
    This class is an abstract class for handling one conversation with a LLM.
    """

    def __init__(
        self,
        logger,
        cost_1M_input_tokens,
        cost_1M_output_tokens,
        system_prompt=None,
        max_prompts=-1,
    ):
        """
        Initializes the LLMConversationManager.

        Args:
            logger: The logger to use.
            cost_1M_input_tokens (int): The cost of 1M input tokens (USD).
            cost_1M_output_tokens (int): The cost of 1M output tokens (USD).
            system_prompt (str): The system prompt to use.
            output_token_limit (int): The maximum number of tokens the model can generate. If -1, the default limit of the model will be used.
            context_window_limit (int): The token capacity used for the context window. If -1, the default limit of the model will be used.
            max_prompts (int): The maximum number of prompts that can be sent to the model. If -1, there is no limit.
        """
        self.logger = logger
        self.cost_1M_input_tokens = cost_1M_input_tokens
        self.cost_1M_output_tokens = cost_1M_output_tokens
        self._system_prompt = system_prompt
        self.usage_input_tokens = 0
        self.usage_output_tokens = 0
        self._prompt_count = 0
        self._context = []
        self._max_prompts = (
            max_prompts if max_prompts != -1 else 1000
        )  # hardcoded limit

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

    @abstractmethod
    def _add_to_context(self, element):
        """
        This method should add an element to self._context.
        The context is a list of lists where each list represents a functional element that can be safely removed from the context together (e.g., a user prompt and corresponding model answer).
        This method should decide whether to add the new element to the most recent list or to a new list.

        Args:
            element: The element to be added to the context.
        """
        pass

    @abstractmethod
    def _context_to_message(self):
        """
        This method should convert the self._context to a message that can be used as a prompt for the model.

        Returns:
            The context as a message. Format is up to the implementation.
        """
        pass

    @abstractmethod
    def _is_context_too_large(self) -> bool:
        """
        This method should check if the context is too large and should be trimmed.

        Returns:
            True if the context is too large, False otherwise.
        """
        pass

    def _manage_context(self):
        """
        This method manages the self._context by removing functional elements if necessary.
        It never removes the first element (containing the system prompt, initial prompt, etc.) and most recent element.
        The strategy followed is to remove the second oldest functional element if the context window limit is reached.
        """
        # Remove elements as long as the context window is too large
        while self._is_context_too_large():
            # never remove the first or last functional element
            if len(self._context) <= 2:
                self.logger.warning(
                    "2 functional context elements exceed the context window limit (i.e., initial interaction and latest query). Please increase the context window limit. Aborting context window management..."
                )
                return
            else:
                # remove the oldest functional element that is not the first or last
                self._context.pop(1)
