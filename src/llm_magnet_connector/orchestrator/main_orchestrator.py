from llm_magnet_connector.llm_interface import (
    LLMConversationManager,
    LLMResponse,
    get_reprompt,
)
from llm_magnet_connector.image_generator import ResponseToImage


class MainOrchestrator:
    """
    This class is the entry point for the LLM Magnet Connector. It prompts the LLM conversation manager with the input prompt and images, passes the response to the image generator, and re-prompts the LLM conversation manager with the generated images.
    This is done until the conversation is finished or the specified number of iterations is reached.
    """

    def __init__(
        self,
        llm_manager: LLMConversationManager,
        image_generator: ResponseToImage,
        max_iterations: int,
        logger
    ):
        """
        Initializes the MainOrchestrator.

        Args:
            llm_manager (LLMConversationManager): The LLMConversationManager to use.
            image_generator (ResponseToImage): The image generator to use.
            max_iterations (int): The maximum number of iterations to run the conversation for (excludes initial prompt).
            logger: The logger to use.
        """
        self._llm_manager = llm_manager
        self._image_generator = image_generator
        self._max_iterations = max_iterations
        self._iteration = 0
        self.logger = logger

    def run(self, initial_prompt: str, initial_images_dir: str):
        """
        Runs the LLM Magnet Connector.

        Args:
            initial_prompt (str): The initial prompt to start the conversation with.
            initial_images_dir (str): The directory where the images for the initial prompt are stored.
        """
        # TODO add logging
        self.logger.info("Starting conversation...")

        # initial prompt
        self.logger.info(f"Prompting LLM with initial prompt and images in {initial_images_dir}")
        response = self._llm_manager.prompt(initial_prompt, initial_images_dir)

        # re-prompt with new images
        while not self.is_terminated(response):
            self.logger.info(f"Answer: {response}")
            # generate images
            images_dir = self._image_generator.response_to_image(response)

            # re-prompt
            if self._iteration >= self._max_iterations:
                self.logger.info(f"Reached maximum number of {self._max_iterations} iterations.")
                break

            self.logger.info(f"Re-prompting for iteration {self._iteration} / {self._max_iterations-1}.")
            prompt = get_reprompt(
                response.optimizer_parameters, self._image_generator.image_index
            ) 
            response = self._llm_manager.prompt(prompt, images_dir)
            self._iteration += 1

        if self.is_terminated(response):
            self.logger.info("LLM states conversation as terminated.")

        self.logger.info("Conversation finished.")
        self.logger.info(f"Input tokens used: {self._llm_manager.usage_input_tokens}")
        self.logger.info(f"Output tokens used: {self._llm_manager.usage_output_tokens}")

    def is_terminated(self, response: LLMResponse) -> bool:
        """
        Checks if the conversation is terminated.

        Args:
            response (LLMResponse): The response from the LLM.
        """
        terminated = False
        badness_criteria = response.badnessCriteria
        if badness_criteria is not None:
            if (
                not badness_criteria.unrealizable_kinks
                and not badness_criteria.ends_not_smooth
                and not badness_criteria.overlapping
                and not badness_criteria.unreasonable_length
            ):  # all False
                terminated = True
        return terminated
