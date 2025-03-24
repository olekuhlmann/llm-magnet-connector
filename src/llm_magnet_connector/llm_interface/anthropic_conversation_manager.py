from dotenv import load_dotenv

load_dotenv()  # load variables from .env into os.environ

from . import LLMResponse, OptimizerParameters, BadnessCriteria, LLMConversationManager
import os
import anthropic
import base64
import mimetypes


class AnthropicConversationManager(LLMConversationManager):
    """
    This class is a subclass of LLMConversationManager and is used to manage a conversation with the Anthropic API.
    Will send the system prompt with the first prompt.

    Uses the environment variable ANTHROPIC_API_KEY.
    """

    def __init__(
        self, system_prompt = None, output_token_limit=8000, context_window_limit=100_000, max_prompts=100
    ):
        super().__init__(output_token_limit, context_window_limit, max_prompts)
        self.__client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = "claude-3-7-sonnet-latest"
        self._thinking = {
            "type": "enabled",
            "budget_tokens": 2000,
        }
        self._system_prompt = system_prompt
        self._temperature = 1  # must be 1 when thinking is enabled
        model_token_limit = 64000 if self._thinking else 8192
        self._max_tokens = (
            model_token_limit
            if output_token_limit == -1
            else min(output_token_limit, model_token_limit)
        )
        model_context_window_limit = 200_000
        self._context_window_limit = (
            model_context_window_limit
            if context_window_limit == -1
            else min(context_window_limit, model_context_window_limit)
        )
        min_context_window_limit = 18000
        if self._context_window_limit < min_context_window_limit:
            raise ValueError(
                f"The context window limit must be at least {min_context_window_limit} tokens, as this has proven to be the number of tokens used up by the initial prompt, model answer, and one re-prompt. Please reconfigure."
            )
        self._max_prompts = (
            max_prompts if max_prompts != -1 else 1000
        )  # hardcoded limit

    def _send_message(self, messages: list, system_prompt: str | None = None):
        """
        Sends a message to the model and increments the prompt count.

        Args:
            messages ([Message]): The messages to send to the model (context and new message).
            system_prompt (str): The system prompt to use. Should only be used for the first prompt.

        Raises:
            ValueError: If the maximum number of prompts has been reached.

        Returns:
            The response from the model.
        """
        self._prompt_count += 1
        if self._prompt_count > self._max_prompts:
            raise ValueError(f"Max number of {self._max_prompts} prompts reached.")

        response = self.__client.messages.create(
            model=self._model,
            messages=messages,
            system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            thinking=self._thinking,
        )

        self.usage_input_tokens += response.usage.input_tokens
        self.usage_output_tokens += response.usage.output_tokens

        return response

    def prompt(self, prompt: str, images_dir: str | None) -> LLMResponse:
        # Convert images to base64 (with text blocks)
        image_blocks = []
        if images_dir is not None:
            for image_file in os.listdir(images_dir):
                image_path = os.path.join(images_dir, image_file)
                image_blocks.extend(
                    AnthropicConversationManager._image_to_base64_message(image_path)
                )

        # Create the message
        new_message = {
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": prompt}],
        }

        # Add the new message to the context
        self._context.append(new_message)

        # Remove old messages if the context window size is exceeded
        self._manage_context()

        # Send the message to the model, include system prompt if this is the first prompt
        system_prompt = self._system_prompt if self._prompt_count == 0 else None
        response = self._send_message(self._context, system_prompt=system_prompt)

        # add response to context
        self._context.append(
            {"role": "assistant", "content": response.content}
        )

        # check stop reason
        if response.stop_reason != "end_turn":
            print(
                f"[Warning]: The LLM answer was stopped due to stop reason '{response['stop_reason']}'."
            )
            

        # TODO Return the response
        print(response)
        return LLMResponse(
            optimizer_parameters=OptimizerParameters(-1, -1, -1, -1),
            badnessCriteria=BadnessCriteria(True, True, True, True),
        )

    def _manage_context(self):
        """
        Manages the context by removing old messages. Will always keep the first user message and corresponding model response as well as the latest user message.
        Removes the the oldest user message and corresponding model response if the context window size is exceeded.
        """

        def count_tokens(messages):
            # Calculate the total token count of the context
            response = self.__client.messages.count_tokens(
                model=self._model,
                messages=self._context,
                thinking=self._thinking,
            )

            return response.input_tokens

        input_tokens = count_tokens(self._context)

        # Remove the oldest user message and corresponding model response if the context window size is exceeded
        while (
            input_tokens > self._context_window_limit * 0.95
        ):  # count_tokens is not exact, so we use 95% of the limit
            # never remove the first user message and corresponding model response, never remove the last user message
            if len(self._context) <= 3:
                raise ValueError(
                    "First user message, corresponding model answer and latest user message exceed context window limit. Please reconfigure."
                )

            # remove the oldest user message and corresponding model response, i.e. indices 2, 3
            if len(self._context) >= 5:
                self._context.pop(2)
                self._context.pop(2)
            else:
                raise Exception(
                    "Something went wrong when managing the context window. This should not happen."
                )

            input_tokens = count_tokens(self._context)

    def _image_to_base64_message(image_path):
        """
        Converts a local image file to a list of two message blocks containing (1) a text block with the image file name, e.g., "Image 0a:" for 0a.png, and (2) the image data in base64 encoding
        Taken from Anthropic's `anthropic-cookbook` example code and modified.

        Args:
            image_path (str): The path to the image file.
        """
        # Open the image file in "read binary" mode
        with open(image_path, "rb") as image_file:
            # Read the contents of the image as a bytes object
            binary_data = image_file.read()

        # Encode the binary data using Base64 encoding
        base64_encoded_data = base64.b64encode(binary_data)

        # Decode base64_encoded_data from bytes to a string
        base64_string = base64_encoded_data.decode("utf-8")

        # Get the MIME type of the image based on its file extension
        mime_type, _ = mimetypes.guess_type(image_path)

        # Create the image block
        image_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64_string,
            },
        }

        # Remove the file extension from the image file name
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        text_block = {"type": "text", "text": f"Image {image_name}:"}

        return [text_block, image_block]
