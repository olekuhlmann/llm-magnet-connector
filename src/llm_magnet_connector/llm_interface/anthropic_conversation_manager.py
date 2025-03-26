from dotenv import load_dotenv

load_dotenv()  # load variables from .env into os.environ

from . import (
    LLMResponse,
    OptimizerParameters,
    BadnessCriteria,
    LLMConversationManager,
    anthropic_think_tool,
)
import os
import anthropic
import base64
import mimetypes
import re


class AnthropicConversationManager(LLMConversationManager):
    """
    This class is a subclass of LLMConversationManager and is used to manage a conversation with the Anthropic API.
    Will send the system prompt with the first prompt.

    Uses the environment variable ANTHROPIC_API_KEY.
    """

    def __init__(
        self,
        logger,
        cost_1M_input_tokens,
        cost_1M_output_tokens,
        system_prompt=None,
        output_token_limit=8000,
        context_window_limit=100_000,
        max_prompts=100,
        thinking=False,
        think_tool=True,
    ):
        """
        {}
        
        
        Additional Args:
            thinking (bool): Whether to enable thinking. Defaults to False.
            think_tool (bool): Whether to use the "think" tool. Defaults to True. (see https://www.anthropic.com/engineering/claude-think-tool)
        """.format(
            LLMConversationManager.__init__.__doc__
        )
        super().__init__(
            logger=logger,
            cost_1M_input_tokens=cost_1M_input_tokens,
            cost_1M_output_tokens=cost_1M_output_tokens,
            system_prompt=system_prompt,
            output_token_limit=output_token_limit,
            context_window_limit=context_window_limit,
            max_prompts=max_prompts,
        )
        self.__client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = "claude-3-7-sonnet-latest"
        if thinking:
            self._thinking = {
                "type": "enabled",
                "budget_tokens": 2000,
            }
        else:
            self._thinking = {"type": "disabled"}
        self._think_tool = think_tool
        self._tools = []
        if self._think_tool:
            system_prompt_suffix, think_tool_schema = anthropic_think_tool()
            self._system_prompt = f"{self._system_prompt if self._system_prompt else ''}\n\n{system_prompt_suffix}"
            self._tools += [think_tool_schema]
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
            tools=self._tools,
        )

        self.usage_input_tokens += response.usage.input_tokens
        self.usage_output_tokens += response.usage.output_tokens

        return response

    def prompt(self, prompt: str, images_dir: str | None) -> LLMResponse:
        def send_prompt(new_message) -> LLMResponse:
            """
            Local helper function to send the prompt.
            If the model answers with a tool_use, the function will call itself recursively with the tool use result.

            Args:
                new_message: The new message to add to the context.
            """
            # Add the new message to the context
            self._context.append(new_message)

            # Remove old messages if the context window size is exceeded
            self._manage_context()

            # Send the message to the model, include system prompt if this is the first prompt
            system_prompt = self._system_prompt if self._prompt_count == 0 else None
            response = self._send_message(self._context, system_prompt=system_prompt)

            # add response to context
            self._context.append({"role": "assistant", "content": response.content})

            # log the response
            self.logger.debug(response)
            for message in response.content:
                self.logger.info(self.__format_message(message))
                
            # check stop reason
            if response.stop_reason == "tool_use":
                tool_use_blocks = [
                    block for block in response.content if block.type == "tool_use"
                ]
                if len(tool_use_blocks) != 1:
                    raise ValueError(
                        f"Expected one tool use block, got {len(tool_use_blocks)} tool use blocks"
                    )
                # get new message with tool use result
                new_message = self._parse_tool_use(tool_use_blocks[0])
                return send_prompt(new_message)

            if response.stop_reason not in ["end_turn", "tool_use"]:
                self.logger.warning(
                    f"The LLM answer was stopped due to stop reason '{response.stop_reason}'."
                )


            # return the response
            response = self._parse_response(response)
            return response

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

        response = send_prompt(new_message)

        return response

    def _parse_tool_use(self, tool_use_block):
        """
        Parses a ToolUseBlock from the LLM and provides the new message to send to the model based on the tool result.

        Args:
            tool_use_block: The tool use block from the LLM response.

        Returns:
            The new message to send to the model.
        """
        # retrieve properties
        tool_name = tool_use_block.name
        tool_input = tool_use_block.input
        tool_use_id = tool_use_block.id

        if tool_use_block.name == "think":
            # return empty block to continue the conversation, tool does not provide a new message
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                }],
            }
        else:
            raise ValueError(f"Unknown tool name: {tool_name}")

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

    def _parse_response(self, response) -> LLMResponse:
        """
        Parses the response from the model and returns a LLMResponse object.
        Assumes that the response either ends with "DONE" or new optimizer parameters in the format [order, ell, rbendmin, t1].
        TODO: Parsing of badness criteria.

        Args:
            response (anthropic.Response): The response from the model.

        Returns:
            The parsed response as a LLMResponse object.
        """
        if response.content[-1].type == "text":
            text = response.content[-1].text
            # check if the response ends with "DONE"
            if text.strip().endswith("DONE"):
                return LLMResponse(None, BadnessCriteria(False, False, False, False))
            else:
                # Find all optimizer parameter matches
                matches = re.findall(
                    r"\[(\d+),\s*(-?[0-9.]+),\s*(-?[0-9.]+),\s*(-?[0-9.]+)\]", text
                )
                if matches:
                    last_match = matches[-1]
                    order = int(last_match[0])
                    ell = float(last_match[1])
                    rbendmin = float(last_match[2])
                    t1 = float(last_match[3])
                    return LLMResponse(
                        OptimizerParameters(order, ell, rbendmin, t1), None
                    )  # Placeholder for badness criteria
                else:
                    raise ValueError(
                        f"Could not find optimizer parameters in LLM response"
                    )

        else:
            raise ValueError(f"Unknown response format: {response.content}")

    def __format_message(self, message) -> str:
        """
        Formats a anthropic.ContentBlock to a string.

        Args:
            message (ContentBlock): The message to format.

        Returns:
            The formatted message.
        """
        if message.type == "thinking":
            return f"""-------------[THINKING]-------------
            {message.thinking}
            --------------------------------------
            """
        elif message.type == "text":
            return f"""-------------[MODEL RESPONSE]-------------
            {message.text}
            --------------------------------------
            """
        elif message.type == "tool_use":
            try:
                text = message.input["thought"]
            except KeyError:
                text = message.input
            return f"""-------------[{message.name}]-------------
            {text}
            --------------------------------------
            """
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
