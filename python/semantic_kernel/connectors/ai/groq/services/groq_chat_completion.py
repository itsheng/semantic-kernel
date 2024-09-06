# Copyright (c) Microsoft. All rights reserved.

import logging
import sys
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.utils.finish_reason import FinishReason

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from groq import AsyncGroq
from groq.types.chat.chat_completion import ChatCompletion, Choice
from groq.types.chat.chat_completion_chunk import ChatCompletionChunk
from groq.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from pydantic import ValidationError

from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.groq.services.groq_base import GroqBase
from semantic_kernel.connectors.ai.groq.settings.groq_settings import GroqSettings
from semantic_kernel.connectors.ai.groq.settings.prompt_execution_settings import GroqChatPromptExecutionSettings
from semantic_kernel.contents import AuthorRole
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.streaming_text_content import StreamingTextContent
from semantic_kernel.exceptions.service_exceptions import ServiceInitializationError, ServiceResponseException
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import trace_chat_completion

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

logger: logging.Logger = logging.getLogger(__name__)


class GroqChatCompletion(GroqBase, ChatCompletionClientBase):
    """Initializes a new instance of the GroqChatCompletion class."""

    def __init__(
        self,
        service_id: str | None = None,
        ai_model_id: str | None = None,
        api_key: str | None = None,
        async_client: AsyncGroq | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize a GroqChatCompletion service.

        Args:
            service_id (Optional[str]): Service ID tied to the execution settings. (Optional)
            ai_model_id (Optional[str]): The model name. (Optional)
            api_key (str | None): The optional API key to use. If provided will override,
                the env vars or .env file value.
            async_client (Optional[AsyncGroq]): A custom Groq client to use for the service. (Optional)
            env_file_path (str | None): Use the environment settings file as a fallback to using env vars.
            env_file_encoding (str | None): The encoding of the environment settings file, defaults to 'utf-8'.
        """
        try:
            groq_settings = GroqSettings.create(
                api_key=api_key,
                model=ai_model_id,
                env_file_path=env_file_path,
                env_file_encoding=env_file_encoding,
            )
        except ValidationError as ex:
            raise ServiceInitializationError("Failed to create Groq settings.", ex) from ex

        if not groq_settings.chat_model_id:
            raise ServiceInitializationError("The MistralAI chat model ID is required.")

        if not async_client:
            async_client = AsyncGroq(
                api_key=groq_settings.api_key.get_secret_value(),
            )

        super().__init__(
            service_id=service_id or groq_settings.chat_model_id,
            ai_model_id=groq_settings.chat_model_id,
            client=async_client,
        )

    @override
    @trace_chat_completion(GroqBase.MODEL_PROVIDER_NAME)
    async def get_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> list[ChatMessageContent]:
        """This is the method that is called from the kernel to get a response from a chat-optimized LLM.

        Args:
            chat_history (ChatHistory): A chat history that contains a list of chat messages,
                that can be rendered into a set of messages, from system, user, assistant and function.
            settings (PromptExecutionSettings): Settings for the request.
            kwargs (Dict[str, Any]): The optional arguments.

        Returns:
            List[ChatMessageContent]: A list of ChatMessageContent objects representing the response(s) from the LLM.
        """
        settings = self.get_prompt_execution_settings_from_settings(settings)
        prepared_chat_history = self._prepare_chat_history_for_request(chat_history)

        try:
            response = await self.client.chat.completions.create(
                model=self.ai_model_id,
                messages=prepared_chat_history,
                stream=True,
                **settings.prepare_settings_dict(),
            )
        except Exception as ex:
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the prompt",
                ex,
            ) from ex

        response_metadata = self._get_metadata_from_response(response)
        return [self._create_chat_message_content(response, choice, response_metadata) for choice in response.choices]

    async def get_streaming_chat_message_contents(
        self,
        chat_history: ChatHistory,
        settings: "PromptExecutionSettings",
        **kwargs: Any,
    ) -> AsyncGenerator[list[StreamingChatMessageContent], Any]:
        """Streams a text completion using a Groq model.

        Note that this method does not support multiple responses.

        Args:
            chat_history (ChatHistory): A chat history that contains a list of chat messages,
                that can be rendered into a set of messages, from system, user, assistant and function.
            settings (PromptExecutionSettings): Request settings.
            kwargs (Dict[str, Any]): The optional arguments.

        Yields:
            List[StreamingChatMessageContent]: Stream of StreamingChatMessageContent objects.
        """
        settings = self.get_prompt_execution_settings_from_settings(settings)
        prepared_chat_history = self._prepare_chat_history_for_request(chat_history)

        try:
            response = await self.client.chat.completions.create(
                model=self.ai_model_id,
                messages=prepared_chat_history,
                stream=True,
                **settings.prepare_settings_dict(),
            )
        except Exception as ex:
            raise ServiceResponseException(
                f"{type(self)} service failed to complete the prompt",
                ex,
            ) from ex

        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            chunk_metadata = self._get_metadata_from_response(chunk)
            yield [
                self._create_streaming_chat_message_content(chunk, choice, chunk_metadata) for choice in chunk.choices
            ]

    def _create_chat_message_content(
        self, response: ChatCompletion, choice: Choice, response_metadata: dict[str, Any]
    ) -> "ChatMessageContent":
        """Create a chat message content object from a choice."""
        metadata = self._get_metadata_from_response(response)
        metadata.update(response_metadata)

        items: list[Any] = []

        if choice.message.content:
            items.append(TextContent(text=choice.message.content))

        return ChatMessageContent(
            inner_content=response,
            ai_model_id=self.ai_model_id,
            metadata=metadata,
            role=AuthorRole(choice.message.role),
            items=items,
            finish_reason=FinishReason(choice.finish_reason) if choice.finish_reason else None,
        )

    def _create_streaming_chat_message_content(
        self,
        chunk: ChatCompletionChunk,
        choice: ChoiceChunk,
        chunk_metadata: dict[str, Any],
    ) -> StreamingChatMessageContent:
        """Create a streaming chat message content object from a choice."""
        metadata = self._get_metadata_from_response(chunk)
        metadata.update(chunk_metadata)

        items: list[Any] = []

        if choice.delta.content is not None:
            items.append(StreamingTextContent(choice_index=choice.index, text=choice.delta.content))

        return StreamingChatMessageContent(
            choice_index=choice.index,
            inner_content=chunk,
            ai_model_id=self.ai_model_id,
            metadata=metadata,
            role=AuthorRole(choice.delta.role) if choice.delta.role else AuthorRole.ASSISTANT,
            finish_reason=FinishReason(choice.finish_reason) if choice.finish_reason else None,
            items=items,
        )

    def _get_metadata_from_response(self, response: ChatCompletion | ChatCompletionChunk) -> dict[str, Any]:
        """Get metadata from the response.

        Args:
            response: The response from the service.

        Returns:
            A dictionary containing metadata.
        """
        metadata: dict[str, Any] = {
            "id": response.id,
            "model": response.model,
            "created": response.created,
            "usage": response.usage,
        }
        return metadata

    @override
    def get_prompt_execution_settings_class(self) -> type["PromptExecutionSettings"]:
        """Get the request settings class."""
        return GroqChatPromptExecutionSettings
