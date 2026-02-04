# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

from .identity_reasoning_parser import IdentityReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
else:
    ChatCompletionRequest = Any


logger = init_logger(__name__)


class KimiK2ReasoningParser(ReasoningParser):
    """
    Kimi K2 parser that delegates to either DeepSeekR1ReasoningParser or
    IdentityReasoningParser based on `thinking` and `separate_reasoning`.

    Unlike DeepSeekV3ReasoningParser which defaults to NOT thinking,
    KimiK2ReasoningParser defaults to thinking mode (uses DeepSeekR1ReasoningParser).

    This parser also filters out "(no content)" placeholder text that the
    Kimi K2 model sometimes generates when making tool calls without text.
    """

    # Tool markers that should be stripped from reasoning content
    TOOL_MARKERS = [
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_section_begin|>",
        "<|tool_call_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
    ]

    def _clean_content(self, text: str | None) -> str | None:
        """
        Clean content by stripping "(no content)" placeholder.
        Returns None if content is empty after cleaning.
        """
        if text is None:
            return None
        cleaned = text.replace("(no content)", "")
        if not cleaned.strip():
            return None
        return cleaned

    def _clean_reasoning(self, text: str | None) -> str | None:
        """
        Clean reasoning content by stripping tool markers and placeholders.
        The model sometimes outputs tool markers inside reasoning blocks.
        Returns None if reasoning is empty after cleaning.
        """
        if text is None:
            return None

        cleaned = text
        # Strip tool markers that may leak into reasoning
        for marker in self.TOOL_MARKERS:
            if marker in cleaned:
                # Remove everything from the first tool marker onwards
                # since that's likely tool call content, not reasoning
                marker_pos = cleaned.find(marker)
                cleaned = cleaned[:marker_pos]
                break

        # Also strip "(no content)" placeholder
        cleaned = cleaned.replace("(no content)", "")

        if not cleaned.strip():
            return None
        return cleaned.strip()

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.pop("chat_template_kwargs", {}) or {}
        # Key difference: default to True instead of False
        thinking = bool(chat_kwargs.pop("thinking", True))

        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, *args, **kwargs)
        else:
            self._parser = IdentityReasoningParser(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        return self._parser.is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: list[int], delta_ids: list[int]
    ) -> bool:
        return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return self._parser.extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest"
    ) -> tuple[str | None, str | None]:
        reasoning, content = self._parser.extract_reasoning(model_output, request)
        # Clean reasoning (strip tool markers) and content (strip placeholders)
        return self._clean_reasoning(reasoning), self._clean_content(content)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        result = self._parser.extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

        # Filter placeholders and tool markers from the result
        if result is None:
            return None

        # Clean both reasoning and content fields
        cleaned_reasoning = self._clean_reasoning(result.reasoning)
        cleaned_content = self._clean_content(result.content)

        # Check if anything changed
        reasoning_changed = result.reasoning != cleaned_reasoning
        content_changed = result.content != cleaned_content

        # If nothing changed, return original
        if not reasoning_changed and not content_changed:
            return result

        # If both are now empty, return None
        if cleaned_reasoning is None and cleaned_content is None:
            # Check if there are tool calls to return
            if result.tool_calls:
                return DeltaMessage(tool_calls=result.tool_calls)
            return None

        # Return cleaned result
        return DeltaMessage(
            content=cleaned_content,
            reasoning=cleaned_reasoning,
            tool_calls=result.tool_calls if result.tool_calls else [],
        )