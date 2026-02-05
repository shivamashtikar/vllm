# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

from collections.abc import Sequence

import json

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    ToolParser,
)

logger = init_logger(__name__)


class KimiK2ToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)
        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[
            str
        ] = []  # map what has been streamed for each tool so far to a list

        # Section-level state management to prevent token leakage
        self.in_tool_section: bool = False
        self.token_buffer: str = ""
        # Buffer size: empirical worst-case for longest marker (~30 chars) * 2
        # + safety margin for unicode + partial overlap. Prevents unbounded growth.
        self.buffer_max_size: int = 1024
        self.section_char_count: int = 0  # Track characters processed in tool section
        self.max_section_chars: int = 8192  # Force exit if section exceeds this
        self._buffer_overflow_logged: bool = False  # Log overflow once per session

        # Support both singular and plural variants
        self.tool_calls_start_token: str = "<|tool_calls_section_begin|>"
        self.tool_calls_end_token: str = "<|tool_calls_section_end|>"
        self.tool_calls_start_token_variants: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_section_begin|>",  # singular variant
        ]
        self.tool_calls_end_token_variants: list[str] = [
            "<|tool_calls_section_end|>",
            "<|tool_call_section_end|>",  # singular variant
        ]

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^<]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>.+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>.+:\d+)\s*")

        # Fallback regex for raw tool call format without special tokens
        # Pattern to detect the START of a raw tool call (before JSON args)
        # Matches: functions.name:id or name:id followed by optional whitespace and {
        self.raw_tool_call_start_regex = re.compile(
            r"(?:functions\.)?(?P<function_name>[\w_]+):(?P<tool_index>\d+)\s*\{",
        )

        # Pattern to extract function name and index from partial match
        self.raw_tool_call_header_regex = re.compile(
            r"(?:functions\.)?(?P<function_name>[\w_]+):(?P<tool_index>\d+)",
        )

        # Patterns to detect early signs of a raw tool call (for early buffering)
        # These detect partial prefixes that might become tool calls
        self.raw_tool_call_prefixes = [
            "functions.",
            "functions",
            "function",
            "functio",
            "functi",
            "funct",
            "func",
        ]

        # Regex to detect "functions.X" or "functions.X:" patterns (partial tool calls)
        self.raw_tool_call_partial_regex = re.compile(
            r"functions\.[\w_]+:?$|[\w_]+:$",
        )

        # State for raw tool call parsing
        self.in_raw_tool_call: bool = False
        self.potential_raw_tool_call: bool = False  # For early prefix detection
        self.raw_tool_call_buffer: str = ""
        self.raw_tool_call_name: str | None = None
        self.raw_tool_call_index: str | None = None
        self.raw_tool_brace_depth: int = 0

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )
        self.tool_calls_start_token_id = self.vocab.get(self.tool_calls_start_token)
        self.tool_calls_end_token_id = self.vocab.get(self.tool_calls_end_token)

        # Get token IDs for all variants
        self.tool_calls_start_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_start_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]
        self.tool_calls_end_token_ids: list[int] = [
            tid
            for variant in self.tool_calls_end_token_variants
            if (tid := self.vocab.get(variant)) is not None
        ]

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if (
            self.tool_calls_start_token_id is None
            or self.tool_calls_end_token_id is None
        ):
            raise RuntimeError(
                "Kimi-K2 Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

    def _check_and_strip_markers(self, text: str) -> tuple[str, bool, bool]:
        """
        Check for section begin/end markers in text and strip them.
        Returns: (cleaned_text, found_section_begin, found_section_end)
        """
        found_begin = False
        found_end = False
        cleaned = text

        # Check for section begin markers (any variant)
        for variant in self.tool_calls_start_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_begin = True

        # Check for section end markers (any variant)
        for variant in self.tool_calls_end_token_variants:
            if variant in cleaned:
                cleaned = cleaned.replace(variant, "")
                found_end = True
        return cleaned, found_begin, found_end

    def _reset_section_state(self) -> None:
        """Reset state when exiting tool section."""
        self.in_tool_section = False
        self.token_buffer = ""
        self.section_char_count = 0

    def reset_streaming_state(self) -> None:
        """
        Reset all streaming state. Call this between requests to prevent
        state leakage when parser instance is reused.
        """
        # Reset section state
        self._reset_section_state()

        # Reset raw tool call state
        self.in_raw_tool_call = False
        self.potential_raw_tool_call = False
        self.raw_tool_call_buffer = ""
        self.raw_tool_call_name = None
        self.raw_tool_call_index = None
        self.raw_tool_brace_depth = 0

        # Reset parent class state
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        logger.debug("Streaming state reset")

    def _find_json_end(self, text: str, start_idx: int) -> int:
        """
        Find the end of a JSON object starting at start_idx.
        Uses bracket counting that handles strings with escaped characters.
        Returns the index after the closing brace, or -1 if not complete.
        """
        if start_idx >= len(text) or text[start_idx] != "{":
            return -1

        depth = 0
        in_string = False
        escape_next = False
        i = start_idx

        while i < len(text):
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == "\\":
                escape_next = True
                i += 1
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return i + 1  # Return index after closing brace

            i += 1

        return -1  # JSON not complete

    def _extract_raw_tool_calls(
        self, model_output: str
    ) -> ExtractedToolCallInformation:
        """
        Fallback extraction for raw tool call format without special tokens.
        Handles format like: functions.name:id{...} or name:id{...}
        Uses proper JSON boundary detection for nested objects.
        """
        try:
            tool_calls = []
            first_match_start = -1
            search_start = 0

            while True:
                # Find the start of a raw tool call
                match = self.raw_tool_call_start_regex.search(
                    model_output, pos=search_start
                )
                if not match:
                    break

                if first_match_start == -1:
                    first_match_start = match.start()

                function_name = match.group("function_name")
                tool_index = match.group("tool_index")

                # Find where the JSON starts (the { in the match)
                json_start = match.end() - 1  # -1 because { is included in match

                # Find the end of the JSON object
                json_end = self._find_json_end(model_output, json_start)
                if json_end == -1:
                    # JSON not complete, skip this match
                    logger.debug(
                        "Raw tool call JSON not complete for %s:%s",
                        function_name,
                        tool_index,
                    )
                    search_start = match.end()
                    continue

                function_args = model_output[json_start:json_end]

                # Validate JSON
                try:
                    json.loads(function_args)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Invalid JSON in raw tool call arguments: %s (error: %s)",
                        function_args[:100],
                        str(e),
                    )
                    search_start = json_end
                    continue

                tool_id = f"functions.{function_name}:{tool_index}"
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        type="function",
                        function=FunctionCall(
                            name=function_name, arguments=function_args
                        ),
                    )
                )

                search_start = json_end

            if tool_calls:
                content = model_output[:first_match_start].strip() or None
                logger.debug(
                    "Extracted %d tool calls using raw format fallback",
                    len(tool_calls),
                )
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content,
                )

            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        except Exception:
            logger.exception("Error in extracting raw tool calls.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def _extract_raw_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """
        Handle streaming extraction for raw tool call format.
        This is used when model outputs tool calls without special tokens.
        Uses bracket counting to detect complete JSON objects.
        """
        try:
            # If we're in potential mode (prefix detected), check if it's confirmed
            if self.potential_raw_tool_call and not self.in_raw_tool_call:
                # Check if we now have the full pattern
                match = self.raw_tool_call_start_regex.search(self.token_buffer)
                if match:
                    # Confirmed! Transition to in_raw_tool_call
                    self.potential_raw_tool_call = False
                    self.in_raw_tool_call = True
                    self.raw_tool_call_name = match.group("function_name")
                    self.raw_tool_call_index = match.group("tool_index")
                    json_start = match.end() - 1
                    self.raw_tool_call_buffer = self.token_buffer[json_start:]
                    self.raw_tool_brace_depth = 0

                    # Count braces in what we have so far
                    in_string = False
                    escape_next = False
                    for char in self.raw_tool_call_buffer:
                        if escape_next:
                            escape_next = False
                            continue
                        if char == "\\":
                            escape_next = True
                            continue
                        if char == '"':
                            in_string = not in_string
                        elif not in_string:
                            if char == "{":
                                self.raw_tool_brace_depth += 1
                            elif char == "}":
                                self.raw_tool_brace_depth -= 1

                    logger.debug(
                        "Confirmed raw tool call: %s:%s, brace_depth=%d",
                        self.raw_tool_call_name,
                        self.raw_tool_call_index,
                        self.raw_tool_brace_depth,
                    )
                    # Continue to process as in_raw_tool_call below
                elif self.raw_tool_call_header_regex.search(self.token_buffer):
                    # Still looks like it could be a tool call, keep waiting
                    return DeltaMessage(content="")
                elif self.raw_tool_call_partial_regex.search(self.token_buffer):
                    # Partial pattern detected, keep waiting
                    return DeltaMessage(content="")
                else:
                    # Check if we still have a prefix
                    buffer_end = self.token_buffer[-20:] if len(self.token_buffer) > 20 else self.token_buffer
                    has_prefix = any(buffer_end.endswith(p) for p in self.raw_tool_call_prefixes)
                    if has_prefix:
                        # Still have a prefix, keep waiting
                        return DeltaMessage(content="")
                    # The potential tool call didn't pan out
                    # This shouldn't happen often, but release buffered content
                    logger.debug("Potential raw tool call did not confirm, releasing buffer")
                    self.potential_raw_tool_call = False
                    # Return the delta as content since it's not a tool call
                    return DeltaMessage(content=delta_text)

            # If we're not in a raw tool call, check if one is starting
            if not self.in_raw_tool_call:
                match = self.raw_tool_call_start_regex.search(self.token_buffer)
                if match:
                    self.in_raw_tool_call = True
                    self.raw_tool_call_name = match.group("function_name")
                    self.raw_tool_call_index = match.group("tool_index")
                    # Start buffering from the { onwards
                    json_start = match.end() - 1
                    self.raw_tool_call_buffer = self.token_buffer[json_start:]
                    self.raw_tool_brace_depth = 0

                    # Count braces in what we have so far
                    in_string = False
                    escape_next = False
                    for char in self.raw_tool_call_buffer:
                        if escape_next:
                            escape_next = False
                            continue
                        if char == "\\":
                            escape_next = True
                            continue
                        if char == '"':
                            in_string = not in_string
                        elif not in_string:
                            if char == "{":
                                self.raw_tool_brace_depth += 1
                            elif char == "}":
                                self.raw_tool_brace_depth -= 1

                    logger.debug(
                        "Started raw tool call: %s:%s, brace_depth=%d",
                        self.raw_tool_call_name,
                        self.raw_tool_call_index,
                        self.raw_tool_brace_depth,
                    )

                    # Suppress content - we're in tool call mode
                    return DeltaMessage(content="")
                else:
                    # No tool call detected yet, but check for partial match
                    # Look for patterns like "functions." or "Edit:" that might
                    # indicate an incoming tool call
                    if self.raw_tool_call_header_regex.search(self.token_buffer):
                        # Partial match - suppress output and wait for more
                        return DeltaMessage(content="")
                    return None

            # We're in a raw tool call - accumulate and check for completion
            # Add delta to the JSON buffer
            self.raw_tool_call_buffer += delta_text

            # Update brace depth for new content
            in_string = False
            escape_next = False

            # We need to track string state from the beginning of buffer
            for char in delta_text:
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                elif not in_string:
                    if char == "{":
                        self.raw_tool_brace_depth += 1
                    elif char == "}":
                        self.raw_tool_brace_depth -= 1

            logger.debug(
                "Raw tool call buffer update: depth=%d, buffer_len=%d",
                self.raw_tool_brace_depth,
                len(self.raw_tool_call_buffer),
            )

            # Check if JSON is complete (all braces closed)
            if self.raw_tool_brace_depth == 0 and "{" in self.raw_tool_call_buffer:
                # JSON is complete
                function_args = self.raw_tool_call_buffer.strip()

                # Validate JSON
                try:
                    json.loads(function_args)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Invalid JSON in raw tool call: %s (error: %s)",
                        function_args[:100],
                        str(e),
                    )
                    # Reset state
                    self.in_raw_tool_call = False
                    self.raw_tool_call_buffer = ""
                    return None

                tool_id = f"functions.{self.raw_tool_call_name}:{self.raw_tool_call_index}"

                # Check if this is a new tool call or continuation
                if not self.current_tool_name_sent:
                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.streamed_args_for_tool.append(function_args)

                    logger.debug(
                        "Emitting complete raw tool call: %s with args length %d",
                        tool_id,
                        len(function_args),
                    )

                    # Reset raw tool call state for next potential tool call
                    self.in_raw_tool_call = False
                    self.raw_tool_call_buffer = ""

                    # Emit the complete tool call
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(
                                    name=self.raw_tool_call_name,
                                    arguments=function_args,
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )

                # Reset for next tool call
                self.in_raw_tool_call = False
                self.raw_tool_call_buffer = ""
                self.current_tool_name_sent = False
                return None

            # JSON not complete yet, suppress output
            return DeltaMessage(content="")

        except Exception:
            logger.exception("Error in raw tool call streaming extraction.")
            self.in_raw_tool_call = False
            self.raw_tool_call_buffer = ""
            return None

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # First, try to extract using special tokens (preferred format)
        if self.tool_calls_start_token in model_output:
            try:
                # there are two possible captures - between tags, or between a
                # tag and end-of-string so the result of
                # findall is an array of tuples where one is a function call and
                # the other is None
                function_call_tuples = self.tool_call_regex.findall(model_output)

                logger.debug("function_call_tuples: %s", function_call_tuples)

                tool_calls = []
                for match in function_call_tuples:
                    function_id, function_args = match
                    # function_id: functions.get_weather:0 or get_weather:0
                    function_name = function_id.split(":")[0].split(".")[-1]
                    tool_calls.append(
                        ToolCall(
                            id=function_id,
                            type="function",
                            function=FunctionCall(
                                name=function_name, arguments=function_args
                            ),
                        )
                    )

                content = model_output[: model_output.find(self.tool_calls_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception("Error in extracting tool call from response.")
                # Fall through to try raw format

        # Fallback: try to extract raw format tool calls (functions.X:N{...})
        # This handles cases where model outputs without special tokens
        return self._extract_raw_tool_calls(model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        logger.debug("delta_text: %s", delta_text)
        logger.debug("delta_token_ids: %s", delta_token_ids)

        # Flag to defer section exit until after tool parsing completes
        deferred_section_exit = False

        # Add delta to buffer for split marker detection
        self.token_buffer += delta_text

        # Enforce buffer size limit to prevent memory issues
        if len(self.token_buffer) > self.buffer_max_size:
            if not self._buffer_overflow_logged:
                logger.warning(
                    "Token buffer exceeded max size (%d bytes), flushing excess. "
                    "This may indicate very long markers or unusual tokenization.",
                    self.buffer_max_size,
                )
                self._buffer_overflow_logged = True
            # Keep only the most recent content that might contain partial markers
            self.token_buffer = self.token_buffer[-self.buffer_max_size // 2 :]

        # Check buffer for section markers (handles split tokens)
        buffered_text, found_section_begin, found_section_end = (
            self._check_and_strip_markers(self.token_buffer)
        )

        # Track section state transitions
        if found_section_begin and not self.in_tool_section:
            logger.debug("Entering tool section")
            self.in_tool_section = True
            self.token_buffer = buffered_text  # Use cleaned buffer
            self.section_char_count = 0  # Reset counter for new section

        if found_section_end and self.in_tool_section:
            logger.debug("Detected section end marker")
            # CRITICAL: Don't exit early if tool_call_end is in this chunk.
            # Tool parser must emit final arguments/close first to avoid dropping
            # the final tool update and leaking tokens into reasoning channel.
            has_tool_end = self.tool_call_end_token_id in delta_token_ids
            if has_tool_end:
                # Defer exit until after tool parsing completes
                deferred_section_exit = True
                logger.debug("Deferring section exit: tool_call_end in same chunk")
                self.token_buffer = buffered_text
            else:
                # No tool call ending, safe to exit immediately
                logger.debug("Exiting tool section")
                self._reset_section_state()
                # Extract any content AFTER the section end marker in delta_text
                # (don't use buffered_text as it contains tool call data)
                post_section_content = ""
                for variant in self.tool_calls_end_token_variants:
                    if variant in delta_text:
                        parts = delta_text.split(variant, 1)
                        if len(parts) > 1:
                            post_section_content = parts[1]
                        break
                if post_section_content.strip():
                    return DeltaMessage(content=post_section_content)
                return DeltaMessage(content="")
        else:
            self.token_buffer = buffered_text

        # Check if any variant of section start token is in current_token_ids
        has_section_token = any(
            tid in current_token_ids for tid in self.tool_calls_start_token_ids
        )

        # Check for raw tool call pattern in buffer (fallback for models that
        # don't use special tokens). Check for:
        # 1. Already in raw tool call mode
        # 2. Complete raw tool call start pattern (functions.X:N{)
        # 3. Partial pattern that could become a tool call (functions.X:N)
        # 4. Early prefix that might become a tool call (functions. or Edit:)
        has_raw_tool_call = self.in_raw_tool_call or self.potential_raw_tool_call

        if not has_raw_tool_call:
            # Check for complete or partial patterns
            if self.raw_tool_call_start_regex.search(self.token_buffer):
                has_raw_tool_call = True
            elif self.raw_tool_call_header_regex.search(self.token_buffer):
                has_raw_tool_call = True
            elif self.raw_tool_call_partial_regex.search(self.token_buffer):
                # Partial pattern like "functions.Edit" or "functions.Edit:"
                self.potential_raw_tool_call = True
                has_raw_tool_call = True
                logger.debug(
                    "Detected partial raw tool call pattern in buffer"
                )
            else:
                # Check for early prefixes at end of buffer
                buffer_end = self.token_buffer[-20:] if len(self.token_buffer) > 20 else self.token_buffer
                for prefix in self.raw_tool_call_prefixes:
                    if buffer_end.endswith(prefix):
                        # Early prefix detected - start buffering
                        self.potential_raw_tool_call = True
                        has_raw_tool_call = True
                        logger.debug(
                            "Detected potential raw tool call prefix: %s",
                            prefix,
                        )
                        break

        # Early return: if no section token and no raw tool call detected yet,
        # return as content
        if not has_section_token and not self.in_tool_section and not has_raw_tool_call:
            logger.debug("No tool call tokens found!")
            # Don't clear buffer - it needs to accumulate partial markers across deltas
            # Buffer overflow is already protected by lines 215-224
            return DeltaMessage(content=delta_text)

        # If we detected a raw tool call pattern (or are in raw mode) but no
        # special tokens, handle via raw extraction
        if has_raw_tool_call and not has_section_token and not self.in_tool_section:
            logger.debug(
                "Processing raw tool call: in_raw=%s, buffer_len=%d",
                self.in_raw_tool_call,
                len(self.token_buffer),
            )
            # For streaming with raw format, we'll collect in buffer and emit
            # tool calls when we detect a complete JSON object
            result = self._extract_raw_tool_calls_streaming(
                previous_text, current_text, delta_text
            )
            if result is not None:
                return result
            # If extraction returned None, continue with normal flow
            return None

        # Strip section markers from delta_text for subsequent processing
        # NOTE: This preprocessing happens BEFORE the regex-based tool call
        # parsing (from PR #24847) to ensure markers are removed cleanly
        # before pattern matching. No double-stripping occurs because
        # section markers and tool call markers are distinct.
        delta_text, _, _ = self._check_and_strip_markers(delta_text)

        # Error recovery: If in tool section for too long, force exit
        if self.in_tool_section:
            self.section_char_count += len(delta_text)
            if self.section_char_count > self.max_section_chars:
                logger.warning(
                    "Tool section exceeded max length (%d chars), forcing exit. "
                    "This may indicate malformed model output.",
                    self.max_section_chars,
                )
                self._reset_section_state()
                # Deferred exit already handled by forced exit above
                # Return remaining content as reasoning (or empty delta if no content)
                return DeltaMessage(content=delta_text if delta_text.strip() else "")

        try:
            # figure out where we are in the parsing by counting tool call
            # start & end tags
            prev_tool_start_count = previous_token_ids.count(
                self.tool_call_start_token_id
            )
            prev_tool_end_count = previous_token_ids.count(self.tool_call_end_token_id)
            cur_tool_start_count = current_token_ids.count(
                self.tool_call_start_token_id
            )
            cur_tool_end_count = current_token_ids.count(self.tool_call_end_token_id)
            tool_call_portion = None
            text_portion = None

            # case: if we're generating text, OR rounding out a tool call
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                # Suppress content between section begin and first tool begin
                # (header noise). Don't suppress content between tools to avoid
                # breaking potential delimiter characters.
                if self.in_tool_section and cur_tool_start_count == 0:
                    logger.debug(
                        "In tool section before first tool, suppressing: %s",
                        delta_text,
                    )
                    # Return empty delta to maintain iterator contract
                    return DeltaMessage(content="")
                logger.debug("Generating text content! skipping tool parsing.")
                return DeltaMessage(content=delta_text)

            if self.tool_call_end_token in delta_text:
                logger.debug("tool_call_end_token in delta_text")
                full_text = current_text + delta_text
                tool_call_portion = (
                    full_text.split(self.tool_call_start_token)[-1]
                    .split(self.tool_call_end_token)[0]
                    .rstrip()
                )
                delta_text = delta_text.split(self.tool_call_end_token)[0].rstrip()
                text_portion = delta_text.split(self.tool_call_end_token)[-1].lstrip()

            # case -- we're starting a new tool call
            if (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count > prev_tool_start_count
            ):
                if len(delta_token_ids) > 1:
                    tool_call_portion = current_text.split(self.tool_call_start_token)[
                        -1
                    ]
                else:
                    tool_call_portion = None
                    delta = None

                text_portion = None

                # set cursors and state appropriately
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")
                logger.debug("Starting on a new tool %s", self.current_tool_id)

            # case -- we're updating an existing tool call
            elif (
                cur_tool_start_count > cur_tool_end_count
                and cur_tool_start_count == prev_tool_start_count
            ):
                # get the portion of the text that's the tool call
                tool_call_portion = current_text.split(self.tool_call_start_token)[-1]
                text_portion = None

            # case -- the current tool call is being closed.
            elif (
                cur_tool_start_count == cur_tool_end_count
                and cur_tool_end_count >= prev_tool_end_count
            ):
                if self.prev_tool_call_arr is None or len(self.prev_tool_call_arr) == 0:
                    logger.debug("attempting to close tool call, but no tool call")
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if diff:
                    diff = (
                        diff.encode("utf-8").decode("unicode_escape")
                        if diff is str
                        else diff
                    )
                    if '"}' not in delta_text:
                        # Handle deferred section exit before returning
                        if deferred_section_exit and self.in_tool_section:
                            self._reset_section_state()
                        return None
                    end_loc = delta_text.rindex('"}')
                    diff = delta_text[:end_loc] + '"}'
                    logger.debug(
                        "Finishing tool and found diff that had not "
                        "been streamed yet: %s",
                        diff,
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += diff
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        logger.debug("Completing deferred section exit")
                        self._reset_section_state()
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(arguments=diff).model_dump(
                                    exclude_none=True
                                ),
                            )
                        ]
                    )

            # case -- otherwise we're just generating text
            else:
                # Check if we're in tool section - if so, suppress
                if self.in_tool_section:
                    logger.debug("In tool section, suppressing text generation")
                    # Handle deferred section exit before returning
                    if deferred_section_exit:
                        self._reset_section_state()
                    return DeltaMessage(content="")
                text = delta_text.replace(self.tool_call_start_token, "")
                text = text.replace(self.tool_call_end_token, "")
                delta = DeltaMessage(tool_calls=[], content=text)
                # Handle deferred section exit before returning
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return delta

            current_tool_call = dict()
            if tool_call_portion:
                current_tool_call_matches = self.stream_tool_call_portion_regex.match(
                    tool_call_portion
                )
                if current_tool_call_matches:
                    tool_id, tool_args = current_tool_call_matches.groups()
                    tool_name = tool_id.split(":")[0].split(".")[-1]
                    current_tool_call["id"] = tool_id.strip()
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    current_tool_call_name_matches = (
                        self.stream_tool_call_name_regex.match(tool_call_portion)
                    )
                    if current_tool_call_name_matches:
                        (tool_id_str,) = current_tool_call_name_matches.groups()
                        tool_name = tool_id_str.split(":")[0].split(".")[-1]
                        current_tool_call["id"] = tool_id_str.strip()
                        current_tool_call["name"] = tool_name
                        current_tool_call["arguments"] = ""
                    else:
                        logger.debug("Not enough token")
                        return None

            # case - we haven't sent the tool name yet. If it's available, send
            #   it. otherwise, wait until it's available.
            if not self.current_tool_name_sent:
                if current_tool_call is None:
                    return None
                function_name: str | None = current_tool_call.get("name")
                tool_id = current_tool_call.get("id")
                if function_name:
                    self.current_tool_name_sent = True
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                type="function",
                                id=tool_id,
                                function=DeltaFunctionCall(
                                    name=function_name
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                else:
                    return None

            # case -- otherwise, send the tool call delta

            # if the tool call portion is None, send the delta as text
            if tool_call_portion is None:
                # if there's text but not tool calls, send that -
                # otherwise None to skip chunk
                # CRITICAL: Never return content if we're in a tool section
                if self.in_tool_section:
                    return None
                delta = (
                    DeltaMessage(content=delta_text)
                    if text_portion is not None
                    else None
                )
                return delta

            # now, the nitty-gritty of tool calls
            # now we have the portion to parse as tool call.

            logger.debug(
                "Trying to parse current tool call with ID %s", self.current_tool_id
            )

            # if we're starting a new tool call, push an empty object in as
            #   a placeholder for the arguments
            if len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})

            # main logic for tool parsing here - compare prev. partially-parsed
            #   JSON to the current partially-parsed JSON
            prev_arguments = self.prev_tool_call_arr[self.current_tool_id].get(
                "arguments"
            )
            cur_arguments = current_tool_call.get("arguments")

            logger.debug("diffing old arguments: %s", prev_arguments)
            logger.debug("against new ones: %s", cur_arguments)

            # case -- no arguments have been created yet. skip sending a delta.
            if not cur_arguments and not prev_arguments:
                logger.debug("Skipping text %s - no arguments", delta_text)
                delta = None

            # case -- prev arguments are defined, but non are now.
            #   probably impossible, but not a fatal error - just keep going
            elif not cur_arguments and prev_arguments:
                logger.error(
                    "should be impossible to have arguments reset "
                    "mid-call. skipping streaming anything."
                )
                delta = None

            # case -- we now have the first info about arguments available from
            #   autocompleting the JSON
            elif cur_arguments and not prev_arguments:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=cur_arguments
                            ).model_dump(exclude_none=True),
                        )
                    ]
                )
                self.streamed_args_for_tool[self.current_tool_id] = cur_arguments

            # last case -- we have an update to existing arguments.
            elif cur_arguments and prev_arguments:
                if (
                    isinstance(delta_text, str)
                    and cur_arguments != prev_arguments
                    and len(cur_arguments) > len(prev_arguments)
                    and cur_arguments.startswith(prev_arguments)
                ):
                    delta_arguments = cur_arguments[len(prev_arguments) :]
                    logger.debug("got diff %s", delta_text)

                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    arguments=delta_arguments
                                ).model_dump(exclude_none=True),
                            )
                        ]
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = cur_arguments
                else:
                    delta = None

            # handle saving the state for the current tool into
            # the "prev" list for use in diffing for the next iteration
            if self.current_tool_id == len(self.prev_tool_call_arr) - 1:
                self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
            else:
                self.prev_tool_call_arr.append(current_tool_call)

            # Handle deferred section exit after tool parsing completes
            if deferred_section_exit and self.in_tool_section:
                logger.debug("Completing deferred section exit")
                self._reset_section_state()

            return delta

        except Exception:
            logger.exception("Error trying to handle streaming tool call.")
            return None  # do not stream a delta. skip this token ID.
