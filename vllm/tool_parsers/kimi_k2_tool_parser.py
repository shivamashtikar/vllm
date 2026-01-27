# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# code modified from deepseekv3_tool_parser.py

from collections.abc import Sequence

# Using 'regex' package (not stdlib 're') for recursive pattern support (?&name)
# Required for nested JSON matching in fallback_tool_call_regex
import regex as re

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
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
        # Buffer size: increased to handle large tool call arguments (file paths,
        # nested JSON). 4096 bytes provides headroom for most practical cases.
        self.buffer_max_size: int = 4096
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
        self.tool_call_argument_token: str = "<|tool_call_argument_begin|>"

        # All markers that should be stripped from content
        # Includes tool markers and thinking markers that may leak
        self.all_tool_markers: list[str] = [
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_section_begin|>",
            "<|tool_call_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_end|>",
            "<|tool_call_argument_begin|>",
        ]

        # Regex pattern to strip thinking blocks: <think>...</think>
        # This handles both complete thinking blocks and partial markers
        self.thinking_pattern = re.compile(
            r"<think>.*?</think>|<think>|</think>",
            re.DOTALL
        )

        # Regex for tool call extraction - supports both:
        # - Kimi format: functions.Read:0
        # - OpenAI format: chatcmpl-tool-xxx
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>(?:(?!<\|tool_call_begin\|>).)*?)\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        # Fallback regex for tool calls WITHOUT special tokens
        # Matches: functions.tool_name:index {...}
        # This handles cases where the model outputs raw tool calls
        # Uses recursive pattern (?P<braces>...) to match nested JSON objects
        self.fallback_tool_call_regex = re.compile(
            r"functions\.(?P<function_name>[a-zA-Z_][a-zA-Z0-9_]*):(?P<index>\d+)\s*(?P<function_arguments>\{(?:[^{}]|(?P<braces>\{(?:[^{}]|(?&braces))*\}))*\})",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"(?P<tool_call_id>[^\s<]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>.*)"
        )

        # Fallback for streaming: detect raw tool call format
        self.stream_fallback_tool_call_regex = re.compile(
            r"functions\.(?P<function_name>[a-zA-Z_][a-zA-Z0-9_]*):(?P<index>\d+)\s*(?P<function_arguments>\{.*)?",
            re.DOTALL,
        )

        self.stream_tool_call_name_regex = re.compile(r"(?P<tool_call_id>[^\s<]+)\s*")

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

        # Log token ID info for debugging
        logger.info(
            "Kimi-K2 Tool parser initialized with token IDs: "
            "section_begin=%s, section_end=%s, call_begin=%s, call_end=%s",
            self.tool_calls_start_token_id,
            self.tool_calls_end_token_id,
            self.tool_call_start_token_id,
            self.tool_call_end_token_id,
        )

        # Warn if tool_call tokens are not found (parsing may fail)
        if self.tool_call_start_token_id is None:
            logger.warning(
                "tool_call_begin token '%s' not found in vocab - "
                "tool call parsing may fail",
                self.tool_call_start_token,
            )
        if self.tool_call_end_token_id is None:
            logger.warning(
                "tool_call_end token '%s' not found in vocab - "
                "tool call parsing may fail",
                self.tool_call_end_token,
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

    def _strip_all_tool_markers(self, text: str) -> str:
        """
        Strip ALL tool-related markers and thinking blocks from text.
        This prevents any marker from leaking into content.
        Also strips raw tool call format (functions.name:index {...}).
        """
        cleaned = text
        for marker in self.all_tool_markers:
            cleaned = cleaned.replace(marker, "")
        # Also strip thinking blocks that may leak into content
        cleaned = self.thinking_pattern.sub("", cleaned)
        # Strip raw tool calls (fallback format without special tokens)
        cleaned = self.fallback_tool_call_regex.sub("", cleaned)
        return cleaned

    def _contains_any_tool_marker(self, text: str) -> bool:
        """
        Check if text contains any tool-related marker.
        Used for text-based detection when token IDs don't match.
        Also detects raw tool call format (functions.name:index).
        """
        # Check for special token markers
        if any(marker in text for marker in self.all_tool_markers):
            return True
        # Check for raw tool call format (fallback)
        if self.stream_fallback_tool_call_regex.search(text):
            return True
        return False

    def _reset_section_state(self) -> None:
        """Reset state when exiting tool section."""
        self.in_tool_section = False
        self.token_buffer = ""
        self.section_char_count = 0

    def _infer_tool_name_from_args(
        self, tool_args: str, request: ChatCompletionRequest
    ) -> str | None:
        """
        Try to infer the tool name from the arguments by matching against
        available tools in the request. This is a fallback when the tool call
        ID doesn't contain the function name.

        Returns the tool name if a unique match is found, otherwise None.
        """
        import json as json_module

        if not request.tools:
            return None

        try:
            # Parse the arguments to extract parameter names
            args_dict = json_module.loads(tool_args) if tool_args else {}
            if not isinstance(args_dict, dict):
                return None

            arg_names = set(args_dict.keys())
            if not arg_names:
                return None

            # Find tools that have matching parameter names
            matching_tools = []
            for tool in request.tools:
                if not hasattr(tool, "function") or not tool.function:
                    continue
                func = tool.function
                if not hasattr(func, "parameters") or not func.parameters:
                    continue

                # Get required and optional parameter names from the tool
                params = func.parameters
                if isinstance(params, dict) and "properties" in params:
                    tool_params = set(params["properties"].keys())
                    # Check if all argument names exist in the tool's parameters
                    if arg_names.issubset(tool_params):
                        matching_tools.append(func.name)

            # Only return if we found exactly one matching tool
            if len(matching_tools) == 1:
                logger.info(
                    "Inferred tool name '%s' from arguments",
                    matching_tools[0],
                )
                return matching_tools[0]

            if len(matching_tools) > 1:
                logger.debug(
                    "Multiple tools match arguments: %s",
                    matching_tools,
                )

            return None

        except (json_module.JSONDecodeError, AttributeError, TypeError) as e:
            logger.debug("Failed to infer tool name from args: %s", e)
            return None

    def reset_streaming_state(self) -> None:
        """
        Reset all streaming state. Call this between requests to prevent
        state leakage when parser instance is reused.
        """
        # Reset section state
        self._reset_section_state()

        # Reset parent class state
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

        logger.debug("Streaming state reset")

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # Check for special token format first
        has_special_tokens = self.tool_calls_start_token in model_output

        # If no special tokens, try fallback raw format detection
        if not has_special_tokens:
            # Try to parse raw format: functions.tool_name:index {...}
            fallback_matches = list(self.fallback_tool_call_regex.finditer(model_output))
            if fallback_matches:
                logger.info(
                    "No special tokens found, but detected %d raw format tool call(s)",
                    len(fallback_matches)
                )
                tool_calls = []
                first_match_start = None
                for match in fallback_matches:
                    if first_match_start is None:
                        first_match_start = match.start()
                    function_name = match.group("function_name")
                    function_args = match.group("function_arguments")
                    index = match.group("index")
                    # Reconstruct the tool call ID in Kimi format
                    tool_call_id = f"functions.{function_name}:{index}"
                    tool_calls.append(
                        ToolCall(
                            id=tool_call_id,
                            type="function",
                            function=FunctionCall(
                                name=function_name, arguments=function_args
                            ),
                        )
                    )

                # Extract content before the first tool call
                content = model_output[:first_match_start] if first_match_start else None
                # Strip any remaining markers from content
                if content:
                    content = self._strip_all_tool_markers(content).strip()
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )
            else:
                # No tool calls found in any format
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

        # Process special token format
        else:
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
                    # Extract function name from tool call ID
                    # Supports:
                    # - Kimi format: functions.get_weather:0 -> get_weather
                    # - OpenAI format: chatcmpl-tool-xxx -> try to infer from args
                    if ":" in function_id:
                        # Kimi format: functions.name:index
                        function_name = function_id.split(":")[0].split(".")[-1]
                    else:
                        # ID doesn't contain ":", try to infer function name
                        function_name = self._infer_tool_name_from_args(
                            function_args, request
                        )
                        if not function_name:
                            if function_id.startswith("chatcmpl-tool-"):
                                logger.warning(
                                    "Tool call ID '%s' appears to be a generated ID "
                                    "without embedded function name. Unable to infer "
                                    "function name from arguments.",
                                    function_id[:50],
                                )
                                function_name = function_id
                            else:
                                # Warn about unexpected format - might be incomplete
                                logger.warning(
                                    "Tool call ID '%s' has unexpected format "
                                    "(expected functions.NAME:INDEX or chatcmpl-tool-*). "
                                    "Using ID as function name.",
                                    function_id[:50],
                                )
                                function_name = function_id
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
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

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

            # CRITICAL: If tool_call_begin is also in this delta, initialize
            # the tool call state now since we'll return early
            if self.tool_call_start_token_id in delta_token_ids:
                logger.debug("Initializing tool call state (tool_call_begin in same delta)")
                self.current_tool_id += 1
                self.current_tool_name_sent = False
                self.streamed_args_for_tool.append("")

            # CRITICAL: Extract content that appeared BEFORE the section marker
            # and return it as reasoning content. This prevents content from
            # leaking into the tool call stream.
            content_before_marker = None
            for variant in self.tool_calls_start_token_variants:
                if variant in delta_text:
                    marker_pos = delta_text.find(variant)
                    if marker_pos > 0:
                        content_before_marker = delta_text[:marker_pos]
                    break

            if content_before_marker and content_before_marker.strip():
                logger.debug(
                    "Returning content before tool section: '%s'",
                    content_before_marker[:50] if len(content_before_marker) > 50
                    else content_before_marker
                )
                return DeltaMessage(content=content_before_marker)
            # If no content before marker, continue to process tool calls
            # but don't return content from this delta
            return None
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
                # CRITICAL: Strip ALL markers from remaining text
                remaining = self._strip_all_tool_markers(buffered_text)
                self._reset_section_state()
                # Return remaining text as reasoning content if non-empty
                if remaining.strip():
                    return DeltaMessage(content=remaining)
                # Return None to skip empty chunks (avoids "(no content)" in clients)
                return None
        else:
            self.token_buffer = buffered_text

        # Check if any variant of section start token is in current_token_ids
        has_section_token = any(
            tid in current_token_ids for tid in self.tool_calls_start_token_ids
        )

        # Also check for tool_call_begin token directly (model may skip section wrapper)
        has_tool_call_token = (
            self.tool_call_start_token_id is not None
            and self.tool_call_start_token_id in current_token_ids
        )

        # CRITICAL FIX: Also check for text-based markers
        # Token IDs may not match if tokenizer splits markers differently
        has_text_markers = self._contains_any_tool_marker(delta_text)
        has_text_markers_in_buffer = self._contains_any_tool_marker(self.token_buffer)

        # Early return: if no section/tool token detected yet, return as reasoning content
        # But ONLY if there are no text-based markers either
        if (not has_section_token and not has_tool_call_token
            and not self.in_tool_section
            and not has_text_markers
            and not has_text_markers_in_buffer):
            logger.debug("No tool call tokens found!")
            # Don't clear buffer - it needs to accumulate partial markers across deltas
            # Buffer overflow is already protected by lines 215-224
            return DeltaMessage(content=delta_text)

        # If we detected markers via text but not token IDs, log warning and enter tool section
        if has_text_markers and not has_section_token and not has_tool_call_token:
            # Check if this is raw format (functions.name:index) vs special token format
            raw_format_match = self.stream_fallback_tool_call_regex.search(
                self.token_buffer
            )
            if raw_format_match:
                logger.info(
                    "Detected raw format tool call in buffer: functions.%s:%s",
                    raw_format_match.group("function_name"),
                    raw_format_match.group("index"),
                )
                # Enter tool section and initialize tool call state
                if not self.in_tool_section:
                    self.in_tool_section = True
                    self.section_char_count = 0
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool.append("")

                    # Extract content before the raw tool call
                    match_start = raw_format_match.start()
                    # Find the start in the original buffer
                    buffer_before_match = self.token_buffer[:match_start]
                    if buffer_before_match.strip():
                        # Clear buffer of processed content
                        self.token_buffer = self.token_buffer[match_start:]
                        cleaned_content = self._strip_all_tool_markers(buffer_before_match)
                        if cleaned_content.strip():
                            return DeltaMessage(content=cleaned_content)

                # Try to parse the raw format tool call from buffer
                function_name = raw_format_match.group("function_name")
                index = raw_format_match.group("index")
                tool_args = raw_format_match.group("function_arguments") or ""
                tool_id = f"functions.{function_name}:{index}"

                # Check if we have complete JSON arguments
                if tool_args and tool_args.strip().endswith("}"):
                    # Complete tool call - emit it
                    if not self.current_tool_name_sent:
                        self.current_tool_name_sent = True
                        current_tool_call = {
                            "id": tool_id,
                            "name": function_name,
                            "arguments": tool_args,
                        }
                        if len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append(current_tool_call)
                        else:
                            self.prev_tool_call_arr[self.current_tool_id] = current_tool_call
                        self.streamed_args_for_tool[self.current_tool_id] = tool_args

                        # Clear buffer and reset state
                        self.token_buffer = ""
                        self._reset_section_state()

                        return DeltaMessage(
                            tool_calls=[
                                DeltaToolCall(
                                    index=self.current_tool_id,
                                    type="function",
                                    id=tool_id,
                                    function=DeltaFunctionCall(
                                        name=function_name,
                                        arguments=tool_args,
                                    ).model_dump(exclude_none=True),
                                )
                            ]
                        )
                # Incomplete - wait for more tokens
                return None
            else:
                logger.warning(
                    "Tool markers detected in text but not in token IDs. "
                    "This may indicate tokenizer mismatch. Markers: %s",
                    [m for m in self.all_tool_markers if m in delta_text]
                )
                # Enter tool section to handle properly
                if not self.in_tool_section:
                    self.in_tool_section = True
                    self.section_char_count = 0

        # Enter tool section if we see tool tokens (even without section wrapper)
        if has_tool_call_token and not self.in_tool_section:
            logger.debug("Detected tool_call_begin without section wrapper, entering tool section")
            self.in_tool_section = True
            self.section_char_count = 0

            # CRITICAL: Initialize tool call state since we'll return early
            logger.debug("Initializing tool call state (tool_call_begin without section)")
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool.append("")

            # CRITICAL: Extract content that appeared BEFORE the tool_call_begin marker
            if self.tool_call_start_token in delta_text:
                marker_pos = delta_text.find(self.tool_call_start_token)
                if marker_pos > 0:
                    content_before_marker = delta_text[:marker_pos]
                    if content_before_marker.strip():
                        logger.debug(
                            "Returning content before tool_call_begin: '%s'",
                            content_before_marker[:50] if len(content_before_marker) > 50
                            else content_before_marker
                        )
                        return DeltaMessage(content=content_before_marker)
            # If no content before marker, continue but don't return content
            return None

        # Strip section markers from delta_text for subsequent processing
        # NOTE: This preprocessing happens BEFORE the regex-based tool call
        # parsing (from PR #24847) to ensure markers are removed cleanly
        # before pattern matching. We only strip section markers here,
        # NOT tool_call markers which are needed for parsing.
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
                # CRITICAL: Strip all markers before returning as content
                cleaned_text = self._strip_all_tool_markers(delta_text)
                # Return remaining content as reasoning, or None for empty content
                return DeltaMessage(content=cleaned_text) if cleaned_text.strip() else None

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

            # Debug logging for tool call state
            if self.in_tool_section or cur_tool_start_count > 0:
                logger.debug(
                    "Tool call state: in_section=%s, prev_start=%d, prev_end=%d, "
                    "cur_start=%d, cur_end=%d",
                    self.in_tool_section,
                    prev_tool_start_count,
                    prev_tool_end_count,
                    cur_tool_start_count,
                    cur_tool_end_count,
                )

            # case: if we're generating text, OR rounding out a tool call
            if (
                cur_tool_start_count == cur_tool_end_count
                and prev_tool_end_count == cur_tool_end_count
                and self.tool_call_end_token not in delta_text
            ):
                # CRITICAL FIX: Suppress ALL content while in tool section
                # This prevents tool call tokens from leaking into content
                if self.in_tool_section:
                    logger.debug(
                        "In tool section, suppressing content: %s",
                        delta_text[:50] if len(delta_text) > 50 else delta_text,
                    )
                    # Return None to skip this chunk
                    return None
                # CRITICAL: Strip any markers that might have leaked through
                cleaned_text = self._strip_all_tool_markers(delta_text)
                if not cleaned_text.strip():
                    return None
                logger.debug(
                    "Returning as content (text gen case): '%.50s...'",
                    cleaned_text,
                )
                return DeltaMessage(content=cleaned_text)

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
                if (self.prev_tool_call_arr is None
                    or len(self.prev_tool_call_arr) == 0
                    or self.current_tool_id >= len(self.prev_tool_call_arr)):
                    logger.debug("attempting to close tool call, but no tool call or invalid index")
                    # Handle deferred section exit before returning
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None
                diff = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                if not diff:
                    # No arguments to stream, just close the tool call
                    logger.debug("closing tool call with no remaining arguments to stream")
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None
                # diff is truthy at this point
                diff = (
                    diff.encode("utf-8").decode("unicode_escape")
                    if isinstance(diff, str)
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
                    # Return None to skip this chunk (avoids "(no content)" in clients)
                    return None
                # CRITICAL: Strip ALL tool markers, not just start/end
                text = self._strip_all_tool_markers(delta_text)
                # Skip empty content after stripping
                if not text.strip():
                    if deferred_section_exit and self.in_tool_section:
                        self._reset_section_state()
                    return None
                delta = DeltaMessage(content=text)
                # Handle deferred section exit before returning
                if deferred_section_exit and self.in_tool_section:
                    self._reset_section_state()
                return delta

            current_tool_call = dict()
            if tool_call_portion:
                # Strip leading/trailing whitespace to fix regex matching
                tool_call_portion = tool_call_portion.strip()
                current_tool_call_matches = self.stream_tool_call_portion_regex.match(
                    tool_call_portion
                )
                if current_tool_call_matches:
                    tool_id, tool_args = current_tool_call_matches.groups()
                    # Extract function name - supports both formats
                    if ":" in tool_id:
                        tool_name = tool_id.split(":")[0].split(".")[-1]
                    else:
                        # ID doesn't contain ":", try to infer function name
                        # from available tools by matching arguments
                        tool_name = self._infer_tool_name_from_args(
                            tool_args, request
                        )
                        if not tool_name:
                            # Fallback: use a cleaned version of the ID
                            # Remove common prefixes like "chatcmpl-tool-"
                            if tool_id.startswith("chatcmpl-tool-"):
                                logger.warning(
                                    "Tool call ID '%s' appears to be a generated ID "
                                    "without embedded function name. Unable to infer "
                                    "function name from arguments.",
                                    tool_id[:50],
                                )
                                # Use the ID as-is but log warning
                                tool_name = tool_id
                            else:
                                tool_name = tool_id
                    current_tool_call["id"] = tool_id
                    current_tool_call["name"] = tool_name
                    current_tool_call["arguments"] = tool_args
                else:
                    # Try fallback regex for raw format (functions.name:index {...})
                    fallback_match = self.stream_fallback_tool_call_regex.match(
                        tool_call_portion
                    )
                    if fallback_match:
                        function_name = fallback_match.group("function_name")
                        index = fallback_match.group("index")
                        tool_args = fallback_match.group("function_arguments") or ""
                        tool_id = f"functions.{function_name}:{index}"
                        current_tool_call["id"] = tool_id
                        current_tool_call["name"] = function_name
                        current_tool_call["arguments"] = tool_args
                        logger.debug(
                            "Matched raw format tool call: %s with args: %s",
                            tool_id, tool_args[:50] if tool_args else ""
                        )
                    else:
                        current_tool_call_name_matches = (
                            self.stream_tool_call_name_regex.match(tool_call_portion)
                        )
                        if current_tool_call_name_matches:
                            (tool_id_str,) = current_tool_call_name_matches.groups()
                            # Extract function name - supports both formats
                            if ":" in tool_id_str:
                                tool_name = tool_id_str.split(":")[0].split(".")[-1]
                            elif tool_id_str.startswith("chatcmpl-tool-"):
                                # OpenAI format ID - use as-is
                                tool_name = tool_id_str
                            else:
                                # CRITICAL FIX: If tool_id doesn't have expected
                                # format (functions.NAME:INDEX or chatcmpl-tool-*),
                                # wait for more tokens. This prevents sending
                                # incomplete IDs like just "functions".
                                logger.debug(
                                    "Incomplete tool ID '%s', waiting for more tokens",
                                    tool_id_str[:30]
                                )
                                return None
                            current_tool_call["id"] = tool_id_str
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

                    # CRITICAL: Initialize prev_tool_call_arr before returning
                    # This prevents IndexError on subsequent iterations
                    if len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append(current_tool_call)
                    else:
                        self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

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
                if text_portion is not None:
                    # CRITICAL: Strip any markers before returning as content
                    cleaned_text = self._strip_all_tool_markers(delta_text)
                    if cleaned_text.strip():
                        delta = DeltaMessage(content=cleaned_text)
                    else:
                        delta = None
                else:
                    delta = None
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
            if self.current_tool_id < len(self.prev_tool_call_arr):
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
            # Handle deferred section exit before returning
            if deferred_section_exit and self.in_tool_section:
                self._reset_section_state()
            return None  # do not stream a delta. skip this token ID.
