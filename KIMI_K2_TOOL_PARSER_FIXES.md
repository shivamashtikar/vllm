# Kimi-K2 Tool Parser Fixes

**Date:** 2023-12-23
**Branch:** pr-24847
**vLLM Version:** 0.14.0rc1.dev85+g369cfcee0

## Overview

This document describes the issues encountered and fixes applied to the Kimi-K2 tool parser and related vLLM components during debugging of tool call streaming functionality.

---

## Issue 1: Circular Import Errors

### Symptoms
After syncing with main branch and building vLLM, the server failed to start with `ImportError`:

```
ImportError: cannot import name 'SamplingParams' from partially initialized module 'vllm'
ImportError: cannot import name 'PoolingParams' from partially initialized module 'vllm'
ImportError: cannot import name '__version__' from partially initialized module 'vllm'
```

### Root Cause
Several files were importing from the top-level `vllm` module, which caused circular import issues during module initialization.

### Resolution
Changed imports to use specific submodules:

| File | Before | After |
|------|--------|-------|
| `vllm/config/vllm.py` | `from vllm import __version__` | `from vllm._version import __version__` |
| `vllm/v1/sample/logits_processor/builtin.py` | `from vllm import SamplingParams` | `from vllm.sampling_params import SamplingParams` |
| `vllm/v1/sample/logits_processor/interface.py` | `from vllm import SamplingParams` | `from vllm.sampling_params import SamplingParams` |
| `vllm/entrypoints/pooling/classify/protocol.py` | `from vllm import PoolingParams` | `from vllm.pooling_params import PoolingParams` |
| `vllm/entrypoints/pooling/embed/protocol.py` | `from vllm import PoolingParams` | `from vllm.pooling_params import PoolingParams` |
| `vllm/entrypoints/pooling/pooling/protocol.py` | `from vllm import PoolingParams` | `from vllm.pooling_params import PoolingParams` |
| `vllm/entrypoints/pooling/score/protocol.py` | `from vllm import PoolingParams` | `from vllm.pooling_params import PoolingParams` |

---

## Issue 2: "(no content)" Tokens in Client Output

### Symptoms
Intermittent "(no content)" tokens appeared in the client output during tool call streaming.

### Root Cause
The tool parser was returning `DeltaMessage(content="")` (empty string) instead of `None` when there was no content to stream. Some clients interpret empty content as "(no content)".

### Resolution
Changed all instances of `return DeltaMessage(content="")` to `return None` in `kimi_k2_tool_parser.py`. When `None` is returned, the streaming loop skips that chunk entirely.

---

## Issue 3: Tool Call ID Format Mismatch

### Symptoms
Tool calls failed to parse when the model generated OpenAI-format tool call IDs like `chatcmpl-tool-abc123` instead of Kimi-format IDs like `functions.Read:0`.

### Root Cause
The regex patterns expected the format `[^<]+:\d+` which requires a colon followed by digits:

```python
# Old pattern - only matched Kimi format
r"(?P<tool_call_id>[^<]+:\d+)"
```

### Resolution
Changed regex patterns to accept any non-whitespace, non-angle-bracket characters:

```python
# New pattern - matches both formats
r"(?P<tool_call_id>[^\s<]+)"
```

Updated patterns in:
- `self.tool_call_regex` (line 65-68)
- `self.stream_tool_call_portion_regex` (line 70-72)
- `self.stream_tool_call_name_regex` (line 74)

Also updated function name extraction to handle both formats:
```python
if ":" in function_id:
    # Kimi format: functions.get_weather:0 -> get_weather
    function_name = function_id.split(":")[0].split(".")[-1]
else:
    # OpenAI format: chatcmpl-tool-xxx -> use ID as-is
    function_name = function_id
```

---

## Issue 4: Whitespace Breaking Regex Matching

### Symptoms
Tool call parsing failed intermittently with "Not enough token" debug messages.

### Root Cause
The `tool_call_portion` string had leading/trailing whitespace that prevented regex patterns from matching.

### Resolution
Added `.strip()` before regex matching:

```python
if tool_call_portion:
    tool_call_portion = tool_call_portion.strip()  # Added this line
    current_tool_call_matches = self.stream_tool_call_portion_regex.match(
        tool_call_portion
    )
```

---

## Issue 5: Content Leaking into Tool Call Stream

### Symptoms
Raw tool call tokens like `<|tool_call_begin|>` appeared in the client's content stream, mixed with regular assistant text.

Example output:
```
I'll help you remove the web search toggle...  <|tool_call_begin|> functions.Task:0 <|tool_call_argument_begin|>
```

### Root Cause
When content appeared BEFORE the tool section marker in the same streaming delta, the content was not being extracted separately. The entire delta (including markers) was being processed together.

### Resolution
Added content extraction logic when entering tool section:

```python
# When section_begin is detected
if found_section_begin and not self.in_tool_section:
    self.in_tool_section = True

    # Initialize tool call state if tool_call_begin is in same delta
    if self.tool_call_start_token_id in delta_token_ids:
        self.current_tool_id += 1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool.append("")

    # Extract content BEFORE the section marker
    for variant in self.tool_calls_start_token_variants:
        if variant in delta_text:
            marker_pos = delta_text.find(variant)
            if marker_pos > 0:
                content_before_marker = delta_text[:marker_pos]
                break

    if content_before_marker and content_before_marker.strip():
        return DeltaMessage(content=content_before_marker)
    return None
```

Similar logic was added for `<|tool_call_begin|>` without section wrapper.

---

## Issue 6: Content Not Suppressed Inside Tool Section

### Symptoms
After entering the tool section, content tokens were still being returned to the client instead of being suppressed.

### Root Cause
The content suppression logic only checked if tool call counts were balanced, not whether we were actually inside a tool section:

```python
# Old logic - didn't check in_tool_section
if cur_tool_start_count == cur_tool_end_count:
    return DeltaMessage(content=delta_text)  # Leaked content!
```

### Resolution
Added explicit check for `in_tool_section`:

```python
if (cur_tool_start_count == cur_tool_end_count
    and prev_tool_end_count == cur_tool_end_count
    and self.tool_call_end_token not in delta_text):
    # CRITICAL: Suppress ALL content while in tool section
    if self.in_tool_section:
        return None  # Don't leak content
    return DeltaMessage(content=delta_text)
```

---

## Issue 7: IndexError on prev_tool_call_arr

### Symptoms
`IndexError: list index out of range` when accessing `prev_tool_call_arr[current_tool_id]`.

### Root Cause
When sending the tool name, the parser returned early without populating `prev_tool_call_arr`, causing the next iteration to fail when accessing the array.

### Resolution
Added initialization of `prev_tool_call_arr` before returning:

```python
if function_name:
    self.current_tool_name_sent = True

    # CRITICAL: Initialize prev_tool_call_arr before returning
    if len(self.prev_tool_call_arr) <= self.current_tool_id:
        self.prev_tool_call_arr.append(current_tool_call)
    else:
        self.prev_tool_call_arr[self.current_tool_id] = current_tool_call

    return DeltaMessage(tool_calls=[...])
```

---

## Files Modified

| File | Changes |
|------|---------|
| `vllm/config/vllm.py` | Fixed `__version__` import |
| `vllm/v1/sample/logits_processor/builtin.py` | Fixed `SamplingParams` import |
| `vllm/v1/sample/logits_processor/interface.py` | Fixed `SamplingParams` import |
| `vllm/entrypoints/pooling/classify/protocol.py` | Fixed `PoolingParams` import |
| `vllm/entrypoints/pooling/embed/protocol.py` | Fixed `PoolingParams` import |
| `vllm/entrypoints/pooling/pooling/protocol.py` | Fixed `PoolingParams` import |
| `vllm/entrypoints/pooling/score/protocol.py` | Fixed `PoolingParams` import |
| `vllm/tool_parsers/kimi_k2_tool_parser.py` | Multiple tool parsing fixes |

---

## Testing

### Unit Tests
Created manual tests to verify:
1. Content before section marker is extracted correctly
2. Tool names are sent with proper IDs
3. Arguments are parsed and streamed
4. Section state is reset after tool calls complete
5. Both Kimi-format and OpenAI-format tool call IDs work

### Test Results
```
✓ Content extraction: "I'll help you! " extracted before tool section
✓ Tool name: "Read" sent with index=0
✓ Arguments: {"file_path": "/test.py"} parsed correctly
✓ Section reset: in_tool_section=False after completion
```

---

## Chat Template Reference

The Kimi-K2 chat template (`/data/kimi-k2/chat_template.jinja`) defines the tool call format:

```jinja
{%- macro render_toolcalls(message) -%}
  <|tool_calls_section_begin|>
  {%- for tool_call in message['tool_calls'] -%}
    {%- set formatted_id = tool_call['id'] -%}
    <|tool_call_begin|>{{ formatted_id }}<|tool_call_argument_begin|>...arguments...<|tool_call_end|>
  {%- endfor -%}
  <|tool_calls_section_end|>
{%- endmacro -%}
```

Token markers:
- `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` - Wrap all tool calls
- `<|tool_call_begin|>` / `<|tool_call_end|>` - Wrap individual tool calls
- `<|tool_call_argument_begin|>` - Separates tool ID from arguments

---

## Issue 8: Token ID vs Text Mismatch Causing Marker Leakage

### Symptoms
Tool markers like `<|tool_call_begin|>` appeared in content even when the parser was supposed to suppress them.

### Root Cause
The early return check only verified token IDs, not text content:
```python
# Old logic - only checked token IDs
if not has_section_token and not has_tool_call_token and not self.in_tool_section:
    return DeltaMessage(content=delta_text)  # Markers leaked!
```

When the tokenizer splits markers into multiple tokens (e.g., `<|` + `tool_call_begin` + `|>`), the token ID check passes but the text still contains the marker string.

### Resolution
Added text-based marker detection alongside token ID checks:
```python
# New logic - also checks text content
has_text_markers = self._contains_any_tool_marker(delta_text)
has_text_markers_in_buffer = self._contains_any_tool_marker(self.token_buffer)

if (not has_section_token and not has_tool_call_token
    and not self.in_tool_section
    and not has_text_markers
    and not has_text_markers_in_buffer):
    return DeltaMessage(content=delta_text)
```

---

## Issue 9: Incomplete Marker Stripping

### Symptoms
Some tool markers and `<think>` blocks appeared in client output.

### Root Cause
The parser only stripped `tool_call_start_token` and `tool_call_end_token`, missing:
- `<|tool_call_argument_begin|>`
- Section markers when returning content
- Thinking blocks (`<think>...</think>`)

### Resolution
Created comprehensive marker stripping:
```python
self.all_tool_markers: list[str] = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_section_begin|>",
    "<|tool_call_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
]

self.thinking_pattern = re.compile(
    r"<think>.*?</think>|<think>|</think>",
    re.DOTALL
)

def _strip_all_tool_markers(self, text: str) -> str:
    cleaned = text
    for marker in self.all_tool_markers:
        cleaned = cleaned.replace(marker, "")
    cleaned = self.thinking_pattern.sub("", cleaned)
    return cleaned
```

All content return points now use this comprehensive stripping function.

---

## Issue 10: Model Outputting Raw Tool Calls Without Special Tokens

### Symptoms
The model outputs tool calls as raw text like:
```
functions.read_file:0  {"path": "/test.py"}
```

Instead of the expected format with special tokens:
```
<|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>{"path":"/test.py"}<|tool_call_end|>
```

### Root Cause
The model may not have been trained or prompted to use the special tokens for tool calls.

### Resolution
Two fixes applied:

1. **Chat Template Update** (`chat_template.jinja`):
   Added explicit instructions in the tool declaration section:
   ```jinja
   When you need to call a tool, you MUST use this exact format:
   <|tool_calls_section_begin|><|tool_call_begin|>TOOL_CALL_ID<|tool_call_argument_begin|>{"arg1":"value1"}<|tool_call_end|><|tool_calls_section_end|>
   ```

2. **Fallback Detection and Stripping** (`kimi_k2_tool_parser.py`):
   Added regex patterns to detect and strip raw tool call format:
   ```python
   self.fallback_tool_call_regex = re.compile(
       r"(functions\.(?P<function_name>[a-zA-Z_][a-zA-Z0-9_]*):(?P<index>\d+))\s*(?P<function_arguments>\{[^}]*\})",
       re.DOTALL,
   )
   ```
   - `_contains_any_tool_marker()` now also detects raw format
   - `_strip_all_tool_markers()` now also strips raw format from content

---

## Issue 11: Fallback Parsing Not Implemented (Only Detection/Stripping)

### Symptoms
Tool calls in raw format (`functions.read_file:0 {"path": "..."}`) were detected and stripped from content, but never actually parsed into proper tool call objects. The client received malformed tool_use blocks with `id: "functions"` and `name: "functions"`.

### Root Cause
The fallback regex patterns (`fallback_tool_call_regex` and `stream_fallback_tool_call_regex`) were only used for:
1. Detection via `_contains_any_tool_marker()`
2. Stripping via `_strip_all_tool_markers()`

They were NOT used for actual parsing and returning `ToolCall` objects.

### Resolution
Added fallback parsing support in both non-streaming and streaming methods:

1. **Non-streaming (`extract_tool_calls`)**: When no special tokens are found, try `fallback_tool_call_regex` to parse raw format:
   ```python
   if not has_special_tokens:
       fallback_matches = list(self.fallback_tool_call_regex.finditer(model_output))
       if fallback_matches:
           # Parse each match into ToolCall objects
           for match in fallback_matches:
               function_name = match.group("function_name")
               function_args = match.group("function_arguments")
               index = match.group("index")
               tool_call_id = f"functions.{function_name}:{index}"
               tool_calls.append(ToolCall(...))
   ```

2. **Streaming (`extract_tool_calls_streaming`)**: When raw format is detected via text markers, parse and emit tool calls:
   ```python
   raw_format_match = self.stream_fallback_tool_call_regex.search(self.token_buffer)
   if raw_format_match:
       # Enter tool section, initialize state
       # Parse function name, index, arguments
       # Emit DeltaMessage with tool_calls
   ```

3. **Improved fallback regex** to handle nested JSON using recursive patterns:
   ```python
   self.fallback_tool_call_regex = re.compile(
       r"functions\.(?P<function_name>[a-zA-Z_][a-zA-Z0-9_]*):(?P<index>\d+)\s*"
       r"(?P<function_arguments>\{(?:[^{}]|(?P<braces>\{(?:[^{}]|(?&braces))*\}))*\})",
       re.DOTALL,
   )
   ```

---

## Issue 12: Agent Stops Mid-Task Without Calling Tools

### Symptoms
The agent successfully calls tools initially, receives results, outputs reasoning content, but then stops instead of continuing with more tool calls. The model outputs content like "I'll explore the codebase..." but never actually generates tool call tokens.

### Root Cause
The model is outputting DESCRIPTIONS of what it plans to do rather than actually generating tool call tokens. This is a model behavior issue:
1. Model generates `<think>I should call Grep...</think>`
2. Model generates "Based on my exploration, I found..." (content)
3. Model generates EOS token
4. Response ends with `finish_reason: "stop"`, no tool calls

The chat template was only declaring tools as JSON without any usage guidance, so the model didn't understand it should USE the tools, not just describe actions.

### Resolution
Updated the chat template to add subtle tool usage guidance in the system message when tools are provided:

```jinja
{%- if messages|length == 0 or messages[0]['role'] != 'system' -%}
  {%- if tools -%}
  <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI. Use the available tools to complete tasks - call tools directly rather than describing what you would do.<|im_end|>
  {%- else -%}
  <|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>
  {%- endif -%}
{%- endif -%}
```

This is less explicit than format instructions (which the model was echoing back) but provides actionable guidance about tool usage behavior.

---

## Known Remaining Issues

1. **Model Behavior**: The model sometimes generates malformed tool calls (missing `<|tool_call_begin|>` tokens) or repeats itself. Behavior may vary based on prompting and context.

2. **Singular vs Plural Variants**: The parser supports both `<|tool_call_section_begin|>` (singular) and `<|tool_calls_section_begin|>` (plural) variants for compatibility.

3. **Thinking Block Tool Calls**: If the model outputs tool calls inside `<think>...</think>` blocks, the reasoning parser captures them before the tool parser can see them.

4. **Custom System Messages**: If the user provides a custom system message, the tool usage hint is not added. Consider adding tool guidance to your custom system message.

---

## Recommendations

1. **Restart vLLM server** after applying these changes (Python changes are picked up by editable install, but a restart ensures clean state).

2. **Monitor logs** with `VLLM_LOGGING_LEVEL=DEBUG` to see tool parsing state transitions.

3. **Check client configuration** if raw tokens still appear - the client may need configuration to properly render tool calls.

4. **Include tool guidance in custom system messages** if you provide your own system message, add a hint like "Use tools directly rather than describing actions."
