"""GLM-5.3-Flash in Strata: its tool-call form (GlmParser), its template, and stop tokens that are one token.
    python -m unittest serve.test_glm"""
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, GlmParser, glm_call_to_qwen, openai_to_messages
from serve.server import stop_ids_of

TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}}}}]
CALL = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>")


def run(text, step=None, reason="stop"):
    p = GlmParser(tools=TOOLS)
    evs = []
    for piece in [text] if step is None else [text[i:i + step] for i in range(0, len(text), step)]:
        evs += p.feed(piece)
    evs += p.finish(reason)
    join = lambda kind: "".join(e.text for e in evs if e.kind == kind)
    return join("reasoning"), join("content"), [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class Parser(unittest.TestCase):
    def test_thinking_text_and_a_call_whole_and_char_by_char(self):
        for step in (None, 1, 4):          # the prompt ends with <think>: the reply starts inside the thinking
            self.assertEqual(run("Need the weather.</think>Let me check." + CALL, step),
                             ("Need the weather.", "Let me check.", [("get_weather", {"city": "Paris", "days": 3})]))

    def test_two_calls(self):
        two = CALL + CALL.replace("Paris", "Rome")
        self.assertEqual([c[1]["city"] for c in run("ok</think>" + two, 2)[2]], ["Paris", "Rome"])

    def test_text_that_only_looks_like_a_tag_is_text(self):
        self.assertEqual(run("hm</think>a <tool_ x and <tool_call is a tag name", 1)[1],
                         "a <tool_ x and <tool_call is a tag name")

    def test_a_cut_call_is_not_a_call(self):
        _, c, calls = run("ok</think>" + CALL[:-len("</tool_call>")], reason="length")
        self.assertEqual(calls, [])
        self.assertIn("Paris", c)

    def test_a_held_call_is_not_a_clean_point(self):
        p = GlmParser(tools=TOOLS)
        p.feed("ok</think><tool_call>get_wea")
        self.assertTrue(p.buf)              # the server must not close the thinking or stop here

    def test_glm_call_to_qwen_leaves_text_alone(self):
        self.assertEqual(glm_call_to_qwen("<tool_call>not a call\nat all</tool_call>"),
                         "<tool_call>not a call\nat all</tool_call>")


class Template(unittest.TestCase):
    def setUp(self):
        self.t = ChatTemplate(Path(__file__).parent / "fixtures" / "glm-5.3-chat_template.jinja")

    def test_a_tool_round_trip(self):
        req = {"messages": [{"role": "user", "content": "Weather in Paris?"},
                            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                            {"role": "tool", "tool_call_id": "c1", "content": "18 C"}],
               "tools": [{"type": "function", "function": TOOLS[0]}]}
        messages, tools, _ = openai_to_messages(req)
        out = self.t.render(messages, tools=tools, reasoning_effort="low")
        self.assertTrue(out.startswith("[gMASK]<sop><|system|>Reasoning Effort: Low"))
        self.assertIn("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
                      "<|observation|><tool_response>18 C</tool_response>", out)
        self.assertTrue(out.endswith("<|assistant|><think>"))

    def test_capabilities(self):
        self.assertTrue(all(self.t.caps.values()), self.t.caps)


class StopIds(unittest.TestCase):
    def test_only_strings_that_are_one_token(self):
        class Tok:
            def encode(self, s, parse_special=False):
                return {"<|endoftext|>": [154820], "<|user|>": [154827]}.get(s, [60, 91, 318, 6043, 91, 29])
        self.assertEqual(stop_ids_of(Tok(), ["<|endoftext|>", "<|user|>", "<|im_end|>"]), {154820, 154827})


if __name__ == "__main__":
    unittest.main()
