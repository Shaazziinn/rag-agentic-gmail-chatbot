import unittest
from unittest.mock import patch

from agent_utils import (
    classify_agent_intent,
    extract_json_object,
    format_message_context,
    run_agentic_chat,
    revise_email_draft,
    validate_email_draft,
)


class LlmResponse:
    def __init__(self, content):
        self.content = content


class FakeLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return LlmResponse(self.responses.pop(0))


class AgentUtilsTest(unittest.TestCase):
    def test_extract_json_object_reads_fenced_json(self):
        content = """
Here is the draft:

```json
{"to": "alex@example.com", "subject": "Hello", "body": "Hi Alex"}
```
"""

        result = extract_json_object(content)

        self.assertEqual(result["to"], "alex@example.com")
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Hi Alex")

    def test_validate_email_draft_requires_to_subject_and_body(self):
        errors = validate_email_draft(
            {
                "to": "",
                "subject": "Follow up",
                "body": "",
            }
        )

        self.assertEqual(errors, ["Missing recipient.", "Missing body."])

    def test_format_message_context_includes_headers_and_snippet(self):
        message = {
            "id": "msg-123",
            "threadId": "thread-456",
            "snippet": "Can we move the meeting to tomorrow?",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Alex <alex@example.com>"},
                    {"name": "Subject", "value": "Meeting"},
                    {"name": "Date", "value": "Wed, 8 Jul 2026"},
                ]
            },
        }

        context = format_message_context(message)

        self.assertIn("Message ID: msg-123", context)
        self.assertIn("Thread ID: thread-456", context)
        self.assertIn("From: Alex <alex@example.com>", context)
        self.assertIn("Subject: Meeting", context)
        self.assertIn("Snippet: Can we move the meeting", context)

    def test_classify_agent_intent_defaults_unknown_to_chat(self):
        llm = FakeLlm(['{"intent": "delete_email"}'])

        intent = classify_agent_intent("Delete old emails", llm)

        self.assertEqual(intent, "chat")

    def test_run_agentic_chat_answers_without_gmail_for_chat_intent(self):
        llm = FakeLlm(
            [
                '{"intent": "chat"}',
                "Agentic AI uses tools to take useful actions.",
            ]
        )

        result = run_agentic_chat(
            "What is agentic AI?",
            llm,
            service=object(),
        )

        self.assertEqual(result["type"], "chat")
        self.assertIn("tools", result["content"])

    @patch("agent_utils.get_message")
    @patch("agent_utils.search_messages")
    def test_run_agentic_chat_summarizes_gmail_for_search_intent(
        self,
        search_messages_mock,
        get_message_mock,
    ):
        search_messages_mock.return_value = [{"id": "msg-123"}]
        get_message_mock.return_value = {
            "id": "msg-123",
            "threadId": "thread-456",
            "snippet": "Interview review tomorrow.",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Vimal <vimal@example.com>"},
                    {"name": "Subject", "value": "Interview"},
                ]
            },
        }
        llm = FakeLlm(
            [
                '{"intent": "gmail_search"}',
                '{"query": "interview newer_than:30d"}',
                "You have an interview review email from Vimal.",
            ]
        )

        result = run_agentic_chat(
            "Summarize my latest interview emails.",
            llm,
            service=object(),
        )

        self.assertEqual(result["type"], "chat")
        self.assertEqual(result["query"], "interview newer_than:30d")
        self.assertIn("Vimal", result["context"])
        self.assertIn("interview review", result["content"])

    @patch("agent_utils.get_message")
    @patch("agent_utils.search_messages")
    def test_run_agentic_chat_returns_pending_draft_for_email_draft_intent(
        self,
        search_messages_mock,
        get_message_mock,
    ):
        search_messages_mock.return_value = [{"id": "msg-123"}]
        get_message_mock.return_value = {
            "id": "msg-123",
            "threadId": "thread-456",
            "snippet": "Please share the update.",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Alex <alex@example.com>"},
                    {"name": "Subject", "value": "Project update"},
                ]
            },
        }
        llm = FakeLlm(
            [
                '{"intent": "email_draft"}',
                '{"query": "from:alex@example.com newer_than:30d"}',
                (
                    '{"to": "alex@example.com", "subject": "Re: Project update", '
                    '"body": "Hi Alex, I will share the update soon.", '
                    '"rationale": "Replying to Alex about the project update."}'
                ),
            ]
        )

        result = run_agentic_chat(
            "Draft a reply to Alex saying I will share the update soon.",
            llm,
            service=object(),
        )

        self.assertEqual(result["type"], "email_draft")
        self.assertEqual(result["draft"]["to"], "alex@example.com")
        self.assertIn("Project update", result["context"])

    def test_revise_email_draft_preserves_required_fields(self):
        llm = FakeLlm(
            [
                (
                    '{"to": "hr@hivepro.com", '
                    '"subject": "Quoro-Hiring-Developer-2026", '
                    '"body": "Dear Hiring Team,\\n\\n202\\n\\nBest regards,\\nShazin Ab", '
                    '"rationale": "Added 202 to the body."}'
                )
            ]
        )
        existing_draft = {
            "to": "hr@hivepro.com",
            "subject": "Quoro-Hiring-Developer-2026",
            "body": "Dear Hiring Team,\n\nBest regards,\nShazin Ab",
            "rationale": "Original draft.",
        }

        revised = revise_email_draft(
            existing_draft,
            "In the body I want you to add 202",
            llm,
        )

        self.assertEqual(revised["to"], "hr@hivepro.com")
        self.assertEqual(revised["subject"], "Quoro-Hiring-Developer-2026")
        self.assertIn("202", revised["body"])


if __name__ == "__main__":
    unittest.main()
