import base64
import unittest
from email import message_from_bytes

from gmail_utils import build_email_message


class GmailUtilsTest(unittest.TestCase):
    def test_build_email_message_creates_gmail_raw_payload(self):
        payload = build_email_message(
            to="receiver@example.com",
            subject="Interview demo",
            body="Hello from the agentic chatbot.",
        )

        self.assertIn("raw", payload)
        decoded = base64.urlsafe_b64decode(payload["raw"].encode("utf-8"))
        message = message_from_bytes(decoded)

        self.assertEqual(message["To"], "receiver@example.com")
        self.assertEqual(message["Subject"], "Interview demo")
        self.assertEqual(
            message.get_payload(),
            "Hello from the agentic chatbot.",
        )


if __name__ == "__main__":
    unittest.main()
