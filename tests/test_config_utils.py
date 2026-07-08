import os
import unittest
from unittest.mock import patch

from config_utils import get_config_value


class ConfigUtilsTest(unittest.TestCase):
    def test_get_config_value_reads_environment_first(self):
        secrets = {"GROQ_API_KEY": "secret-key"}

        with patch.dict(os.environ, {"GROQ_API_KEY": "env-key"}):
            value = get_config_value("GROQ_API_KEY", secrets)

        self.assertEqual(value, "env-key")

    def test_get_config_value_reads_secrets_when_environment_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            value = get_config_value(
                "GROQ_API_KEY",
                {"GROQ_API_KEY": "secret-key"},
            )

        self.assertEqual(value, "secret-key")

    def test_get_config_value_returns_none_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            value = get_config_value("GROQ_API_KEY", {})

        self.assertIsNone(value)


if __name__ == "__main__":
    unittest.main()
