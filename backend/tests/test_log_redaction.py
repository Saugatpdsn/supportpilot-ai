import logging

from app.guardrails import RedactingFilter, redact


def test_api_keys_are_redacted():
    out = redact("using AIzaSyA1234567890abcdefghijklmnop for the call")
    assert "AIza" not in out
    assert "[REDACTED_API_KEY]" in out


def test_credential_assignments_are_redacted():
    assert "hunter2" not in redact("password: hunter2")
    assert "abc123" not in redact("GOOGLE_API_KEY=abc123")


def test_emails_are_redacted():
    assert "jane@example.com" not in redact("customer jane@example.com asked")


def test_filter_redacts_formatted_log_records():
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "key %s", ("AIzaSyA1234567890abcdefghijklmnop",), None
    )
    assert RedactingFilter().filter(record)
    assert "AIza" not in record.getMessage()
    assert "[REDACTED_API_KEY]" in record.getMessage()