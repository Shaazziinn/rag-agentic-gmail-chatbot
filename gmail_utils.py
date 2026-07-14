import base64
import os
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]


def build_email_message(to, subject, body):
    message = MIMEText(body)
    message["To"] = to
    message["Subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return {"raw": raw}


def _token_info_from_secrets():
    """Return the authorized-user token dict from Streamlit secrets, or None.

    Streamlit Community Cloud has no local token.json and an ephemeral disk, so
    the token (the full contents of token.json) is stored in st.secrets under a
    [gmail_token] section instead. This is imported lazily so the module still
    works outside a Streamlit runtime (e.g. in tests).
    """
    try:
        import streamlit as st
    except ModuleNotFoundError:
        return None

    try:
        section = st.secrets["gmail_token"]
    except (KeyError, FileNotFoundError, AttributeError):
        return None

    return dict(section)


def _refresh_if_needed(creds):
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return creds


def get_gmail_service(
    credentials_path="credentials.json",
    token_path="token.json",
):
    # 1. Cloud / preferred: token provided via st.secrets (no browser needed).
    token_info = _token_info_from_secrets()
    if token_info:
        creds = _refresh_if_needed(
            Credentials.from_authorized_user_info(token_info, SCOPES)
        )
        if creds and creds.valid:
            return build("gmail", "v1", credentials=creds)

    # 2. Local development: reuse token.json on disk, refreshing if expired.
    if os.path.exists(token_path):
        creds = _refresh_if_needed(
            Credentials.from_authorized_user_file(token_path, SCOPES)
        )
        if creds and creds.valid:
            with open(token_path, "w", encoding="utf-8") as token_file:
                token_file.write(creds.to_json())
            return build("gmail", "v1", credentials=creds)

    # 3. First-time local authorization only. This opens a browser and CANNOT
    #    run on a headless server, so it is gated on credentials.json existing
    #    locally. On Streamlit Cloud that file is absent, so we fail loudly with
    #    guidance instead of hanging on run_local_server().
    if os.path.exists(credentials_path):
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())
        return build("gmail", "v1", credentials=creds)

    raise RuntimeError(
        "No Gmail credentials available. On Streamlit Cloud, add the contents "
        "of token.json to app secrets under a [gmail_token] section. Locally, "
        "place credentials.json in the project root and run once to authorize."
    )


def list_labels(service):
    result = service.users().labels().list(userId="me").execute()
    return [label["name"] for label in result.get("labels", [])]


def search_messages(service, query, max_results=5):
    result = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    return result.get("messages", [])


def get_message(service, message_id):
    return (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="metadata")
        .execute()
    )


def send_email(service, to, subject, body):
    message = build_email_message(to=to, subject=subject, body=body)
    return service.users().messages().send(userId="me", body=message).execute()
