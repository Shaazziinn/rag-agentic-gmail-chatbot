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


def get_gmail_service(
    credentials_path="credentials.json",
    token_path="token.json",
):
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Missing {credentials_path}. Download it from Google Cloud "
            "OAuth Client credentials and place it in the project root."
        )

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path,
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


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
