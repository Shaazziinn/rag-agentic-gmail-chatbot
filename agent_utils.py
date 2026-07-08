import json
import re

from gmail_utils import get_message, search_messages


def extract_json_object(content):
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    object_match = re.search(r"\{.*\}", content, re.DOTALL)
    if object_match:
        return json.loads(object_match.group(0))

    raise ValueError("No JSON object found in LLM response.")


def validate_email_draft(draft):
    errors = []

    if not draft.get("to", "").strip():
        errors.append("Missing recipient.")
    if not draft.get("subject", "").strip():
        errors.append("Missing subject.")
    if not draft.get("body", "").strip():
        errors.append("Missing body.")

    return errors


def _header_value(message, header_name):
    headers = message.get("payload", {}).get("headers", [])
    for header in headers:
        if header.get("name", "").lower() == header_name.lower():
            return header.get("value", "")
    return ""


def format_message_context(message):
    lines = [
        f"Message ID: {message.get('id', '')}",
        f"Thread ID: {message.get('threadId', '')}",
        f"From: {_header_value(message, 'From')}",
        f"To: {_header_value(message, 'To')}",
        f"Subject: {_header_value(message, 'Subject')}",
        f"Date: {_header_value(message, 'Date')}",
        f"Snippet: {message.get('snippet', '')}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


def _llm_text(llm, prompt):
    response = llm.invoke(prompt)
    return response.content


def classify_agent_intent(user_request, llm):
    prompt = f"""
You are routing a message inside an agentic Gmail chatbot.

Classify the user's request into one of these intents:

1. chat
Use this for normal conversation, explanations, greetings, or questions about the app.

2. gmail_search
Use this when the user wants to search, read, summarize, or understand emails.

3. email_draft
Use this when the user wants to draft, reply, compose, or send an email.

Return only JSON:
{{"intent": "chat"}}
or
{{"intent": "gmail_search"}}
or
{{"intent": "email_draft"}}

User request:
{user_request}
"""
    try:
        parsed = extract_json_object(_llm_text(llm, prompt))
        intent = parsed.get("intent", "chat").strip()
    except (ValueError, json.JSONDecodeError):
        return "chat"

    if intent not in ["chat", "gmail_search", "email_draft"]:
        return "chat"

    return intent


def create_chat_response(user_request, llm):
    prompt = f"""
You are an agentic AI chatbot.

You can chat normally, explain AI concepts, and help the user understand
RAG, agents, tools, Gmail API, and human approval.

Answer naturally and clearly.

User:
{user_request}
"""
    return _llm_text(llm, prompt)


def create_gmail_query(user_request, llm):
    prompt = f"""
You convert a user's email task into a Gmail search query.

Return only JSON with this shape:
{{"query": "gmail search query"}}

Rules:
- Prefer concise Gmail search syntax.
- If the user asks for recent email and gives no other filter, use "in:inbox newer_than:30d".
- If the user mentions a sender, use from: when possible.
- Do not include explanations.

User request:
{user_request}
"""
    try:
        parsed = extract_json_object(_llm_text(llm, prompt))
        query = parsed.get("query", "").strip()
    except (ValueError, json.JSONDecodeError):
        query = ""

    return query or "in:inbox newer_than:30d"


def create_gmail_answer(user_request, gmail_context, llm):
    prompt = f"""
You are an agentic Gmail chatbot.

Answer the user's question using the Gmail context below.
If no matching emails were found, say that clearly.
Do not draft or send an email unless the user specifically asked to draft, reply, compose, or send.

User request:
{user_request}

Gmail context:
{gmail_context}
"""
    return _llm_text(llm, prompt)


def create_email_draft(user_request, gmail_context, llm):
    prompt = f"""
You are an email assistant inside an agentic chatbot.

Create a draft email proposal from the user's request and Gmail context.
Return only JSON with this shape:
{{
  "to": "recipient email address",
  "subject": "email subject",
  "body": "email body",
  "rationale": "short reason for this draft"
}}

Rules:
- Do not invent facts that are not in the user request or Gmail context.
- If replying to a thread, address the sender from the relevant email.
- Keep the body professional and concise.
- Do not say the email has been sent.

User request:
{user_request}

Gmail context:
{gmail_context}
"""
    draft = extract_json_object(_llm_text(llm, prompt))
    errors = validate_email_draft(draft)
    if errors:
        raise ValueError("Invalid email draft: " + " ".join(errors))
    return draft


def revise_email_draft(existing_draft, edit_request, llm):
    prompt = f"""
You revise a pending email draft inside an agentic Gmail chatbot.

Apply the user's edit request to the existing draft.
Return only JSON with this exact shape:
{{
  "to": "recipient email address",
  "subject": "email subject",
  "body": "email body",
  "rationale": "short reason for this revision"
}}

Rules:
- Preserve the existing recipient unless the user explicitly changes it.
- Preserve the existing subject unless the user explicitly changes it.
- Preserve the existing body except for the requested edit.
- Do not send the email.
- Do not remove required fields.

Existing draft:
{json.dumps(existing_draft, indent=2)}

User edit request:
{edit_request}
"""
    revised_draft = extract_json_object(_llm_text(llm, prompt))

    revised_draft.setdefault("to", existing_draft.get("to", ""))
    revised_draft.setdefault("subject", existing_draft.get("subject", ""))
    revised_draft.setdefault("body", existing_draft.get("body", ""))
    revised_draft.setdefault("rationale", "Updated the pending draft.")

    errors = validate_email_draft(revised_draft)
    if errors:
        raise ValueError("Invalid email draft revision: " + " ".join(errors))

    return revised_draft


def run_agentic_chat(user_request, llm, service, max_results=3):
    intent = classify_agent_intent(user_request, llm)

    if intent == "chat":
        return {
            "type": "chat",
            "content": create_chat_response(user_request, llm),
        }

    query = create_gmail_query(user_request, llm)
    message_refs = search_messages(service, query, max_results=max_results)

    messages = [get_message(service, message["id"]) for message in message_refs]
    gmail_context = "\n\n---\n\n".join(
        format_message_context(message) for message in messages
    )

    if not gmail_context:
        gmail_context = "No matching Gmail messages were found."

    if intent == "gmail_search":
        return {
            "type": "chat",
            "content": create_gmail_answer(user_request, gmail_context, llm),
            "query": query,
            "context": gmail_context,
        }

    draft = create_email_draft(user_request, gmail_context, llm)

    return {
        "type": "email_draft",
        "query": query,
        "context": gmail_context,
        "draft": draft,
    }


def run_gmail_agent(user_request, llm, service, max_results=3):
    query = create_gmail_query(user_request, llm)
    message_refs = search_messages(service, query, max_results=max_results)

    messages = [get_message(service, message["id"]) for message in message_refs]
    gmail_context = "\n\n---\n\n".join(
        format_message_context(message) for message in messages
    )

    if not gmail_context:
        gmail_context = "No matching Gmail messages were found."

    draft = create_email_draft(user_request, gmail_context, llm)

    return {
        "type": "email_draft",
        "query": query,
        "context": gmail_context,
        "draft": draft,
    }
