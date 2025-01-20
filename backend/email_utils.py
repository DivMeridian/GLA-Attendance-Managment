from azure.communication.email import EmailClient
from dotenv import load_dotenv
import os 

load_dotenv()

def send_attendance_email(to_email: str, subject: str, plain_text_body: str, html_body: str = None):
    """
    Sends an email using Azure Communication Services EmailClient.

    :param to_email: Recipient's email address.
    :param subject: Email subject.
    :param plain_text_body: Plain text content of the email.
    :param html_body: HTML content of the email (optional).
    """
    try:
        # Connection string for Azure Communication Services
        connection_string = "endpoint=https://ai-mailing.unitedstates.communication.azure.com/;accesskey=AVEVdvPHOzVUMAnXrGlm63cgvVPyWFuTQZLPxCona26mdeiqKif5JQQJ99AKACULyCphD9BDAAAAAZCSycfM"
        client = EmailClient.from_connection_string(connection_string)

        # Email message details
        message = {
            "senderAddress": "DoNotReply@onmeridian.com",
            "recipients": {
                "to": [{"address": to_email}]
            },
            "content": {
                "subject": subject,
                "plainText": plain_text_body,
                "html": html_body or f"<html><body><p>{plain_text_body}</p></body></html>"
            },
        }

        # Send email
        poller = client.begin_send(message)
        result = poller.result()
        return "Email sent successfully"

    except Exception as ex:
        print(f"Failed to send email to {to_email}: {ex}")