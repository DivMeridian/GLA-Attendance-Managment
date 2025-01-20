import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os 

load_dotenv()

def send_attendance_email(to_email: str, subject: str, body: str):
    """
    Sends an email with the specified subject and body to the given recipient.
    
    :param to_email: Recipient's email address.
    :param subject: Subject of the email.
    :param body: Body content of the email.
    """
    try:
        # SMTP server configuration
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")  # Use environment variables for better security

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Connect to the SMTP server and send the email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
        
        print(f"Email sent successfully to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}. Error: {str(e)}")
