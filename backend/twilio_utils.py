from twilio.rest import Client

class TwilioClient:
    def __init__(self, account_sid, auth_token, twilio_number):
        """
        Initialize the Twilio client.
        :param account_sid: Twilio Account SID.
        :param auth_token: Twilio Auth Token.
        :param twilio_number: Twilio phone number.
        """
        self.client = Client(account_sid, auth_token)
        self.twilio_number = twilio_number

    def send_sms(self, to_number, message_body):
        """
        Send an SMS using Twilio.
        :param to_number: Recipient's phone number.
        :param message_body: Message content.
        """
        try:
            message = self.client.messages.create(
                body=message_body,
                from_=self.twilio_number,
                to=to_number
            )
            print(f"Message sent successfully to {to_number} with SID: {message.sid}")
        except Exception as e:
            print(f"Failed to send message: {str(e)}")
