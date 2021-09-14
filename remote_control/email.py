from email.message import EmailMessage
from smtplib import SMTP
from traceback import print_exc


def send_email(config, email_address, subject, body, attachments=None):
    if email_address is None or email_address == 'None':
        return
    try:
        smtp_config = config.get('smtp', {})
        username = smtp_config.get('username')
        password = smtp_config.get('password')

        smtp = SMTP(smtp_config.get('host', 'localhost'), timeout=20)
        try:
            smtp.starttls()
        except:
            pass
        if username is not None or password is not None:
            smtp.login(username, password)

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = smtp_config.get('from_address', email_address)
        msg['To'] = email_address
        msg.set_content(body)

        for filename, content in (attachments or {}).items():
            msg.add_attachment(content, filename=filename)

        smtp.send_message(msg)
        smtp.quit()
    except Exception:
        print('Email sending failed:')
        print_exc()
        # Suppress exception so that other processes may succeed
