import os
from email.message import EmailMessage
import ssl
import smtplib


def email_sender(receiver):
    sender = "EMAIL"
    email_pass = "PASSWORD"

    # subject = "File of passwords"
    subject = "Weekly Finance Report sign up"
    

    em = EmailMessage()
    em["From"] = sender
    em["To"] = receiver
    em["Subject"] = subject

    #HTML body 
    html_content = f"""
    <html>
    <body>
        <h2>Weekly Finance Report</h2>

        <p>Thank you for signing up! \n We hope you make use of your weekly financial reports.</p>
    </body>
</html>
"""
    em.add_alternative(html_content, subtype="html")

    data = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=data) as smtp:
        smtp.login(sender, email_pass)
        smtp.sendmail(sender, receiver, em.as_string())
