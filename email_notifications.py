import smtplib
import os

# Email variables definition
# -----------------------------------------------------------------------------------------------
sender = 'svirahonda@gmail.com'
receiver = ['svirahonda@gmail.com'] #replace this by the owner's email address
smtp_provider = 'smtp.gmail.com'
smtp_port = 587
smtp_account = os.getenv("email_address")
smtp_password = os.getenv("gmail_password")
# -----------------------------------------------------------------------------------------------

def training_result(result,model_acc):

    if result == 'ok':
        message = 'A training job has ended recently. The model reached '+str(model_acc)+' during evaluation, therefore has been saved to GCS. Check the GCP logs for more information.'
    if result == 'failed':
        message = 'A recent training job has failed. None of the models reached an acceptable accuracy, therefore the tranining execution had to be forcefully ended. Check the GCP logs for more information.'
    message = 'Subject: {}\n\n{}'.format('An automatic training job has ended recently', message)

    try:
        server = smtplib.SMTP(smtp_provider,smtp_port)
        server.starttls()
        server.login(smtp_account,smtp_password)
        server.sendmail(sender, receiver, message)         
        print('Email sent successfully',flush=True)
        return
    except Exception as e:
        print('Something went wrong. Unable to send email.',flush=True)
        print('Exception: ',e)
        return

def exception(e_message):

    try:
        message = 'Subject: {}\n\n{}'.format('An automatic training job has failed recently', e_message)
        server = smtplib.SMTP(smtp_provider,smtp_port)
        server.starttls()
        server.login(smtp_account,smtp_password)
        server.sendmail(sender, receiver, message)         
        print('Email sent successfully',flush=True)
        return
    except Exception as e:
        print('Something went wrong. Unable to send email.',flush=True)
        print('Exception: ',e)
        return
