import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, 'datasets', 'flagged_patients_report.csv')
load_dotenv()

def send_email(email, name, risk,last_consultation):
    # Email account credentials
    email_password = os.getenv("EMAIL_PASS")
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = email #gracemboothe@gmail.com

    # 1. Calculate if last appointment was > 9 months ago
    last_date = datetime.strptime(last_consultation, '%Y-%m-%d')
    days_since = (datetime.now() - last_date).days
    is_overdue = days_since > 270 # 9 months ~ 270 days

    nhs_booking_url = "https://www.nhs.uk/nhs-services/gps/gp-appointments-and-bookings/"
    overdue_msg = ""
    if is_overdue:
        overdue_msg = f"\n⚠️ NOTICE: Our records show your last consultation was over 9 months ago ({last_consultation}). It is vital for patients in your risk category to have regular check-ups."

    # Create the email
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "IMPORTANT: Health data flagged"

    # Email body
    body = f"""Dear {name},
        This is Grace from Viva.
        Based on our recent AI health analysis, your predicted risk level is {risk} times higher than the average for breast cancer.
        {overdue_msg}

        We strongly recommend that you book an appointment with your GP as soon as possible. You can book online via the official NHS portal here:
        {nhs_booking_url}

        Best regards,
        Grace
        Viva Clinical Team
        """

    message.attach(MIMEText(body, "plain"))

    # Connect to the SMTP server and send the email
    try:
        # For Gmail use smtp.gmail.com with port 587
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Secure the connection
        server.login(sender_email, email_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    except Exception as e:
        print(f"❌ Failed to send to {name}: {e}")

def run_automation():
    #Reads the flagged report and triggers emails for all at-risk patients
    if not os.path.exists(REPORT_PATH):
        print(f"⚠️ Error: {REPORT_PATH} not found. Run the ML prediction script first!")
        return

    # Load the flagged patients
    df = pd.read_csv(REPORT_PATH)
    
    if df.empty:
        print("Empty report. No emails to send today.")
        return

    # Loop through the rows and send personalized emails
    for index, row in df.iterrows():
        # Using the exact column names from your generated CSV
        patient_name = row['Patient_Name']
        patient_email = row['patient_email']
        risk_score = round(row['predicted_relative_risk'], 2)
        last_consultation = row['Last_Consultation_Date']
        
        send_email(patient_email, patient_name, risk_score,last_consultation)
    
    print(f"\n✅ Automation complete. {len(df)} patients notified.")

if __name__ == "__main__":
    run_automation()