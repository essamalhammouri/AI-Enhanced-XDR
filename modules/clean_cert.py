import pandas as pd
import os

# <- Change this to your r4.2 folder path
INPUT_PATH  = r"C:\Users\essam\Downloads\r4.2\\"
OUTPUT_PATH = r"C:\Users\essam\xdr_project\data\cert_clean\\"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- LOGON ---
logon = pd.read_csv(INPUT_PATH + "logon.csv")
logon = logon[["id","date","user","pc","activity"]]
logon["date"] = pd.to_datetime(logon["date"])
logon.to_csv(OUTPUT_PATH + "logon_clean.csv", index=False)
print(f"logon done: {len(logon)} rows")

# --- FILE ---
file = pd.read_csv(INPUT_PATH + "file.csv")
file = file[["id","date","user","pc","filename"]]
file["date"] = pd.to_datetime(file["date"])
file.to_csv(OUTPUT_PATH + "file_clean.csv", index=False)
print(f"file done: {len(file)} rows")

# --- DEVICE ---
device = pd.read_csv(INPUT_PATH + "device.csv")
device = device[["id","date","user","pc","activity"]]
device["date"] = pd.to_datetime(device["date"])
device.to_csv(OUTPUT_PATH + "device_clean.csv", index=False)
print(f"device done: {len(device)} rows")

# --- HTTP ---
http = pd.read_csv(INPUT_PATH + "http.csv")
http = http[["id","date","user","pc","url"]]
http["date"] = pd.to_datetime(http["date"])
http.to_csv(OUTPUT_PATH + "http_clean.csv", index=False)
print(f"http done: {len(http)} rows")

# --- EMAIL ---
email = pd.read_csv(INPUT_PATH + "email.csv")
email = email[["id","date","user","pc","to","from","size","attachments"]]
email["date"] = pd.to_datetime(email["date"])
email.to_csv(OUTPUT_PATH + "email_clean.csv", index=False)
print(f"email done: {len(email)} rows")

print("\n[OK] All files cleaned and saved to:", OUTPUT_PATH)