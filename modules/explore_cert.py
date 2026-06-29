import pandas as pd

path = r"C:\Users\essam\Downloads\r4.2\\"

logon  = pd.read_csv(path + "logon.csv",  nrows=5)
file   = pd.read_csv(path + "file.csv",   nrows=5)
device = pd.read_csv(path + "device.csv", nrows=5)
http   = pd.read_csv(path + "http.csv",   nrows=5)
email  = pd.read_csv(path + "email.csv",  nrows=5)

print("=== LOGON ===");   print(logon.to_string())
print("\n=== FILE ===");    print(file.to_string())
print("\n=== DEVICE ===");  print(device.to_string())
print("\n=== HTTP ===");    print(http.to_string())
print("\n=== EMAIL ===");   print(email.to_string())