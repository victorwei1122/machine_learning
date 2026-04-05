import urllib.request
import ssl
import os

def fetch_telco_churn():
    print("Downloading Telco Customer Churn dataset...")
    # Standard repository for this well-known dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '../data'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), '../data/customer_churn.csv')
    
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, context=context) as response, open(out_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"✅ Downloaded successfully and saved to {os.path.abspath(out_path)}")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    fetch_telco_churn()
