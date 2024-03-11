import requests
import os

class APIClient:
    def __init__(self):
        self.session = requests.Session()

    def perform_get_request(self, url, headers=None):
        try:
            response = self.session.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response
        except requests.RequestException as e:
            print("Error during GET request:", e)
            return None

    def download_response_content(self, response, save_path):
        try:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete. Saved to:", save_path)
        except Exception as e:
            print("Error during content download:", e)
