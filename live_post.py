"""Send a POST request to the *deployed* API and print the result.

Usage (PowerShell / bash)::

    python live_post.py https://YOUR-APP.onrender.com

or set the ``API_URL`` environment variable instead of passing an argument.
"""
import os
import sys

import requests

SAMPLE = {
    "age": 45,
    "workclass": "Private",
    "fnlgt": 160000,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}


def main(argv):
    base_url = argv[1] if len(argv) > 1 else os.environ.get("API_URL")
    if not base_url:
        sys.exit("Provide the live URL as an argument or set API_URL.")
    # Free hosting tiers may need up to a minute to wake from sleep.
    response = requests.post(
        base_url.rstrip("/") + "/predict", json=SAMPLE, timeout=120
    )
    print("Status code:", response.status_code)
    print("Model inference result:", response.json())


if __name__ == "__main__":
    main(sys.argv)
