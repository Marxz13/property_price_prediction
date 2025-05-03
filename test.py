import requests
import json

url = "http://localhost:5000/api/valuation/predict"
payload = {
    "place": {
        "formattedAddress": "Jalan SS 2/3, SS 2, 47300 Petaling Jaya, Selangor, Malaysia",
        "location": {
            "lat": 3.1107989,
            "lng": 101.6127874
        }
    },
    "propertyType$1": "1",
    "propertyType$2": "0",
    "propertyDetails$1": {
        "buildUpAreaInSqft": 861,
        "landAreaInSqft": 861
    },
    "propertyDetails$2": {
        "bedroomCount": 2,
        "bathroomCount": 2
    },
    "furnishing_encoded": 2,
    "tenure_encoded": 0,
    "propertyTitle_encoded": 2,
    "propertyClass_encoded": 0,
    "others": {
        "structure": {
            "structure": ["4"],
            "structureExtension": "0"
        }
    },
    "flooringTypes": ["1"],
    "livingRoom": "3",
    "kitchen": 2,
    "bathroom": 2,
    "homeExterior": 2,
    "view": 0,
    "security": "1",
}

headers = {
    'Content-Type': 'application/json'
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.status_code)
print(response.json())