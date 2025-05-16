
import json
import requests


def retrieve_pages(keywords):
    # Remove duplicates
    keywords = list(set(keywords))

    # Define the API endpoint and headers
    api_url = 'http://localhost:8000/dbpedia/search-data'
    headers = {
        'Content-Type': 'application/json'
    }

    # Define the payload for the API request
    payload = {
        'keywords': keywords,
        'limit': 3
    }

    # Make the API request
    response = requests.post(api_url, headers=headers, json=payload)

    # Check the response status code
    if response.status_code == 200:
        # Parse the response JSON
        pages = response.json().get('pages', [])
        return pages
    else:
        print('Failed to retrieve pages:', response.status_code, response.text)
        return []


def extract_pages_mapping(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    results = []

    # Extract the keywords from the JSON data
    for item in data:
        # Retrieve pages based on the keywords
        keywords = item['exact_keywords']
        entry_id = item['entry_id']
        pages = retrieve_pages(keywords)

        # Print the retrieved pages
        print(pages)
        results.append({'entry_id': entry_id, 'pages': pages})
    return results


if __name__ == '__main__':
    # Define the path to the JSON file
    json_file_path = 'data/streaming/history_learning_data_sample.json'
    extract_pages_mapping(json_file_path)
