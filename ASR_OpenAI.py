# Make a GET request to the GitHub API URL for the audio directory
headers = {
    **({"Authorization": f"token {github_token}"} if github_token else {}),
    "Accept": "application/vnd.github.v3+json" # Specify API version
}

test_files_wav = []
try:
    print(f"🔍 Requesting file list from: {github_api_url}...")
    response = requests.get(github_api_url, headers=headers)

    # Check the response status code
    response.raise_for_status() # Raise an exception for bad status codes (e.g., 404, 401)

    # Parse the JSON response
    contents = response.json()

    # Iterate through the items and filter for .wav files
    for item in contents:
        if item.get("type") == "file" and item.get("name").endswith(".wav"):
            test_files_wav.append(item["name"])

    # Print the number of audio files found
    print(f"✅ Found {len(test_files_wav)} audio files using GitHub API.")

except requests.exceptions.RequestException as e:
    # Print an error message if the API call failed
    print(f"❌ Error fetching file list from GitHub API: {e}")
    print("⚠️ Please ensure your GitHub token is correct and has access to the repository, or if the repo is public, that the API allows unauthenticated access to contents.")
    test_files_wav = [] # Clear the list if API call fails
except Exception as e:
    print(f"❌ An unexpected error occurred while processing GitHub API response: {e}")
    test_files_wav = [] # Clear the list if processing fails
