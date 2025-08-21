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


# Define a pool of audio-transcript pairs to be used as examples
# These examples should be representative of the audio data and their corresponding accurate transcripts.
# We need at least 3 examples to support up to three-shot prompting.
few_shot_examples_full = [
    {"audio_text": "और सामने जो मुझे दिख रही हैं", "transcript": "सामने भी मुझे दिख रही हैं."},
    {"audio_text": "लाइन में रो में बड़े बड़े पड़े हुए हैं", "transcript": "लाइन में, रो میں बड़े बड़े पेड़ पड़े हुए हैं."},
    {"audio_text": "सर्दियों में खूब अच्छे से पीला फूल देगी", "transcript": "सर्दियों میں خوب اچھے سے پیلا پھول دے گی"},
    # Add more examples if you want to experiment with more shots later, though 3 are sufficient for this task.
    {"audio_text": "किसान से थोड़ी سی بھی ہمدردی ہے", "transcript": "کسان سے تھوڑی سی بھی ہمدردی ہے"},
]

print(f"Defined a pool of {len(few_shot_examples_full)} potential few-shot examples.")
# Optionally, print the examples to review them
# for i, example in enumerate(few_shot_examples_full):
#     print(f"Example {i+1}: Audio: '{example['audio_text']}', Transcript: '{example['transcript']}'")
