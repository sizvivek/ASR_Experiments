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



# Initialize a list to store the results for zero-shot prompting
zero_shot_results = []

# Select the first 20 file names for processing
test_files_to_process = test_files_wav[:20]
print(f"\n▶️ Starting zero-shot ASR process for {len(test_files_to_process)} files...")

# Create directory to store downloaded audio files (re-checked for robustness)
download_dir = "downloaded_audios"
if not os.path.exists(download_dir):
    os.makedirs(download_dir, exist_ok=True)
    print(f"✅ Directory '{download_dir}' created.")
elif not os.path.isdir(download_dir) or not os.access(download_dir, os.W_OK):
     print(f"❌ Error: Download directory '{download_dir}' does not exist or is not writable.")
     test_files_to_process = [] # Prevent processing if directory is not ready
else:
     print(f"✅ Download directory '{download_dir}' is accessible and writable.")


if not test_files_to_process:
    print("\n⚠️ No audio files selected for processing. Skipping transcription and evaluation.")
else:
    # Create a mapping from file name to ground truth from the loaded DataFrame
    # Reload ground truth if it was not available from previous execution
    if 'ground_truth_df' not in locals() or ground_truth_df.empty:
        csv_url = "https://raw.githubusercontent.com/KrishiVaani/KrishiVaani/main/LM/Known/krishivaani_known.csv"
        ground_truth_df = pd.read_csv(csv_url)
        print("Ground truth data reloaded.")

    ground_truth_map = ground_truth_df.set_index('File')['ground_truth'].to_dict()
    print(f"\nLoaded ground truth for {len(ground_truth_map)} files from the CSV.")

    for fname in test_files_to_process:
        print(f"\n🎤 Processing {fname} (Zero-Shot)...")

        # Get ground truth transcript from the DataFrame map
        gt_text = ground_truth_map.get(fname)

        if gt_text is None:
            print(f"⚠️ Ground truth not found in DataFrame for {fname}. Skipping transcription and evaluation for this file.")
            continue # Skip to the next file
        else:
             # Ensure ground truth is a string, handle potential non-string types
            gt_text = str(gt_text).strip()
            if not gt_text:
                 print(f"⚠️ Ground truth is empty for {fname}. Skipping transcription and evaluation for this file.")
                 continue # Skip if ground truth is empty
            print("✅ Ground truth found in DataFrame.")


        audio_url = audio_base + fname
        save_path = os.path.join(download_dir, fname)

        # Download audio if not exists
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"✅ File '{fname}' already exists locally. Skipping download.")
        else:
            try:
                print(f"⬇️ Downloading {fname} from {audio_url} to {save_path}...")
                with requests.get(audio_url, stream=True) as r:
                    r.raise_for_status()
                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"✅ Download of {fname} complete.")
            except requests.exceptions.RequestException as e:
                print(f"❌ Error downloading {fname}: {e}. Skipping transcription and evaluation for this file.")
                continue # Skip ASR if download fails

        # Verify file existence and size after download attempt or if it existed
        if not (os.path.exists(save_path) and os.path.getsize(save_path) > 0):
             print(f"❌ Verification failed: File '{fname}' was not saved correctly or is empty. Skipping ASR for this file.")
             continue # Skip ASR if file not saved


        # Run ASR (zero-shot)
        print(f"✨ Running ASR on {fname} (Zero-Shot)...")
        try:
            with open(save_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=f,
                    # No prompt parameter for zero-shot
                )
            pred_text = response.text.strip()
            print("✅ ASR transcription complete.")

            # Compute WER
            print("📊 Computing WER...")
            wer = jiwer.wer(gt_text, pred_text)
            print(f"✅ WER for {fname}: {wer:.4f}")

            # Store results
            zero_shot_results.append({
                "file": fname,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "WER": wer,
                "prompt_type": "zero-shot"
            })
        except Exception as e:
            print(f"❌ An unexpected error occurred during ASR or WER calculation for {fname}: {e}. Skipping.")
            continue

    print("\n🏁 Finished processing files for zero-shot prompting.")

# ===============================
# 🔹 Save and Show Zero-Shot Results
# ===============================
print("\n📝 Saving and showing zero-shot results...")
if zero_shot_results:
    df_zero_shot = pd.DataFrame(zero_shot_results).sort_values("WER")
    df_zero_shot.to_csv("asr_zero_shot_results.csv", index=False)

    print("\n✅ Zero-shot transcription results saved to asr_zero_shot_results.csv\n")
    display(df_zero_shot)
else:
    print("\n⚠️ No zero-shot transcription results were generated.")
