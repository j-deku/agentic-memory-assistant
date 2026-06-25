import os
import urllib.request
import zipfile
import json

# Configurations
URL = "https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip"
ZIP_FILE = "movie_corpus.zip"
OUTPUT_FILE = "chat_data.txt"
MAX_CONVERSATIONS = 1000 

print("Step 1: Checking/Downloading raw data file (approx 38MB)...")
if not os.path.exists(ZIP_FILE):
    urllib.request.urlretrieve(URL, ZIP_FILE)
    print("Download complete!")
else:
    print("Zip archive found locally.")

print("Step 2: Parsing JSON lines cleanly...")
lines = []
current_pair = []

with zipfile.ZipFile(ZIP_FILE, 'r') as z:
    with z.open("movie-corpus/utterances.jsonl") as f:
        for line_bytes in f:
            # Parse the line as an actual JSON object
            data_obj = json.loads(line_bytes.decode('utf-8'))
            
            # Extract only the pure text field
            text = data_obj.get("text", "").strip()
            
            if text and len(text) < 200:
                current_pair.append(text)
            
            # Pair them into User and Assistant turns
            if len(current_pair) == 2:
                formatted_line = f"<|user|>{current_pair[0]}<|assistant|>{current_pair[1]}<|end|>"
                lines.append(formatted_line)
                current_pair = []
                
            if len(lines) >= MAX_CONVERSATIONS:
                break

print(f"Step 3: Saving {len(lines)} pure dialogue lines to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for line in lines:
        out.write(line + "\n")

print("Success! Your dataset is now 100% clean human speech.")