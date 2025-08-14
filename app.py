import os
import requests
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

print("Loading HF_TOKEN from environment...")
token = os.environ.get("HF_TOKEN")
if token:
    print(f"Token found: {token[:10]}...")  # Show first 10 chars for verification
else:
    print("ERROR: HF_TOKEN not found in environment!")
    exit(1)

print("Initializing InferenceClient...")
client = InferenceClient(
    provider="fal-ai",
    api_key=token,
)

# Create output directory if it doesn't exist
output_dir = "generated_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

print("Generating video with prompt: 'A young man walking on the street'")
print("This may take a while...")

try:
    video = client.text_to_video(
        "a sex scene with 2 naked girls licking each others ",
        model="Wan-AI/Wan2.2-T2V-A14B",
    )
    
    print("Video generation complete!")
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{timestamp}.mp4"
    filepath = os.path.join(output_dir, filename)
    
    # Check if video is a URL or binary data
    if isinstance(video, str) and video.startswith(('http://', 'https://')):
        # If it's a URL, download the video
        print(f"Downloading video from URL: {video}")
        response = requests.get(video)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Video saved to: {filepath}")
        
    elif isinstance(video, bytes):
        # If it's binary data, save directly
        with open(filepath, 'wb') as f:
            f.write(video)
        print(f"Video saved to: {filepath}")
        
    elif hasattr(video, 'read'):
        # If it's a file-like object
        with open(filepath, 'wb') as f:
            f.write(video.read())
        print(f"Video saved to: {filepath}")
        
    else:
        # Print the type and content for debugging
        print(f"Unexpected video format: {type(video)}")
        print(f"Video content: {video}")
        
        # Try to extract URL if it's in a different format
        if hasattr(video, '__dict__'):
            print(f"Video attributes: {video.__dict__}")
            
        # Save metadata to a text file for inspection
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.txt")
        with open(metadata_file, 'w') as f:
            f.write(f"Video type: {type(video)}\n")
            f.write(f"Video content: {str(video)}\n")
            if hasattr(video, '__dict__'):
                f.write(f"Video attributes: {video.__dict__}\n")
        print(f"Metadata saved to: {metadata_file}")
        
    print("\nVideo generation and saving complete!")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
