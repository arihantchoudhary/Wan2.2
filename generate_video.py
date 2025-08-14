import os
import sys
import requests
import argparse
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

def generate_and_save_video(prompt, output_dir="generated_videos", model="Wan-AI/Wan2.2-T2V-A14B"):
    """
    Generate a video from a text prompt and save it to disk.
    
    Args:
        prompt (str): The text prompt for video generation
        output_dir (str): Directory to save the generated video
        model (str): The model to use for generation
    
    Returns:
        str: Path to the saved video file
    """
    load_dotenv()
    
    # Load token
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in environment. Please set it in .env file.")
    
    print(f"Initializing InferenceClient with model: {model}")
    client = InferenceClient(
        provider="fal-ai",
        api_key=token,
    )
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"Generating video with prompt: '{prompt}'")
    print("This may take a while...")
    
    try:
        # Generate video
        video = client.text_to_video(
            prompt,
            model=model,
        )
        
        print("Video generation complete!")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a safe filename from the prompt (first 30 chars, alphanumeric only)
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c.isspace()).replace(" ", "_")
        filename = f"{safe_prompt}_{timestamp}.mp4"
        filepath = os.path.join(output_dir, filename)
        
        # Save the video based on its format
        if isinstance(video, str) and video.startswith(('http://', 'https://')):
            # If it's a URL, download the video
            print(f"Downloading video from URL...")
            response = requests.get(video)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
        elif isinstance(video, bytes):
            # If it's binary data, save directly
            with open(filepath, 'wb') as f:
                f.write(video)
                
        elif hasattr(video, 'read'):
            # If it's a file-like object
            with open(filepath, 'wb') as f:
                f.write(video.read())
        else:
            # Handle unexpected format
            print(f"Warning: Unexpected video format: {type(video)}")
            # Save metadata for debugging
            metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.txt")
            with open(metadata_file, 'w') as f:
                f.write(f"Video type: {type(video)}\n")
                f.write(f"Video content: {str(video)}\n")
            print(f"Metadata saved to: {metadata_file}")
            return None
        
        print(f"✅ Video saved successfully to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate videos from text prompts using Wan-AI model')
    parser.add_argument('prompt', nargs='?', default=None, help='Text prompt for video generation')
    parser.add_argument('--output-dir', '-o', default='generated_videos', help='Output directory for videos (default: generated_videos)')
    parser.add_argument('--model', '-m', default='Wan-AI/Wan2.2-T2V-A14B', help='Model to use for generation')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode - enter multiple prompts')
    
    args = parser.parse_args()
    
    if args.interactive:
        print("🎬 Interactive Video Generation Mode")
        print("Enter prompts to generate videos (type 'quit' to exit)")
        print("-" * 50)
        
        while True:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not prompt:
                print("Please enter a valid prompt.")
                continue
                
            filepath = generate_and_save_video(prompt, args.output_dir, args.model)
            if filepath:
                print(f"\n📹 Video ready: {filepath}")
            print("-" * 50)
    else:
        # Single prompt mode
        if not args.prompt:
            print("Error: Please provide a prompt or use --interactive mode")
            parser.print_help()
            sys.exit(1)
            
        filepath = generate_and_save_video(args.prompt, args.output_dir, args.model)
        if filepath:
            print(f"\n📹 Video ready: {filepath}")
            print(f"You can play it with: open {filepath}")

if __name__ == "__main__":
    main()
