import argparse
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve API credentials
api_key = os.getenv("OAI_KEY")
api_endpoint = os.getenv("OAI_ENDPOINT")

def test_openai_client():
    """Test the AzureOpenAI client connection and response format."""
    
    if not api_key or not api_endpoint:
        print("[ERROR] API Key or Endpoint is missing. Check your .env file or environment variables.")
        return False

    try:
        client = AzureOpenAI(
            api_version="2024-05-13",
            azure_endpoint=api_endpoint,
            api_key=api_key,
        )

        print("[INFO] Client initialized successfully.")

        # Basic test query to check connection
        test_prompt = "What is 2 + 2?"
        response = client.chat.completions.create(
            model="gpt-4o",  # Adjust based on available models in your Azure account
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0,
            max_tokens=10
        )

        if response and response.choices:
            print("[SUCCESS] API connection successful. Response received:")
            print(json.dumps(response.choices[0].message.content, indent=2))
            return True
        else:
            print("[ERROR] API connection test failed. No valid response.")
            return False

    except Exception as e:
        print(f"[ERROR] Exception occurred while testing API: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Run API connection tests')

    args = parser.parse_args()

    if args.test:
        print("[INFO] Running OpenAI API connection tests...")
        success = test_openai_client()
        if not success:
            print("[WARNING] API test failed. Please check your setup.")
    else:
        print("[INFO] Running the main application...")
        # Call your existing search and evaluate functions here
