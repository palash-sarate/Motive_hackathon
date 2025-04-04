import boto3
import json
import tempfile
import os

aws_access_key_id = ''
aws_secret_access_key = ''

def read_s3_file(bucket_name, key_name, aws_access_key_id, aws_secret_access_key):
    """
    Reads a file from S3 and returns its content.

    Args:
        bucket_name (str): The S3 bucket name
        key_name (str): The S3 object key
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key

    Returns:
        str: The content of the file
    """
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Download file from S3
            s3_client.download_file(bucket_name, key_name, temp_path)

            # Read the file content
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return content

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print(f"Error reading file from S3: {str(e)}")
        raise

# def process_web_content(content):
#     """
#     Processes the web content and creates a dictionary with page keys and data tags.

#     Args:
#         content (str): The content of the web_content.txt file

#     Returns:
#         dict: Dictionary with page keys and their corresponding data
#     """
#     pages = {}
#     current_page = None
#     current_data = []

#     # Split content into lines
#     lines = content.split('\n')

#     for line in lines:
#         line = line.strip()

#         # Skip empty lines
#         if not line:
#             continue

#         # Check if line starts with a page marker (e.g., "PAGE:" or similar)
#         if line.startswith('PAGE:') or line.startswith('URL:'):
#             # If we have a current page, save its data
#             if current_page:
#                 pages[current_page] = {
#                     'data': '\n'.join(current_data),
#                     'tags': extract_tags(current_data)
#                 }

#             # Start new page
#             current_page = line
#             current_data = []
#         # elif line.startswith("TITLE:"):
#         #     # If previous entry exists, add it to entries
#         #     if entry:
#         #         entries.append(entry)
#         #     entry = {"title": line.replace("TITLE:", "").strip()}
#         else:
#             # Add line to current page's data
#             current_data.append(line)

#     # Add the last page if exists
#     if current_page and current_data:
#         pages[current_page] = {
#             'data': '\n'.join(current_data),
#             'tags': extract_tags(current_data)
#         }

#     return pages
def process_web_content(content):
    """
    Processes the web content and creates a structured dictionary per entry.
    """

    entries = []
    entry = {}

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("-----"):
            continue

        if line.startswith("TITLE:"):
            if entry:
                entries.append(entry)
            entry = {"title": line.replace("TITLE:", "").strip()}
            capture_field = None

        elif line.startswith("URL:"):
            entry["url"] = line.replace("URL:", "").strip()
            capture_field = None

        # elif line.startswith("METADATA:"):
        #     entry["metadata"] = []
        #     capture_field = None
        # elif line.startswith("categories:"):
        #     entry["categories"] = []
        #     capture_field = None
        # elif line.startswith("tags:"):
        #     entry["tags"] = []
        #     capture_field = None

        elif line.startswith("EXCERPT:"):
            entry["excerpt"] = ""
            capture_field = "excerpt"

        elif line.startswith("CONTENT:"):
            entry["content"] = ""
            capture_field = "content"

        else:
            if capture_field == "excerpt":
                entry["excerpt"] += (line + "\n")
            elif capture_field == "content":
                entry["content"] += (line + "\n")

    # Add the last entry
    if entry:
        entries.append(entry)

    return entries

def extract_tags(data_lines):
    """
    Extracts tags from the data lines.

    Args:
        data_lines (list): List of data lines for a page

    Returns:
        list: List of extracted tags
    """
    tags = []
    for line in data_lines:
        # Look for tag markers (e.g., "TAG:", "CATEGORY:", etc.)
        if 'TAG:' in line or 'CATEGORY:' in line:
            tag = line.split(':', 1)[1].strip()
            if tag:
                tags.append(tag)
    return tags

# function to download provided web_content.txt from s3
def download_web_content():
    bucket_name = 'motiverse-2025-data'
    key_name = 'web_content.txt'

    try:
        # Read the file from S3
        print(f"Reading {key_name} from {bucket_name}...")
        content = read_s3_file(bucket_name, key_name, aws_access_key_id, aws_secret_access_key)
        # save the content to a file
        with open(key_name, 'w', encoding='utf-8') as f:
            f.write(content)
        # Process the content
        print("Processing content...")
        pages = process_web_content(content)
        # save the processed pages to a file
        with open('processed_pages.json', 'w', encoding='utf-8') as f:
            json.dump(pages, f, ensure_ascii=False, indent=4)
        print(f"Processed {len(pages)} pages.")

        # Print sample of the processed data
        print("\nSample of processed data:")
        for page_key, page_data in list(pages.items())[:2]:  # Show first 2 pages as sample
            print(f"\nPage: {page_key}")
            print(f"Tags: {page_data['tags']}")
            print(f"Data preview: {page_data['data'][:200]}...")

    except Exception as e:
        print(f"Error: {str(e)}")
        

if __name__ == "__main__":
    download_web_content()
    