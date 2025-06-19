import qrcode

def generate_github_qr(repo_url, filename="github_repo_qr.png"):
    """Generates a QR code for a GitHub repository URL.

    Args:
        repo_url: The GitHub repository URL.
        filename: The name of the file to save the QR code as.
    """
    try:
        img = qrcode.make(repo_url)
        img.save(filename)
        print(f"QR code generated and saved as {filename}")
    except Exception as e:
        print(f"Error generating QR code: {e}")

if __name__ == "__main__":
    github_repo_url = "https://github.com/kpnowak/OUH-Internship-Krzysztof-Nowak.git" # Replace with your repo URL
    generate_github_qr(github_repo_url)