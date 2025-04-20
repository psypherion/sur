echo "Installing Necessary Tools: FFMPEG and all required libraries"
echo "Now Running -> sudo apt install ffmpeg"

sudo apt install ffmpeg

# Preparing environment
echo "Now preparing Environment."
python -m venv venv
source venv/bin/activate

echo "Now running -> pip install -r requirements.txt"
pip install -r requirements.txt



