import os
import sys

# Add project root directory to sys.path so app module can be imported cleanly on Vercel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
