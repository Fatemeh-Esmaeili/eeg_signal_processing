#!/usr/bin/env python3
"""
EEG Pipeline Runner
Simple script to execute the filtering pipeline
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import our module
sys.path.append(os.path.dirname(__file__))

from eeg_filtering_pipeline import EEGFilteringPipeline

def main():
    # Your configuration
    root_path = "PEEG"
    output_directory = "filtered_EEG"
    
    print("🚀 Starting EEG Filtering Pipeline...")
    
    # Initialize pipeline
    pipeline = EEGFilteringPipeline(root_path, output_directory)

    # Try different file patterns until we find your files
    file_patterns = ["**/*.eeg", "**/*.vhdr", "**/*.edf", "**/*.set", "**/*.fif"]
    
    # ✅ These are the correct parameters from your notebook
    print(f"Using filter range: {pipeline.filter_params['l_freq']}-{pipeline.filter_params['h_freq']} Hz")
    
    for pattern in file_patterns:
        files = list(Path(root_path).glob(pattern))
        if files:
            print(f"✅ Found {len(files)} files with pattern: {pattern}")
            pipeline.run_pipeline(file_pattern=pattern)
            break
        else:
            print("❌ No EEG files found with any pattern")
            # Run the pipeline
            pipeline.run_pipeline()
    
    

if __name__ == "__main__":
    main()