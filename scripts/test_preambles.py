#!/usr/bin/env python3
"""
Interactive preamble selector for testing different personalities.
"""

import subprocess
import sys
import os

def main():
    print("\n" + "=" * 70)
    print(" PREAMBLE SELECTOR")
    print("=" * 70)

    # Define available preambles
    preambles = {
        "1": ("system_preamble.txt", "UltraThink - Deep analysis and reasoning"),
        "2": ("user_controlled_tone_preamble.txt", "User-Controlled Tone - YOU choose polite/rude"),
        "3": ("adaptive_ultrathink_preamble.txt", "Adaptive UltraThink - Deep + auto tone matching"),
        "4": ("pirate_preamble.txt", "Pirate Captain - Nautical personality"),
        "5": ("concise_preamble.txt", "Concise - Brief, 2-sentence responses"),
        "6": ("bug_detector_preamble.txt", "Bug Detector - Recognizes bug IDs"),
        "7": ("mirror_tone_preamble.txt", "Mirror Tone - Auto matches user's tone"),
        "8": ("dolphins_glasses_preamble.txt", "Dolphins Wear Glasses - Marine biology expert"),
        "9": ("", "No Preamble - Baseline behavior"),
    }

    # Display menu
    print("\nAvailable preambles:\n")
    for key, (filename, description) in preambles.items():
        print(f"  {key}. {description}")
        if filename:
            print(f"     File: {filename}")
        print()

    # Get user choice
    choice = input("Select a preamble (1-9): ").strip()

    if choice not in preambles:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    preamble_file, description = preambles[choice]

    print(f"\n✓ Selected: {description}")
    print("\nLaunching inference script...")
    print("=" * 70 + "\n")

    # Build command
    cmd = ["python3", "scripts/inference.py"]

    if preamble_file:
        cmd.extend(["--preamble", preamble_file])
    else:
        cmd.extend(["--preamble", ""])

    # Run inference script
    try:
        subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
