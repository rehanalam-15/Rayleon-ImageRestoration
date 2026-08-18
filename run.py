"""
Team Rayleon
AI-Based Restoration of Degraded Images

Official submission entry point.

Usage:
    python run.py <input-dir> <output-dir>
"""

import sys


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Rayleon Image Restoration pipeline initialized.")


if __name__ == "__main__":
    main()
