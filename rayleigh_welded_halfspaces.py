"""Backward-compatible script entrypoint.

This module now delegates to the importable package implementation.
"""

from rayleigh_welded_halfspaces import run_demo


def main():
    run_demo(show=True, output_prefix="mm_")


if __name__ == "__main__":
    main()
