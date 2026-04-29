#!/usr/bin/env python3
"""
Ada Guardian CLI - Standalone testing and integration interface for Ada.
"""

import argparse
import sys
import json
from pathlib import Path

try:
    from ada_guardian import Guardian
except ImportError:
    print("Error: ada_guardian module not found.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Ada Guardian - Security and validation layer for Ada integration"
    )
    parser.add_argument(
        "--check",
        type=str,
        help="Check a file or directory for issues"
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Validate JSON or YAML data from a file"
    )
    parser.add_argument(
        "--scan",
        type=str,
        help="Scan for security vulnerabilities in code"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Generate a report in given format (json, text, html)",
        default="text"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Ada Guardian 0.1.0"
    )

    args = parser.parse_args()

    if not any([args.check, args.validate, args.scan]):
        parser.print_help()
        sys.exit(1)

    guardian = Guardian()

    if args.check:
        result = guardian.check_path(args.check)
        if not args.quiet:
            print(json.dumps(result, indent=2) if args.report == "json" else result)
        sys.exit(0 if result.get("success", False) else 1)

    if args.validate:
        try:
            with open(args.validate, 'r') as f:
                data = f.read()
            result = guardian.validate_data(data)
            if not args.quiet:
                print(json.dumps(result, indent=2) if args.report == "json" else result)
            sys.exit(0 if result.get("valid", False) else 1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)

    if args.scan:
        result = guardian.scan_code(args.scan)
        if not args.quiet:
            if args.report == "json":
                print(json.dumps(result, indent=2))
            elif args.report == "html":
                print("<html><body><pre>" + str(result) + "</pre></body></html>")
            else:
                print(result)
        if result.get("vulnerabilities", []):
            sys.exit(1)
        sys.exit(0)


if __name__ == "__main__":
    main()
