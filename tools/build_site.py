#!/usr/bin/env python3
import argparse
import sys
from sitegen import build_outputs, write_outputs


def main():
    parser = argparse.ArgumentParser(description="Generate the static xwcv site from data/*.json")
    parser.add_argument("--check", action="store_true", help="report generated files that differ without writing")
    args = parser.parse_args()
    changed = write_outputs(build_outputs(), check=args.check)
    if args.check and changed:
        print("Generated files are out of date:")
        for path in changed:
            print(" - " + path)
        return 1
    print(("Generated files are up to date" if args.check else "Site is up to date") if not changed else ("Generated %d file(s)" % len(changed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
