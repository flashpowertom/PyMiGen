"""Générateur d'image Mistral : script Python principal."""

import sys


def main() -> int:
    print("Bonjour depuis le Générateur d'image Mistral !")
    print("Arguments reçus :", sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
