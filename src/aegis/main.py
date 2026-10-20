from aegis.config import get_settings


def main() -> None:
    get_settings()
    print("AEGIS initialized.")

if __name__ == "__main__":
    main()