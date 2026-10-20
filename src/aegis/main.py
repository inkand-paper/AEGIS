from aegis.config import get_settings
from aegis.log import setup_logging


def main() -> None:
    setup_logging()
    get_settings()  # fails fast if the environment is misconfigured
    print("AEGIS initialized.")


if __name__ == "__main__":
    main()