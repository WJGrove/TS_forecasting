from config.settings import settings

def main() -> None:
    print(f"Environment: {settings.environment}")
    print(f"Log level:   {settings.log_level}")
    print(f"DB URL:      {settings.db_url}")
    print(f"OpenAI key set? {'yes' if settings.openai_api_key else 'no'}")

if __name__ == "__main__":
    main()
