from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DB_URL: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    MISTRAL_API_KEY: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_SERVER: str = "localhost"
    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )

    @property
    def DATABASE_URL(self):
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"


settings = Settings()
