"""
UserProfile — structured representation of the user's professional background.

The profile is the fixed input to every matching and generation step:
  - The rule-based matcher uses it to compute relevance scores
  - The AI ranking agent will receive it verbatim as prompt context
  - Resume tailoring and cover letter agents read it to select which
    skills and experiences to emphasise per job

Storing the profile in code (rather than in .env or a database) is intentional:
  - It is version-controlled alongside the rest of the project
  - Changes to the profile are explicit, reviewable, and dated via git history
  - The LLM receives a structured object, not a free-text string, so every
    field can be independently validated and referenced by name
"""

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """The user's professional background used for all matching and generation."""

    years_of_experience: int
    primary_languages: list[str]
    frameworks: list[str]
    backend_skills: list[str]
    databases: list[str]
    cloud: list[str]
    preferred_roles: list[str]
    preferred_locations: list[str]

    def all_skills(self) -> list[str]:
        """Flat list of every technical skill across all categories."""
        return (
            self.primary_languages
            + self.frameworks
            + self.backend_skills
            + self.databases
            + self.cloud
        )

    def summary(self) -> str:
        return (
            f"{self.years_of_experience} YoE | "
            f"Languages: {', '.join(self.primary_languages)} | "
            f"Frameworks: {', '.join(self.frameworks)} | "
            f"Roles: {', '.join(self.preferred_roles)}"
        )


# Default profile matching the user's current background.
# Update this as skills and preferences evolve.
DEFAULT_PROFILE = UserProfile(
    years_of_experience=3,
    primary_languages=["Java", "SQL"],
    frameworks=["Spring Boot", "Spring MVC", "Hibernate", "JUnit"],
    backend_skills=[
        "REST APIs",
        "Microservices",
        "Distributed Systems",
        "System Design",
        "Git",
        "Maven",
        "Gradle",
        "Docker",
        "Kafka",
        "Redis",
    ],
    databases=["PostgreSQL", "MySQL", "MongoDB", "SQL"],
    cloud=["AWS", "GCP"],
    preferred_roles=[
        "Software Engineer II",
        "Backend Engineer",
        "Platform Engineer",
        "Java Backend Developer",
        "Senior Software Engineer",
        "Software Engineer",
        "Backend Developer",
        "Java Developer",
    ],
    preferred_locations=["Remote", "San Francisco, CA", "Seattle, WA", "New York, NY"],
)
