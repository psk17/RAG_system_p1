# Contributing

Thank you for your interest in contributing to this project! We welcome bug reports, feature requests, documentation improvements, and code contributions.

## How to contribute

1. Fork the repository and create a new branch for your change.
2. Keep changes small and focused.
3. Run the test suite before submitting a pull request:
   ```bash
   poetry run pytest tests/ -q
   ```
4. Run linting to ensure style compliance:
   ```bash
   poetry run ruff check src tests
   ```
5. Open a pull request against the `main` branch with a clear description.

## Reporting issues

- Search existing issues before opening a new one.
- Provide a clear title, a description of the problem, and steps to reproduce.
- Include relevant logs or test output if available.

## Code style

- Follow the existing project conventions.
- Use `ruff` for linting and formatting checks.
- Keep documentation and comments up to date.

## Local development

- Create a `.env` file from `.env.example` if needed.
- Use the local Redis and environment setup when testing features that require memory or sessions.

Thanks again for contributing!