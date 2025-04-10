# Contributing to Quant-Aero-Log

Thank you for your interest in contributing to Quant-Aero-Log! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/quant-aero-log.git
   cd quant-aero-log
   ```
3. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards:
   - Use type hints
   - Write docstrings
   - Follow PEP 8 style guide
   - Write tests for new functionality

3. Run tests and checks:
   ```bash
   pytest
   mypy .
   black .
   flake8
   ```

4. Commit your changes:
   ```bash
   git commit -m "Description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a Pull Request

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for all function parameters and return values
- Write docstrings following [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 90%
- Use pytest for testing
- Include both positive and negative test cases

### Documentation

- Update documentation for all new features
- Include usage examples
- Document any breaking changes

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include relevant tests
3. Update documentation as needed
4. Ensure all tests pass
5. Request review from maintainers

## Review Process

- PRs will be reviewed by maintainers
- Feedback will be provided within a reasonable timeframe
- Changes may be requested before merging

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Build and publish the package

## Questions?

Feel free to open an issue or contact the maintainers for any questions about contributing. 