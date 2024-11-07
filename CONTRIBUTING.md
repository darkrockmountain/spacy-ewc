# Contributing to spaCy-EWC

Thank you for considering contributing to **spaCy-EWC**! Contributions of all kinds are welcome, including bug reports, feature suggestions, and code improvements. To ensure a smooth contribution process, please follow the guidelines below.

## Table of Contents

- [How to Report a Bug](#how-to-report-a-bug)
- [How to Suggest a Feature](#how-to-suggest-a-feature)
- [How to Set Up Your Development Environment](#how-to-set-up-your-development-environment)
- [Branch Naming Conventions](#branch-naming-conventions)
- [Coding Guidelines](#coding-guidelines)
- [How to Submit a Pull Request](#how-to-submit-a-pull-request)
- [Review and Approval Process](#review-and-approval-process)
- [Community Guidelines](#community-guidelines)

## How to Report a Bug

1. **Search Existing Issues**: Check if the bug has already been reported to avoid duplicates.
2. **Open a New Issue**: If the issue is new, open an issue and include:

   - A clear and descriptive title.
   - Steps to reproduce the bug.
   - Expected and actual results.
   - Relevant logs, screenshots, or other supporting information.

   Use our [Bug Report Template](./.github/ISSUE_TEMPLATE/bug_report.md) as a reference.

## How to Suggest a Feature

1. **Search Existing Issues**: Before suggesting a new feature, check if it has already been requested.
2. **Open a New Issue**: If the feature is new, open an issue and include:

   - A clear and descriptive title.
   - A detailed description of the feature.
   - Possible use cases and benefits.

   Use our [Feature Request Template](./.github/ISSUE_TEMPLATE/feature_request.md) as a guide.

## How to Set Up Your Development Environment

1. **Fork the Repository**: Fork the repository to your GitHub account.
2. **Clone the Repository**: Clone your forked repository to your local machine.
   ```bash
   git clone https://github.com/yourusername/spacy-ewc.git
   ```
3. **Navigate to the Project Directory**:
   ```bash
   cd spacy-ewc
   ```
4. **Install Dependencies**:
   - **Core dependencies**:
     ```bash
     pip install .
     ```
   - **Development dependencies** (recommended for contributors):
     ```bash
     pip install .[dev]
     ```
5. **Run Tests**: Verify that everything is working by running the tests.
   ```bash
   pytest
   ```

## Branch Naming Conventions

Please follow these branch naming conventions for clarity:

- **Feature branches**: `feature/your-feature-name`
- **Bugfix branches**: `bugfix/your-bug-name`
- **Documentation branches**: `docs/your-doc-name`
- **Hotfix branches**: `hotfix/your-hotfix-name`

## Coding Guidelines

1. **Code Style**: Follow Python's PEP 8 style guide. Use `black` to format code and `flake8` for linting.
2. **Documentation**: Document all public classes, functions, and methods. Include clear docstrings following the [Google style guide](https://google.github.io/styleguide/pyguide.html).
3. **Testing**: Write tests for new features or bug fixes. Ensure all tests pass before submitting a pull request.
4. **Commit Messages**: Write clear and concise commit messages. Follow this format:
   ```
   type(scope): description
   ```
   Example:
   ```
   feat(ner): add EWC penalty to NER component training
   ```

## How to Submit a Pull Request

1. **Create a Branch**: Create a new branch for your feature or bugfix.
   ```bash
   git checkout -b feature/my-new-feature
   ```
2. **Commit Your Changes**: Commit with a clear message.
   ```bash
   git commit -m "feat: add new feature"
   ```
3. **Push to Your Fork**: Push the branch to your forked repository.
   ```bash
   git push origin feature/my-new-feature
   ```
4. **Open a Pull Request**: Go to the original repository on GitHub and open a pull request. Describe your changes in detail.

   **Important**: Ensure the pull request is made against the `develop` branch, not `main`. PRs to `main` will be automatically rejected.

   Refer to our [Pull Request Template](./.github/PULL_REQUEST_TEMPLATE.md) when submitting.

## Review and Approval Process

1. **Review**: Project maintainers will review your pull request for code quality, adherence to guidelines, and overall impact.
2. **Approval**: If the pull request meets all requirements, it will be approved and merged. If adjustments are needed, feedback will be provided.
3. **Timeline**: Reviews typically take up to a few days. Please be patient as maintainers review and provide feedback.

## Community Guidelines

- **Respect**: Be respectful and constructive in all interactions.
- **Inclusivity**: Ensure contributions are inclusive and accessible to all.
- **Collaboration**: Work collaboratively and be open to feedback.
- **Confidentiality**: Avoid sharing private information (e.g., personal email addresses) without permission.
- **Code of Conduct**: By contributing, you agree to follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

By following these guidelines, you’ll help us maintain a positive and productive environment for all contributors. Thank you for helping improve spaCy-EWC!
