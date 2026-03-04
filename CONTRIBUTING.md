# Contributing to AI Interior Design Studio

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/ai-interior-design.git
   cd ai-interior-design
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

5. Download model weights:
   ```bash
   python scripts/download_weights.py
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and modular

## Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request

## Testing

Run tests before submitting:
```bash
pytest tests/ -v
```

## Pull Request Guidelines

- Reference any related issues
- Include a clear description of changes
- Update documentation if needed
- Add tests for new functionality
- Ensure all tests pass

## Reporting Bugs

When reporting bugs, please include:
- Python version
- Operating system
- GPU details (if applicable)
- Steps to reproduce
- Expected vs actual behavior

## Questions?

Feel free to open an issue for any questions or discussions.
