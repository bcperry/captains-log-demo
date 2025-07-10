# Contributing to Captain's Log

Thank you for your interest in contributing to Captain's Log! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when available
3. **Provide detailed information** including:
   - Environment details (OS, Python version, Azure region)
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs
   - Screenshots if applicable

### Feature Requests

1. **Check the roadmap** first to see if it's already planned
2. **Open a discussion** for major features before implementing
3. **Provide use cases** and explain the value proposition
4. **Consider backward compatibility** implications

### Code Contributions

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** (see below)
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with clear description

## 🏗️ Development Setup

### Prerequisites

- Python 3.11+
- Azure subscription with Speech Services and OpenAI
- Git
- FFmpeg

### Local Development

1. **Clone your fork**
   ```bash
   git clone https://github.com/bcperry/captains-log-demo.git
   cd captains-log-demo
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r app/requirements.txt
   pip install -r dev-requirements.txt  # If available
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .azure/captainslog/.env
   # Edit .env with your Azure credentials
   ```

5. **Run the application**
   ```bash
   cd app
   streamlit run app.py
   ```

### Development Dependencies

For development, you might want to install additional tools:

```bash
pip install black flake8 pytest pytest-cov mypy
```

## 📝 Coding Standards

### Python Style Guide

- Follow **PEP 8** style guidelines
- Use **Black** for code formatting
- Maximum line length: **88 characters**
- Use **type hints** for function signatures
- Write **docstrings** for classes and functions

### Code Structure

```python
def function_name(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
    """
    # Implementation
    pass
```

### Error Handling

- Use **specific exception types** instead of generic `Exception`
- **Log errors** appropriately with context
- **Provide meaningful error messages** to users
- **Handle Azure service errors** gracefully

### Security Guidelines

- **Never commit secrets** or credentials
- **Use environment variables** for configuration
- **Validate input** from users
- **Sanitize file uploads** and names
- **Follow Azure security best practices**

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_transcription.py
```

### Writing Tests

- Write tests for **new functionality**
- Test **error conditions** and edge cases
- Use **mocking** for Azure services in tests
- Follow **AAA pattern** (Arrange, Act, Assert)

Example test structure:
```python
def test_transcription_with_valid_audio():
    # Arrange
    transcriber = AudioTranscriber()
    audio_data = create_test_audio()
    
    # Act
    result = transcriber.transcribe_audio(audio_data)
    
    # Assert
    assert result.success is True
    assert len(result.text) > 0
```

## 📚 Documentation

### Code Documentation

- **Docstrings** for all public methods and classes
- **Inline comments** for complex logic
- **Type hints** for better IDE support
- **README updates** for new features

### User Documentation

- Update **README.md** for new features
- Add **troubleshooting** entries for common issues
- Include **examples** in documentation
- Keep **deployment guides** current

## 🔄 Pull Request Process

### Before Submitting

1. **Test your changes** locally
2. **Run code formatting** with Black
3. **Check for linting errors** with flake8
4. **Update documentation** if needed
5. **Add tests** for new functionality

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tested locally
- [ ] Added/updated tests
- [ ] All tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in different environments
4. **Documentation review** if applicable
5. **Merge** when approved

## 🎯 Areas for Contribution

### High Priority

- **Performance improvements** for large audio files
- **Error handling** enhancements
- **Accessibility** features
- **Mobile responsiveness**

### Medium Priority

- **Additional language support**
- **Export format options**
- **Batch processing** features
- **Configuration management**

### Low Priority

- **UI/UX improvements**
- **Additional AI analysis features**
- **Integration with other services**
- **Performance monitoring**

## 🏷️ Git Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `docs/description` - Documentation updates

### Commit Messages

Follow conventional commits format:
```
type(scope): brief description

Detailed description if needed

Fixes #issue-number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 🌟 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **Special mentions** in project updates

## 📞 Getting Help

- **GitHub Discussions** for questions
- **Discord/Slack** for real-time chat (if available)
- **Email maintainers** for sensitive issues

## 📋 Code of Conduct

Please be respectful and professional in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

Thank you for contributing to Captain's Log! 🚀
