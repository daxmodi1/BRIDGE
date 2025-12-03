# Contributing to BRIDGE

First off, thank you for considering contributing to BRIDGE! 🌉

BRIDGE (BERT Representations for Identifying Depression via Gradient Estimators) aims to help improve mental health awareness through AI and machine learning. Every contribution, no matter how small, makes a difference.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. By participating, you are expected to uphold this code.

### Our Standards

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of Machine Learning and NLP

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/mental-health-sentiment-analysis.git
   cd mental-health-sentiment-analysis
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/mental-health-sentiment-analysis.git
   ```

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates.

When creating a bug report, include:
- A clear and descriptive title
- Steps to reproduce the behavior
- Expected vs actual behavior
- Screenshots if applicable
- Your environment details (OS, Python version, etc.)

### 💡 Suggesting Enhancements

Enhancement suggestions are welcome! Please include:
- A clear and descriptive title
- Detailed description of the proposed enhancement
- Why this enhancement would be useful
- Possible implementation approach

### 🔧 Code Contributions

#### Good First Issues

Look for issues labeled `good first issue` or `beginner-friendly`.

#### Areas for Contribution

- **Model Improvements**: New algorithms, hyperparameter tuning
- **Feature Engineering**: New text features, preprocessing techniques
- **Documentation**: Improve README, add docstrings, tutorials
- **Testing**: Add unit tests, integration tests
- **Visualization**: New plots, interactive dashboards
- **Performance**: Optimization, efficiency improvements

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** from Kaggle and place it in `data/`

4. **Run notebooks** to ensure everything works:
   ```bash
   jupyter notebook
   ```

## Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise

```python
def predict_mental_health(text: str) -> str:
    """
    Predict mental health status from input text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        str: Predicted mental health category
        
    Example:
        >>> result = predict_mental_health("I feel anxious today")
        >>> print(result)
        'Anxiety'
    """
    # Implementation here
    pass
```

### Jupyter Notebooks

- Clear outputs before committing
- Use markdown cells to explain your analysis
- Keep code cells focused on single tasks
- Include visualizations where helpful

### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat(model): add LSTM-based classifier

- Implemented bidirectional LSTM model
- Added attention mechanism
- Achieved 91% accuracy on test set
```

## Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit them

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request** on GitHub

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] Tests added/updated (if applicable)
- [ ] PR description clearly explains changes
- [ ] Related issue linked (if applicable)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## How Has This Been Tested?
Describe testing approach

## Screenshots (if applicable)
Add screenshots here

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
```

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

---

Thank you for contributing! Together, we can make a difference in mental health awareness. 🧠💚
