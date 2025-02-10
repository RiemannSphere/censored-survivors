from setuptools import setup, find_packages

setup(
    name="censored-survivors",
    version="0.1.0",
    description="Survival analysis tools for customer churn prediction",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "lifelines>=0.27.0",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.8",
) 