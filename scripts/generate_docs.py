#!/usr/bin/env python3
"""Script to generate user-focused API documentation."""

import importlib
import inspect
from pathlib import Path
from typing import Any, List, Optional


def filter_readme_content(content: str) -> str:
    """Filter out development-related sections from README."""
    lines = content.split('\n')
    filtered_lines = []
    skip_section = False
    
    # Sections to skip
    skip_headers = [
        '## 🛠️ Installation',
        '### Prerequisites',
        '### Installation Options',
        '### Local Development Setup',
        '## 🧪 Development',
        '## 🤝 Contributing',
        '## 📄 License',
    ]
    
    for line in lines:
        # Check if we're entering a section to skip
        if any(line.startswith(header) for header in skip_headers):
            skip_section = True
            continue
            
        # Check if we're entering a new main section (##)
        if line.startswith('## '):
            skip_section = False
            
        # Add line if we're not in a section to skip
        if not skip_section:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


def format_example(example: str) -> str:
    """Format an example section, ensuring proper indentation."""
    lines = example.split('\n')
    # Remove common indentation
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def document_model_class(cls: Any) -> Optional[str]:
    """Document a model class (input/output models)."""
    if not cls.__module__.endswith('.models'):
        return None
        
    doc = inspect.getdoc(cls) or ""
    parts = [f"### {cls.__name__}\n"]
    
    if doc:
        parts.append(f"{doc}\n")
    
    # Get fields and their types
    try:
        if hasattr(cls, '__annotations__'):
            parts.append("**Fields:**\n")
            for field_name, field_type in cls.__annotations__.items():
                if not field_name.startswith('_'):
                    parts.append(f"- `{field_name}`: `{field_type.__name__ if hasattr(field_type, '__name__') else str(field_type)}`")
            parts.append("\n")
    except Exception:
        pass
    
    return "\n".join(parts)


def document_runner_class(cls: Any) -> Optional[str]:
    """Document a runner class (the main model implementation classes)."""
    if not cls.__module__.endswith('.run'):
        return None
        
    doc = inspect.getdoc(cls) or ""
    parts = [f"### {cls.__name__}\n"]
    
    if doc:
        parts.append(f"{doc}\n")
    
    # Document the main methods (fit, predict, fit_predict)
    main_methods = ['fit', 'predict', 'fit_predict']
    for method_name in main_methods:
        if hasattr(cls, method_name):
            method = getattr(cls, method_name)
            doc = inspect.getdoc(method) or ""
            if doc:
                parts.append(f"#### `{method_name}`\n")
                parts.append(f"{doc}\n")
    
    return "\n".join(parts)


def document_module_usage(module_name: str) -> str:
    """Generate user-focused documentation for a module."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        return f"Error importing {module_name}: {e}"
    
    # Get the module name without the full path
    short_name = module_name.split('.')[-2]  # e.g., 'kaplan_meier' from 'censored_survivors.kaplan_meier.models'
    
    parts: List[str] = []
    
    # Only add module header for the first part (models)
    if module_name.endswith('.models'):
        parts.append(f"## {short_name.replace('_', ' ').title()}\n")
        
        # Add module docstring if it exists
        if module.__doc__:
            parts.append(f"{module.__doc__.strip()}\n")
    
    # Document classes
    classes = inspect.getmembers(module, predicate=inspect.isclass)
    for name, cls in classes:
        if cls.__module__ == module_name and not name.startswith('_'):
            if module_name.endswith('.models'):
                doc = document_model_class(cls)
            else:
                doc = document_runner_class(cls)
            if doc:
                parts.append(doc)
    
    return "\n".join(parts)


def main():
    """Generate user-focused API documentation."""
    output_dir = Path("docs/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "documentation.md"
    
    # Read and filter the README content
    readme_content = Path("README.md").read_text()
    filtered_readme = filter_readme_content(readme_content)
    
    # Define the module pairs we want to document
    module_pairs = [
        ("censored_survivors.kaplan_meier.models", "censored_survivors.kaplan_meier.run"),
        ("censored_survivors.cox_proportional_hazards.models", "censored_survivors.cox_proportional_hazards.run"),
        ("censored_survivors.weibull_rnn.models", "censored_survivors.weibull_rnn.run"),
    ]
    
    # Start with the overview
    docs = [
        "# Censored Survivors\n",
        "## Overview\n",
        filtered_readme,
        "\n# API Reference\n",
        "\nThis section describes the main components and how to use them in your code.\n"
    ]
    
    # Generate documentation for each module pair
    for models_module, run_module in module_pairs:
        docs.append(document_module_usage(models_module))
        docs.append(document_module_usage(run_module))
    
    # Write the documentation
    output_file.write_text("\n".join(docs))
    print(f"\nAPI documentation generated: {output_file}")
    print("You can now open this markdown file in any markdown viewer.")


if __name__ == "__main__":
    main() 