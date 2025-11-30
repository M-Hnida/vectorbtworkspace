"""
Comprehensive stub generator for VectorBT.
Generates complete type stubs for all Portfolio methods and properties.
"""

import inspect
from pathlib import Path
from typing import Any


def get_signature_string(func, name: str, is_method: bool = True) -> str:
    """Get a comprehensive signature string for a function."""
    try:
        sig = inspect.signature(func)
        params = []
        
        # Add 'self' for instance methods
        if is_method:
            params.append("self")
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
                
            # Build parameter with Any type (comprehensive, not strict)
            param_str = f"{param_name}: Any"
            
            # Add default value
            if param.default != inspect.Parameter.empty:
                if param.default is None:
                    param_str += " = None"
                elif isinstance(param.default, str):
                    param_str += f' = "{param.default}"'
                elif isinstance(param.default, (int, float, bool)):
                    param_str += f" = {param.default}"
                else:
                    param_str += " = ..."
            
            params.append(param_str)
        
        params_str = ", ".join(params)
        return f"def {name}({params_str}) -> Any: ..."
        
    except Exception:
        # Fallback for complex signatures
        return f"def {name}(self, *args: Any, **kwargs: Any) -> Any: ..."


def generate_class_stub(cls, class_name: str) -> list[str]:
    """Generate comprehensive stub content for a class."""
    lines = [f"class {class_name}:", '    """VectorBT {class_name} with type hints for IntelliSense"""', ""]
    
    # Get all public attributes
    attrs = [attr for attr in dir(cls) if not attr.startswith('_')]
    
    # Separate into categories
    static_methods = []
    class_methods = []
    properties = []
    regular_methods = []
    
    for attr_name in attrs:
        try:
            attr = getattr(cls, attr_name)
            
            # Check if it's a static method
            if isinstance(inspect.getattr_static(cls, attr_name), staticmethod):
                static_methods.append(attr_name)
            # Check if it's a class method
            elif isinstance(inspect.getattr_static(cls, attr_name), classmethod):
                class_methods.append(attr_name)
            # Check if it's a property
            elif isinstance(inspect.getattr_static(cls, attr_name), property):
                properties.append(attr_name)
            # Check if it's a method
            elif callable(attr):
                regular_methods.append(attr_name)
                
        except Exception:
            # If we can't determine, treat as regular method if callable
            if callable(getattr(cls, attr_name, None)):
                regular_methods.append(attr_name)
    
    # Generate static methods
    if static_methods:
        lines.append("    # Static methods")
        for method_name in sorted(static_methods):
            try:
                method = getattr(cls, method_name)
                sig = get_signature_string(method, method_name, is_method=False)
                lines.append(f"    @staticmethod")
                lines.append(f"    {sig}")
                lines.append("")
            except Exception:
                lines.append(f"    @staticmethod")
                lines.append(f"    def {method_name}(*args: Any, **kwargs: Any) -> Any: ...")
                lines.append("")
    
    # Generate class methods
    if class_methods:
        lines.append("    # Class methods")
        for method_name in sorted(class_methods):
            try:
                method = getattr(cls, method_name)
                # Class methods should have 'cls' not 'self'
                sig = get_signature_string(method, method_name, is_method=False)
                sig = sig.replace("def " + method_name + "(", "def " + method_name + "(cls, ", 1)
                lines.append(f"    @classmethod")
                lines.append(f"    {sig}")
                lines.append("")
            except Exception:
                lines.append(f"    @classmethod")
                lines.append(f"    def {method_name}(cls, *args: Any, **kwargs: Any) -> Any: ...")
                lines.append("")
    
    # Generate properties
    if properties:
        lines.append("    # Properties")
        for prop_name in sorted(properties):
            lines.append(f"    @property")
            lines.append(f"    def {prop_name}(self) -> Any: ...")
            lines.append("")
    
    # Generate regular methods
    if regular_methods:
        lines.append("    # Methods")
        for method_name in sorted(regular_methods):
            try:
                method = getattr(cls, method_name)
                sig = get_signature_string(method, method_name)
                lines.append(f"    {sig}")
                lines.append("")
            except Exception:
                lines.append(f"    def {method_name}(self, *args: Any, **kwargs: Any) -> Any: ...")
                lines.append("")
    
    return lines


def generate_vectorbt_stubs(output_path: str):
    """Generate comprehensive vectorbt stub file."""
    try:
        import vectorbt as vbt
    except ImportError:
        print("❌ Error: vectorbt is not installed")
        return
    
    lines = [
        "# vectorbt.pyi",
        "# Comprehensive type stubs for vectorbt library",
        "# Generated by scripts/generate_vectorbt_stubs.py",
        "",
        "from typing import Optional, Union, Any, Callable, Tuple, List, Dict",
        "import pandas as pd",
        "import numpy as np",
        "import numpy.typing as npt",
        "",
        ""
    ]
    
    # Generate Portfolio class
    print("📝 Generating Portfolio class stubs...")
    lines.extend(generate_class_stub(vbt.Portfolio, "Portfolio"))
    lines.append("")
    
    # Try to get IndicatorFactory
    try:
        if hasattr(vbt, 'IndicatorFactory'):
            print("📝 Generating IndicatorFactory stubs...")
            lines.extend(generate_class_stub(vbt.IndicatorFactory, "IndicatorFactory"))
            lines.append("")
    except Exception as e:
        print(f"⚠️  Could not generate IndicatorFactory stubs: {e}")
    
    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✅ Successfully generated stubs at: {output_path}")
    print(f"📊 Total lines generated: {len(lines)}")
    
    # Count methods
    methods = len([l for l in lines if '    def ' in l])
    properties = len([l for l in lines if '    @property' in l])
    print(f"   - Properties: {properties}")
    print(f"   - Methods: {methods - properties}")


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "vectorbt.pyi"
    generate_vectorbt_stubs(str(output_path))
