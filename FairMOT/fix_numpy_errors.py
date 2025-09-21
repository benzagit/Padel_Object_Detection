# File: fix_numpy_errors.py
import os
import re

def fix_numpy_in_file(filepath):
    """Reads a file, fixes numpy deprecations, and writes back if changed."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    
    # Fix 1: np.float -> float
    content, count1 = re.subn(r'np\.float', 'float', content)

    # Fix 2: Add np. prefix to standalone float32
    has_float32 = 'float32' in content
    has_numpy_import = 'import numpy as np' in content
    
    # Add import if needed
    if has_float32 and not has_numpy_import:
        if 'from __future__ import' in content:
            # Add import after __future__ imports
            content = re.sub(r'(from __future__ import .*\n)', r'\1import numpy as np\n', content, 1)
        else:
            content = 'import numpy as np\n' + content
    
    # Replace standalone float32 with np.float32
    content, count2 = re.subn(r'(?<!np\.)float32', 'np.float32', content)

    if original_content != content:
        print(f'Fixed: {filepath} ({count1+count2} replacements)')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    """Main function to walk through the src directory and fix files."""
    src_dir = 'src'
    if not os.path.isdir(src_dir):
        print(f'Error: Directory "{src_dir}" not found in current location.')
        return

    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                fix_numpy_in_file(os.path.join(root, file))
    print("\nNumPy code fixes applied.")

if __name__ == '__main__':
    main()
