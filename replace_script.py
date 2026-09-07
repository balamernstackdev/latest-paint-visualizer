import os
import glob

def replace_in_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='utf-16') as f:
                content = f.read()
        except:
            return
        
    new_content = content.replace('width="stretch"', 'width="stretch"').replace('width="content"', 'width="content"')
    
    # replace st.html( with st.html(
    new_content = new_content.replace('st.html(', 'st.html(')
    new_content = new_content.replace('""", unsafe_allow_javascript=True)', '""", unsafe_allow_javascript=True)')
    new_content = new_content.replace('""", unsafe_allow_javascript=True', '""", unsafe_allow_javascript=True')
    
    if content != new_content:
        # Determine encoding for write
        enc = 'utf-8'
        try:
            with open(path, 'r', encoding='utf-8') as f:
                f.read()
        except:
            enc = 'utf-16'
        with open(path, 'w', encoding=enc) as f:
            f.write(new_content)
        print(f'Updated {path}')

for py_file in glob.glob('**/*.py', recursive=True):
    if 'venv' in py_file or 'env' in py_file or '.venv' in py_file:
        continue
    replace_in_file(py_file)
