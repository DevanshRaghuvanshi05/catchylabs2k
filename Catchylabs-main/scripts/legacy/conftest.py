from pathlib import Path

collect_ignore = [path.name for path in Path(__file__).parent.glob('*.py') if path.name != 'conftest.py']
