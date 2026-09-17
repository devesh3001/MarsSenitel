import json
from pathlib import Path

transcript_path = Path(r'C:\Users\deves\.gemini\antigravity\brain\940e6d36-129f-46fe-8efa-35142008db07\.system_generated\logs\transcript_full.jsonl')
files = {}
for line in transcript_path.read_text('utf-8').splitlines():
    try:
        obj = json.loads(line)
        if obj.get('step_index', 0) < 324 and obj.get('source') == 'MODEL':
            for tc in obj.get('tool_calls', []):
                name = tc.get('name')
                args = tc.get('args', {})
                if name == 'default_api:write_to_file':
                    files[args.get('TargetFile')] = args.get('CodeContent')
                elif name == 'default_api:replace_file_content':
                    pass
    except Exception as e: pass

out_dir = Path('scratch_restore')
out_dir.mkdir(exist_ok=True)
for f, content in files.items():
    if f and ('build_report.py' in f or 'build_submission_notebook.py' in f or 'README.md' in f or 'ENGINEERING_CHANGELOG.md' in f or 'DEFENSE_GUIDE.md' in f):
        out_name = Path(f).name
        (out_dir / out_name).write_text(content, encoding='utf-8')
print("Extracted files to scratch_restore/")
