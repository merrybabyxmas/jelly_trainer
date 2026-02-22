"""
Code completion post-processing for HumanEval/MBPP evalplus evaluation.
Adapted from PiSSA/utils/code_process.py.

Input format (from gen_hf.generate_responses):
    [{'type': 'humaneval'/'mbpp', 'output': generated_code, 'answer': task_id}, ...]

Output format (for evalplus):
    {'humaneval': [{'task_id': str, 'completion': str}, ...],
     'mbpp':      [{'task_id': str, 'completion': str}, ...]}
"""
import json
import os


def clean_completion(completion: str) -> str:
    """Extract and clean code from model output (removes markdown fences, driver code, etc.)"""
    completion = completion.replace("\r", "")

    if '```python' in completion:
        def_line = completion.index('```python')
        completion = completion[def_line:].strip()
        completion = completion.replace('```python', '')
        try:
            next_line = completion.index('\n```')
            completion = completion[:next_line].strip()
        except ValueError:
            pass

    if '__name__ == "__main__"' in completion:
        next_line = completion.index('if __name__ == "__main__":')
        completion = completion[:next_line].strip()

    if "# Example usage" in completion:
        next_line = completion.index('# Example usage')
        completion = completion[:next_line].strip()

    if "assert" in completion:
        next_line = completion.index('assert')
        completion = completion[:next_line].strip()

    return completion


def process_code_completions(responses: list) -> dict:
    """
    Process generated code completions into evalplus-compatible format.

    Args:
        responses: list of {'type': 'humaneval'/'mbpp', 'output': generated_code, 'answer': task_id}
            - type:   'humaneval' or 'mbpp'
            - output: raw generated text from the model
            - answer: task_id string (e.g. 'HumanEval/0', 'Mbpp/2')

    Returns:
        dict with keys 'humaneval' and 'mbpp', each containing a list of
        {'task_id': str, 'completion': str} dicts ready for evalplus.
    """
    humaneval_output = []
    mbpp_output = []

    for item in responses:
        dtype = item.get('type', '')
        if dtype not in ('humaneval', 'mbpp'):
            continue
        task_id = str(item['answer'])
        completion = clean_completion(item['output'])
        entry = {'task_id': task_id, 'completion': completion}
        if dtype == 'humaneval':
            humaneval_output.append(entry)
        else:
            mbpp_output.append(entry)

    return {'humaneval': humaneval_output, 'mbpp': mbpp_output}


def write_evalplus_jsonl(entries: list, output_file: str):
    """Write processed completions to JSONL for evalplus.evaluate."""
    with open(output_file, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    print(f"[CODE_PROCESS] Saved {len(entries)} entries to {output_file}")
