"""
Math accuracy evaluation utilities.
Adapted from PiSSA/utils/test_acc.py.

Core functions for extracting and comparing math answers from model outputs.
Supports: gsm8k, math, boolq, piqa, siqa, arc_challenge, arc_easy, openbookqa,
          hellaswag, winogrande
"""
import json
import re
from fractions import Fraction
from collections import defaultdict


# ============================================================
# String normalization (for MATH symbolic comparison)
# ============================================================

def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except AssertionError:
        return string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


# ============================================================
# Answer extraction
# ============================================================

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    """Extract numeric answer from text containing 'The answer is: X'."""
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    frac = Fraction(match.group().replace(',', ''))
                    return round(float(frac.numerator / frac.denominator))
                return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        return None
    return None


def process_math_results(completion, answer):
    """Check if model completion contains correct MATH answer."""
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0].strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        return is_equiv(extract_ans, answer)
    return False


def extract_commonsense_answer(dataset, sentence: str) -> str:
    """Extract structured answer for commonsense tasks."""
    sentence_ = sentence.strip()
    patterns = {
        'boolq': r'true|false',
        'piqa': r'solution1|solution2',
        'siqa': r'answer1|answer2|answer3|answer4|answer5',
        'arc_challenge': r'answer1|answer2|answer3|answer4|answer5',
        'arc_easy': r'answer1|answer2|answer3|answer4|answer5',
        'openbookqa': r'answer1|answer2|answer3|answer4|answer5',
        'hellaswag': r'ending1|ending2|ending3|ending4',
        'winogrande': r'option1|option2',
    }
    pattern = patterns.get(dataset)
    if pattern:
        pred_answers = re.findall(pattern, sentence_)
        return pred_answers[0] if pred_answers else ""
    return ""


# ============================================================
# Evaluation
# ============================================================

def evaluate_results(data_list):
    """
    Evaluate a list of result dicts.

    Args:
        data_list: list of dicts with keys:
            - type: 'gsm8k', 'math', 'boolq', etc.
            - output: model-generated text
            - answer: ground truth answer string

    Returns:
        dict with per-type accuracy and 'overall' accuracy
    """
    results = defaultdict(list)
    for data in data_list:
        dtype = data['type']
        output = data['output']
        answer = data['answer']

        if dtype == 'gsm8k':
            y_pred = extract_answer_number(output)
            if y_pred is not None:
                try:
                    results[dtype].append(float(y_pred) == float(answer))
                except (ValueError, TypeError):
                    results[dtype].append(False)
            else:
                results[dtype].append(False)

        elif dtype == 'math':
            results[dtype].append(process_math_results(output, answer))

        elif dtype in ('boolq', 'piqa', 'siqa', 'arc_challenge', 'arc_easy',
                       'openbookqa', 'hellaswag', 'winogrande'):
            y_pred = extract_commonsense_answer(dtype, output)
            results[dtype].append(y_pred == answer if y_pred else False)

    acc_dict = {}
    total_correct = 0
    total_count = 0
    for key, values in results.items():
        acc = sum(values) / len(values) if values else 0.0
        acc_dict[key] = acc
        total_correct += sum(values)
        total_count += len(values)

    acc_dict['overall'] = total_correct / total_count if total_count > 0 else 0.0
    acc_dict['counts'] = {k: len(v) for k, v in results.items()}
    return acc_dict


def evaluate_jsonl(input_file):
    """
    Evaluate a JSONL file (compatible with gen_vllm.py output format).

    Args:
        input_file: path to JSONL file, each line has type/output/answer fields

    Returns:
        dict with per-type accuracy and overall
    """
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return evaluate_results(data_list)


def print_acc_results(acc_dict, prefix="[TEST ACC]"):
    """Pretty-print accuracy results."""
    print(f"\n{'='*60}")
    counts = acc_dict.get('counts', {})
    for key, acc in acc_dict.items():
        if key in ('overall', 'counts'):
            continue
        n = counts.get(key, '?')
        print(f"{prefix} {key:<20} acc={acc*100:.2f}%  (n={n})")
    print(f"{prefix} {'overall':<20} acc={acc_dict.get('overall', 0)*100:.2f}%  "
          f"(n={sum(counts.values()) if counts else '?'})")
    print(f"{'='*60}\n")


# ============================================================
# CLI entry point (compatible with original test_acc.py usage)
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()

    acc_dict = evaluate_jsonl(args.input_file)
    counts = acc_dict.get('counts', {})
    for key, acc in acc_dict.items():
        if key in ('overall', 'counts'):
            continue
        n = counts.get(key, len([]))
        print(f'{key} length==== {n} , {key} acc==== {acc}')
