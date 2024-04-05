from argparse import ArgumentParser
import itertools
from typing import List
import random


def try_parse_value(val: str):
    """
    Try cast first to int, else float, else string.
    """
    if isinstance(val, bool):
        return val

    classes = (int, float, str)
    for cl in classes:
        try:
            val = cl(val)
            return val
        except BaseException:
            pass
    return val


def split_key(key: str, val) -> dict:
    r = {}
    if "." not in key:
        return {key: val}

    r = {}
    r[key.split(".")[-1]] = val
    for k in reversed(key.split(".")[:-1]):
        r = {k: r}

    return r


def merge_keys(base: dict, other: dict) -> dict:
    for k in other.keys():
        if isinstance(other[k], dict):
            if k in base:
                base[k] = {**base[k], **other[k]}
            else:
                base[k] = other[k]
        else:
            base[k] = other[k]
    return base


def check_required_args(parser: ArgumentParser, unknown_args: List[str], known_args: dict):
    """
    Check if all required args are passed.
    """
    unknown_args_strings = [a.split('=')[0].split()[0] for a in unknown_args]
    required_args = [a for a in parser._actions if a.required]
    err_args = []
    for option_args in required_args:
        option_strings = option_args.option_strings
        if option_args.default is None and not any([option_string in unknown_args_strings or option_string.split('--')[1] in known_args.keys() for option_string in option_strings]):
            choices = option_args.choices if not isinstance(
                option_args.choices, dict) else list(option_args.choices.keys())
            err_args.append(
                f'[{", ".join(option_strings)}]: ({choices}) {option_args.help}')

    return len(err_args) == 0, err_args


def get_default_args(parser: ArgumentParser):
    args_with_default = [a for a in parser._actions if a.default is not None or (
        a.choices is not None and None in a.choices)]

    return {kr: arg.default for arg in args_with_default if (k := (arg.option_strings if isinstance(
        arg.option_strings, str) else arg.option_strings[0])) and (kr := (k.split('--')[1] if '--' in k else k.split('-')[1]))}


def create_params_list(params: dict) -> List[dict]:
    """
    Build list of arguments given the params.
    A different list is created for each given experiment.

    :params: Input in form {'key': [values...], ...}
    :returns: [{ 'key': <value1>, ...}, { 'key': <value2>, ...}]
    """

    vals = []
    for k in params.keys():
        vals.append(params[k])
    prod = itertools.product(*vals)

    r = [{k: v for k, v in zip(params.keys(), args)} for args in prod]

    return clean_param_list(r)


def clean_param_list(params: dict) -> List[dict]:
    """
    Cleans parameters dictionary.

    Returns: list of dicts with params.
    """
    ret = []

    for job_params in params:
        # check empty params
        if not job_params:
            continue

        ret.append(job_params)

    return ret


def parse_unknown_args(args_str: str, single_values=False, try_parse=False, split_keys=False) -> dict:
    """
    Parse arg string with args in form:
    --key=value
    --key=[value1, value2,...]
    --key value
    --key value1, value2

    single_values: if True return dictionary with only first value per key: { key: value, ...}.
    Returns: dictionary with parsed args { key: [values], ... }.
    """
    parts = args_str.strip().split('--')[1:]  # part 0 is empty by default

    ret = {}
    for part in parts:
        if part == "":
            continue

        part = part.strip()
        if len(part.split('=')) > 1:  # form key=value
            p = part.split('=')

            key = p[0]
            value = p[1]
        else:  # form -key value
            p = part.split()
            key = p[0]

            if len(p) == 1:
                # print(f"WARNING: Param '{p}' should be set in ArgumentParser. Will default to 'True' instead.")
                value = True
            elif len(p) > 2:
                value = "[" + ",".join(p[1:]) + "]"
            else:
                value = p[1]

        key = key[1:] if key.startswith(
            '-') else key  # check if form -key or key

        if isinstance(value, bool):
            if single_values:
                values = value
            else:
                values = [value]
        elif value.startswith('['):
            assert value.endswith(']') and not single_values
            values = [v.strip()
                      for v in value[1:-1].split(',') if v.strip() != '']
        elif len(value.split(',')) > 1:
            assert not value.endswith(']') and not single_values
            values = [v.strip() for v in value.split(',') if v.strip() != '']
        else:
            assert not value.endswith(']')
            if single_values:
                values = value
            else:
                values = [value]

        if try_parse:
            if isinstance(values, list):
                values = [try_parse_value(v) for v in values]
            else:
                values = try_parse_value(values)

        if split_keys:
            ret = merge_keys(ret, split_key(key, values))
        else:
            ret = merge_keys(ret, {key: values})
    return ret
