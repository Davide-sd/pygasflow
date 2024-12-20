
"""
This module contains functions to modify the generation of the Sphinx
documentation. They are meant to be private and will be modified/deleted
as the code progresses. Use it at your own risk.
"""
import param
from param.parameterized import label_formatter
import inspect

param.parameterized.docstring_signature = False
param.parameterized.docstring_describe_params = False

# Parameter attributes which are never shown
IGNORED_ATTRS = [
    'precedence', 'check_on_set', 'instantiate', 'pickle_default_value',
    'watchers', 'compute_default_fn', 'doc', 'owner', 'per_instance',
    'is_instance', 'name', 'time_fn', 'time_dependent', 'allow_refs',
    'nested_refs', 'rx', 'label', 'softbounds', 'step'
]

# Default parameter attribute values (value not shown if it matches defaults)
DEFAULT_VALUES = {'allow_None': False, 'readonly': False, 'constant': False}


def should_skip_this_member(pobj, name, value, label):
    try:
        is_default = bool(DEFAULT_VALUES.get(name) == value)
    except Exception:
        is_default = False

    skip = (
        name.startswith('_') or
        name in IGNORED_ATTRS or
        inspect.ismethod(value) or
        inspect.isfunction(value) or
        value is None or
        is_default or
        (name == 'label' and pobj.label != label)
    )
    return skip


def process_List(child, pobj, ptype, label, members):
    lines = [f"{child} : {ptype}"]
    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if name == "class_":
            skip = True
        if name == "bounds":
            if (
                (value == (0, None))
                or (value == (None, None))
                or (value is None)
            ):
                skip = True
            elif (value[1] is None) and value[0]:
                name = "min_length"
                value = str(value[0])
            elif (value[0] is None) and value[1]:
                name = "max_length"
                value = str(value[1])
            else:
                name = "length"
                if value[0] != value[1]:
                    value = f"{value[0]} <= len({child}) <= {value[1]}"
                else:
                    value = str(value[0])
        if name == "item_type":
            if value is None:
                skip = True
            elif "'" in str(value):
                values = str(value).split(",")
                values = [s.split("'")[1] for s in values]
                if len(values) == 1:
                    value = values[0]
                else:
                    value = "(%s)" % ", ".join(values)
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def process_ClassSelector(child, pobj, ptype, label, members):
    types = pobj.class_
    if isinstance(types, type):
        types = str(types).split("'")[1]
    elif isinstance(types, tuple):
        types = str(types).split(",")
        for i, v in enumerate(types):
            types[i] = v.split("'")[1]
        types = "(%s)" % ", ".join(types)
    lines = [f"{child} : {types}"]

    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if name == "class_":
            skip = True
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def process_Number(child, pobj, ptype, label, members):
    lines = [f"{child} : {ptype}"]
    constant = pobj.constant
    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if constant and name == "default":
            skip = True
        if name == "constant":
            name = "read only"
        if name == "inclusive_bounds":
            skip = True
        if name == "bounds":
            inclusive_bounds = pobj.inclusive_bounds

            if (value == (None, None)) or (value is None):
                skip = True
            elif all(v is not None for v in value):
                symbol_1 = "<=" if inclusive_bounds[0] else "<"
                symbol_2 = "<=" if inclusive_bounds[1] else "<"
                value = f"{value[0]} {symbol_1} {child} {symbol_2} {value[1]}"
            elif value[0] is None:
                symbol = "<=" if inclusive_bounds[0] else "<"
                value = f"{child} {symbol} {value[1]}"
            else:
                symbol = ">=" if inclusive_bounds[1] else ">"
                value = f"{child} {symbol} {value[0]}"
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def process_Parameter(child, pobj, ptype, label, members):
    lines = [f"{child} : {ptype}"]
    for name, value in members:
        skip = should_skip_this_member(pobj, name, value, label)
        if name == "class_":
            skip = True
        if not skip:
            lines.append(f"   :{name}: {value}")
    return lines


def param_formatter(app, what, name, obj, options, lines):
    if what == 'module':
        lines = ["start"]

    if what == 'class' and isinstance(obj, param.parameterized.ParameterizedMetaclass):

        parameters = ['name']
        # lines.extend(["Notes", "====="])
        lines.extend(["Parameters", "=========="])

        params = [p for p in obj.param if p not in parameters]
        for child in params:
            if child in ["print_level", "name"]:
                continue
            if child[0] == "_":
                continue
            if (
                hasattr(obj, "_params_to_document")
                and child not in obj._params_to_document
            ):
                continue

            pobj = obj.param[child]
            label = label_formatter(pobj.name)
            doc = pobj.doc or ""
            members = inspect.getmembers(pobj)
            ptype = pobj.__class__.__name__

            if ptype == "List":
                func = process_List
            elif ptype == "ClassSelector":
                func = process_ClassSelector
            elif ptype in ["Number", "Integer"]:
                func = process_Number
            else:
                func = process_Parameter
            p_lines = func(child, pobj, ptype, label, members)
            p_lines.extend([
                '',
                '   %s' % doc,
                ''
            ])
            lines.extend(p_lines)


def _modify_example_compressible_app(code):
    lines = code.split("\n")
    if "show()" in lines[-1]:
        lines[-1] = lines[-1].split(".")[0]
    # I want to show the isentropic section, with a few numerical values
    # in the tabulator
    lines.insert(-1, "app.main[0].active = 1")
    # mach numbers
    lines.insert(-1, "app.sidebar[0].object.rx.value.objects[1].value = '2, 3, 4'")
    # gamma
    lines.insert(-1, "app.sidebar[0].object.rx.value.objects[3].value = '1.1, 1.2, 1.3, 1.4'")
    new_code = "\n".join(lines)
    return new_code


def _modify_diagram_show_only_figure(code):
    lines = code.split("\n")
    lines[-1] = "pn.pane.Bokeh(d.figure)"
    new_code = "\n".join(lines)
    return new_code


def _modify_diagram_show_interactive_app(code):
    lines = code.split("\n")
    lines[-1] = "d = " + lines[-1]
    lines.append("d.__panel__()")
    new_code = "\n".join(lines)
    return new_code


def modify_panel_code(code):
    if "compressible_app" in code:
        return _modify_example_compressible_app(code)
    elif "Diagram" in code:
        if "d.show_figure()" in code:
            return _modify_diagram_show_only_figure(code)
        return _modify_diagram_show_interactive_app(code)
    return code
