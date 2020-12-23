import numpy as np
from pygasflow.utils.common import convert_to_ndarray, ret_correct_vals
from timeit import default_timer as timer
from functools import wraps
import inspect

# NOTE: I have decided to not explicitely do a type chek on variables.
# This means you can pass any value on any variable: if it works, great,
# if it doesn't, that's your problem: read the docs and be smart. :)

def _check_specific_heat_ratio(gamma):
    if gamma < 1:
        raise ValueError("The specific heats ratio must be > 1.")

def _check_mach_number(M, value):
    if not np.all(M >= value):
        raise ValueError("The Mach number must be >= {}.".format(value))

def _check_flag(flag):
    flag = flag.lower()
    if flag not in ["sub", "super"]:
        raise ValueError("Flag can be either 'sub' or 'super'.")
    return flag

def _check_flag_shockwave(flag):
    flag = flag.lower()
    if flag not in ["weak", "strong"]:
        raise ValueError("Flag can be either 'weak' or 'strong'.")
    return flag

# TODO: this approach let the user insert an angle=None even into
# functions that need a float value!
def _check_angle(angle_name, angle):
    #  some functions can accept an angle with value None!
    if angle is not None:
        angle = np.asarray(angle)
        if np.any(angle < 0) or np.any(angle > 90):
            raise ValueError("The {} must be 0 <= {} <= 90.".format(*angle_name))


# This decorator is used to convert and check the arguments of the
# functions in the shockwave module
def check_shockwave(var=None):
    def decorator(original_function):
        indeces = [0]
        if not callable(var):
            indeces = var

        @wraps(original_function)
        @as_array(indeces)
        def wrapper_function(*args, **kwargs):
            args = list(args)
            all_param = get_parameters_dict(original_function, *args, **kwargs)
            
            if "M" in all_param.keys():
                _check_mach_number(all_param["M"], 0)
            if "M1" in all_param.keys():
                _check_mach_number(all_param["M1"], 1)
            if "MN1" in all_param.keys():
                _check_mach_number(all_param["MN1"], 0)
            if "gamma" in all_param.keys():
                _check_specific_heat_ratio(all_param["gamma"])
            if "beta" in all_param.keys():
                _check_angle(["shock wave angle", "beta"], all_param["beta"])
            if "theta" in all_param.keys():
                _check_angle(["flow deflection angle", "theta"], all_param["theta"])
            if "theta_c" in all_param.keys():
                _check_angle(["half-cone angle", "theta_c"], all_param["theta_c"])
            # all_param include all parameters, even if they were not specified.
            # therefore, I need to check if flag has been effectively given!
            if "flag" in all_param.keys() and len(args) == len(all_param.keys()):
                args[-1] = _check_flag_shockwave(all_param["flag"])

            res = original_function(*args, **kwargs)
            return ret_correct_vals(res)
        
        def no_check_function(*args, **kwargs):
            res = original_function(*args, **kwargs)
            return ret_correct_vals(res)
        wrapper_function.__no_check = no_check_function
        return wrapper_function

    if callable(var):
        return decorator(var)
    else:
        return decorator


# This decorator is used to convert and check the arguments of the
# function in the modules: isentropic, fanno, rayleigh, generic
def check(var=None):
    def decorator(original_function):
        indeces = [0]
        if not callable(var):
            indeces = var

        @wraps(original_function)
        @as_array(indeces)
        def wrapper_function(*args, **kwargs):
            args = list(args)
            all_param = get_parameters_dict(original_function, *args, **kwargs)

            if "M" in all_param.keys():
                _check_mach_number(all_param["M"], 0)
            # TODO: the following check is used in the shockwave.mach_downstream
            # function. Ideally, it should be inside check_shockwave decorator,
            # but that already does a check for M1. Alternatives:
            # 1. build a new ad-hoc decorator only for shockwave.mach_downstream
            # 2. see if it's possible to pass comparison arguments to the decorator,
            # for example to say check M1 > 0.
            if "M1" in all_param.keys():
                _check_mach_number(all_param["M1"], 0)
            if "gamma" in all_param.keys():
                _check_specific_heat_ratio(all_param["gamma"])
            
            # all_param include all parameters, even if they were not specified.
            # therefore, I need to check if flag has been effectively given!
            if "flag" in all_param.keys() and len(args) > 1:
                args[1] = _check_flag(all_param["flag"])

            res = original_function(*args, **kwargs)
            return ret_correct_vals(res)
        
        def no_check_function(*args, **kwargs):
            res = original_function(*args, **kwargs)
            return ret_correct_vals(res)
        wrapper_function.__no_check = no_check_function
        return wrapper_function

    if callable(var):
        return decorator(var)
    else:
        return decorator

# Convert the arguments specified in index_list to np.ndarray.
# By applying this conversion, the function will be able to handle
# as argument both a number, a list of numbers or a np.ndarray.
def as_array(index_list=[0]):
    """
    Convert the arguments specified in index_list to np.ndarray.
    With this we can pass a number, a list of numbers or a np.ndarray.

    Parameters
    ----------
        original_function : callable
        index_list : list
    """
    def decorator(original_function):
        @wraps(original_function)
        def wrapper_function(*args, **kwargs):
            # def func(x):
            #     """
            #     Check if it is of type np.ndarray.
            #     If not, convert to np.ndarray and make sure it is at least 
            #     1 dimensional.
            #     """
            #     if not isinstance(x, np.ndarray):
            #         return np.atleast_1d(np.array(x, copy=False, dtype=np.float64))
            #     if x.ndim == 0:
            #         return np.atleast_1d(np.array(x, copy=False, dtype=np.float64))
            #     return x
            
            args = list(args)
            for i in index_list:
                # args[i] = np.asarray(args[i], dtype=np.float64)
                args[i] = convert_to_ndarray(args[i])
            return original_function(*args, **kwargs)
        wrapper_function.__no_check = original_function
        return wrapper_function
    return decorator


def get_parameters_dict(original_function, *args, **kwargs):
    """
    Get a dictionary of parameters passed to the original_function.
    """
    param = inspect.signature(original_function).parameters
    all_param = {
        k: args[n] if n < len(args) else v.default
        for n, (k, v) in enumerate(param.items()) if k != 'kwargs'
    }
    return all_param

# This decorator is used to compute the average execution time of a function.
# By default, @Average_Execution_Time repeats the function 10 times.
# It is possible to override the number of repetition by providing an 
# input number: @Average_Execution_Time(20)
def average_execution_time(N=10):
    def decorator(original_function):
        n = 10
        if not callable(N):
            n = N
        @wraps(original_function)
        def wrapper_function(*args):
            t = 0
            for i in range(n):
                print(i)
                start = timer()
                result = original_function(*args)
                end = timer()
                t += (end - start)
            t /= n
            print("{} averaged execution time for {} repetitions: {} sec".format(original_function.__name__, n, t))
            return result
        return wrapper_function
    
    if callable(N):
        return decorator(N)
    else:
        return decorator
