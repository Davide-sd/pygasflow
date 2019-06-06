import numpy as np
from timeit import default_timer as timer
from functools import wraps
import inspect

# NOTE: I have decided to not explicitely do a type chek on variables.
# This means you can pass any value on any variable: if it works, great,
# if it doesn't, that's your problem: read the docs and be smart. :)

def _Check_Specific_Heat_Ratio(gamma):
    assert gamma > 1, "The specific heats ratio must be > 1."

def _Check_Mach_Number(M, value):
    assert np.all(M >= value), "The Mach number must be >= {}.".format(value)

def _Check_Flag(flag):
    flag = flag.lower()
    assert flag in ["sub", "super"], "Flag can be either 'sub' or 'super'."
    return flag

def _Check_Flag_Shockwave(flag):
    flag = flag.lower()
    assert flag in ["weak", "strong"], "Flag can be either 'weak' or 'strong'."
    return flag

# TODO: this approach let the user insert an angle=None even into
# functions that need a float value!
def _Check_Angle(angle_name, angle):
    #  some functions can accept an angle with value None!
    if angle is not None:
        angle = np.asarray(angle)
        assert np.all(angle >= 0) and np.all(angle <= 90), "The {} must be 0 <= {} <= 90.".format(*angle_name)


# This decorator is used to convert and check the arguments of the
# functions in the shockwave module
def Check_Shockwave(var=None):
    def decorator(original_function):
        indeces = [0]
        if not callable(var):
            indeces = var

        @wraps(original_function)
        @As_Array(indeces)
        def wrapper_function(*args, **kwargs):
            args = list(args)
            all_param = Get_Parameters_Dict(original_function, *args, **kwargs)
            
            if "M" in all_param.keys():
                _Check_Mach_Number(all_param["M"], 0)
            if "M1" in all_param.keys():
                _Check_Mach_Number(all_param["M1"], 1)
            if "MN1" in all_param.keys():
                _Check_Mach_Number(all_param["MN1"], 1)
            if "gamma" in all_param.keys():
                _Check_Specific_Heat_Ratio(all_param["gamma"])
            if "beta" in all_param.keys():
                _Check_Angle(["shock wave angle", "beta"], all_param["beta"])
            if "theta" in all_param.keys():
                _Check_Angle(["flow deflection angle", "theta"], all_param["theta"])
            if "theta_c" in all_param.keys():
                _Check_Angle(["half-cone angle", "theta_c"], all_param["theta_c"])
            # all_param include all parameters, even if they were not specified.
            # therefore, I need to check if flag has been effectively given!
            if "flag" in all_param.keys() and len(args) == len(all_param.keys()):
                args[-1] = _Check_Flag_Shockwave(all_param["flag"])

            return original_function(*args, **kwargs)
        wrapper_function.__no_check = original_function
        return wrapper_function

    if callable(var):
        return decorator(var)
    else:
        return decorator


# This decorator is used to convert and check the arguments of the
# function in the modules: isentropic, fanno, rayleigh, generic
def Check(var=None):
    def decorator(original_function):
        indeces = [0]
        if not callable(var):
            indeces = var

        @wraps(original_function)
        @As_Array(indeces)
        def wrapper_function(*args, **kwargs):
            args = list(args)
            all_param = Get_Parameters_Dict(original_function, *args, **kwargs)

            if "M" in all_param.keys():
                _Check_Mach_Number(all_param["M"], 0)
            if "gamma" in all_param.keys():
                _Check_Specific_Heat_Ratio(all_param["gamma"])
            
            # all_param include all parameters, even if they were not specified.
            # therefore, I need to check if flag has been effectively given!
            if "flag" in all_param.keys() and len(args) > 1:
                args[1] = _Check_Flag(all_param["flag"])

            return original_function(*args, **kwargs)
        wrapper_function.__no_check = original_function
        return wrapper_function

    if callable(var):
        return decorator(var)
    else:
        return decorator


# Convert the arguments specified in index_list to np.ndarray.
# By applying this conversion, the function will be able to handle
# as argument both a number, a list of numbers or a np.ndarray.
def As_Array(index_list=[0]):
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
            # print("Converting to array", index_list)
            args = list(args)
            for i in index_list:
                args[i] = np.asarray(args[i], dtype=np.float64)
            return original_function(*args, **kwargs)
        wrapper_function.__no_check = original_function
        return wrapper_function
    return decorator


def Get_Parameters_Dict(original_function, *args, **kwargs):
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
def Average_Execution_Time(N=10):
    def decorator(original_function):
        n = 10
        if not callable(N):
            n = N
        @wraps(original_function)
        def wrapper_function(*args):
            print("asd", n)
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
