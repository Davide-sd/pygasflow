import numpy as np
from timeit import default_timer as timer
from functools import wraps
import inspect

# TODO:
# 1. check_M_gamma. Either M > 0 or change the ratio functions, for instance:
#    T/T* -> T*/T, ... or maybe take into account M=0 into the ratio function.
# 2. Implement type checking?!?!?
# 3. Do I really need @do_nothing???

# It does nothig to the arguments.
# It only provides the __bypass_decorator functionality.
def do_nothing(original_function):
    @wraps(original_function)
    @convert_first_argument
    def wrapper_function(*args, **kwargs):
        return original_function(*args, **kwargs)
    wrapper_function.__bypass_decorator = original_function
    return wrapper_function


# After making sure the first argument passed to the functions is of type np.ndarray,
# check that Mach >= 0 and gamma > 1.
# TODO: is it possible to use a decorator with argument and merge this and the following
# decorator togheter?
def check_M_gamma_shockwave(original_function):
    @wraps(original_function)
    @convert_first_argument
    def wrapper_function(*args, **kwargs):
        # check Mach number
        assert np.all(args[0] >= 1), "The Mach number must be >= 1."
        if len(args) > 1:
            assert args[1] > 1, "The specific heats ratio must be > 1."
        return original_function(*args, **kwargs)
    wrapper_function.__bypass_decorator = original_function
    return wrapper_function

# After making sure the first argument passed to the functions is of type np.ndarray,
# check that Mach >= 0 and gamma > 1.
def check_M_gamma(original_function):
    @wraps(original_function)
    @convert_first_argument
    def wrapper_function(*args, **kwargs):
        # check Mach number
        assert np.all(args[0] > 0), "The Mach number must be > 0."
        if len(args) > 1:
            assert args[1] > 1, "The specific heats ratio must be > 1."
        return original_function(*args, **kwargs)
    wrapper_function.__bypass_decorator = original_function
    return wrapper_function

# After making sure the first argument passed to the functions is of type np.ndarray,
# check that gamma > 1.
def check_ratio_gamma(original_function):
    @wraps(original_function)
    @convert_first_argument
    def wrapper_function(*args, **kwargs):
        # get a dictionary of parameters passed to the original_function
        param = inspect.signature(original_function).parameters
        all_param = {
            k: args[n] if n < len(args) else v.default
            for n, (k, v) in enumerate(param.items()) if k != 'kwargs'
        }
        # all_param.update(kwargs)

        if "gamma" in all_param.keys():
            assert all_param["gamma"] > 1, "The specific heats ratio must be > 1."
        # if "flag" in all_param.keys():
        #     args = list(args)
        #     # flag should be the second argument
        #     args[1] = args[1].lower()
        #     assert flag in ["sub", "sup"], "flag can be either 'sub' or 'sup'."
        return original_function(*args, **kwargs)
    wrapper_function.__bypass_decorator = original_function
    return wrapper_function

# make sure the first argument passed to the functions is of type np.ndarray
# with this we can pass a number, a list of numbers or a np.ndarray
def convert_first_argument(original_function):
    @wraps(original_function)
    def wrapper_function(*args):
        args = list(args)
        args[0] = np.asarray(args[0], dtype=np.float64)
        return original_function(*args)
    wrapper_function.__bypass_decorator = original_function
    return wrapper_function

# Print the execution time for the decorated function
def execution_time(original_function):
    @wraps(original_function)
    def wrapper_function(*args):
        start = timer()
        result = original_function(*args)
        end = timer()
        print("{} executed in {} sec".format(original_function.__name__, end - start))
        return result

    return wrapper_function

# Print the average execution time for the decorated function. N is the number of repetition to perform.
def average_execution_time(N=10):
    def execution_time(original_function):
        @wraps(original_function)
        def wrapper_function(*args):
            t = 0
            for i in range(N):
                start = timer()
                result = original_function(*args)
                end = timer()
                t += (end - start)
            t /= N
            print("{} averaged execution time for {} repetitions: {} sec".format(original_function.__name__, N, t))
            return result
        return wrapper_function
    return execution_time
