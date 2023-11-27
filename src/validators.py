"""
A Python module containing functions to validate different parts of the Risk
game.

Author: Kieran Ahn
Date: 11/23/2023
"""


def validate(toValidate, condition: bool, message: str, exceptionType: Exception = Exception):
    """
    Basic validator function, which evaluates a condition and raises an error
    if that condition is not met.

    :params:\n
    object          --  the object we are validating\n
    condition       --  the condition to evaluate, which evaluates to a boolean\n
    message         --  the error message to raise if the validator fails\n
    exceptionType   --  The type of exception to raise, if any are applicable\n

    :returns:\n
    toValidate      -- the validated object\n
    """

    if (not condition):
        raise exceptionType(message)

    return toValidate


def validate_is_type[T](object, desired_type: T):
    """
    Validates whether an object is a given type or not

    :params:\n
    object  --  an object that could be a certain type\n
    type    --  the type we want the object to be\n

    :returns:\n
    object  --  the validated object confirmed to be a certain type
    """
    object_type = type(object)

    return validate(object, object_type is desired_type or issubclass(object_type, desired_type), f'Expected {desired_type}, but got {object_type}', TypeError)
