import logging
from function_schema import get_function_schema

logger = logging.getLogger(__name__)


def calculate_fixed_monthly_payment(principal, annual_interest_rate, months):
    """Calculate the fixed monthly payment for a loan.

    Parameters:
        principal (float): The initial amount of the loan.
        annual_interest_rate (float): The annual interest rate as a percentage (e.g., 5 for 5%).
        months (int): The number of months to repay the loan.

    Returns:
        float: The fixed monthly payment amount.
    """
    monthly_interest_rate = annual_interest_rate / (12 * 100)
    if monthly_interest_rate == 0:
        return principal / months
    return principal * (monthly_interest_rate * (1 + monthly_interest_rate) ** months) / ((1 + monthly_interest_rate) ** months - 1)


def calculate_future_value(principal, annual_interest_rate, years):
    """
    Calculate the total amount and interest earned after a given number of years.

    Parameters:
        principal (float): The initial amount of money.
        annual_interest_rate (float): The annual interest rate as a percentage (e.g., 5 for 5%).
        years (int): The number of years the money is invested for.

    Returns:
        float: the total amount after the given years
    """
    rate_decimal = annual_interest_rate / 100
    total_amount = principal * (1 + rate_decimal) ** years
    return f"số tiền bạn nhận được sau {years} là {total_amount} VNĐ"



def get_tool_schema(function):
    return {
        "type": "function",
        "function": get_function_schema(function)
    }
