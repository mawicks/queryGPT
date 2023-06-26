import os

IRS990_SCHEMA = "irs990"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
# The token counts returned by tiktoken don't exctly match the actual token
# counts in the API Allow a little margin of error to stay below the 16k limit.
# (Tiktoken typically underestimates the token count by eight.)
INPUT_TOKEN_GOAL = 14_900
MODEL = "gpt-3.5-turbo-16k"  # Or use "text-davinci-003" for GPT-3
