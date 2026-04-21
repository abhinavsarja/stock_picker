#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from stock_picker_1.crew import StockPicker1

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the research crew.
    """
    inputs = {
        'sector': 'Technology',
        "current_date": str(datetime.now()),
        "output_dir": "/tmp/stock_picker_output",
    }

    # Create and run the crew
    result = StockPicker1().crew().kickoff(inputs=inputs)

    # Print the result
    print("\n\n=== FINAL DECISION ===\n\n")
    print(result.raw)


