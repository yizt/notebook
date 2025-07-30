#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@ Author: yizuotian
@ Date: 2025-03-20 10:16:34
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-20 10:16:35
@ FilePath: /mqtt_examples/arrow_test.py
@ Description: Do edit!
"""
from loguru import logger
def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    import pyarrow as pa
    import pyarrow.feather as feather

    # Create a table from a list of dictionaries
    data = [
        {"name": "Alice", "age": 25, "gender": "F"},
        {"name": "Bob", "age": 30, "gender": "M"},
        {"name": "Charlie", "age": 35, "gender": "M"}
    ]
    data = {'n_legs': [2, 4, 5, 100], 'animals': ["Flamingo", "Horse", "Brittle stars", "Centipede"]}
    table = pa.Table.from_pydict(data)

    # Write the table to an Arrow file
    feather.write_feather(table, "data.arrow")

    # Read the table from an Arrow file
    table = feather.read_table("data.arrow")
    logger.debug(table)

if __name__ == "__main__":
	test()
