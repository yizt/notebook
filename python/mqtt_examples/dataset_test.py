"""
@ Author: yizuotian
@ Date: 2025-03-19 17:30:10
@ LastEditors: yizuotian
@ LastEditTime: 2025-03-19 17:30:11
@ FilePath: /mqtt_examples/dataset_test.py
@ Description: Do edit!
"""
from datasets import Dataset
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from loguru import logger

def test():
    """
    @ description: 
    @ param {type} 
    @ return: 
    """
    data_dict = {
        "text": ["This is a sample text.", "Another sample text."],
        "label": [0, 1]
    }
    dataset = Dataset.from_dict(data_dict)
    logger.debug(dataset)

    data_dict = {
        "text": ["This is a sample text.", "Another sample text.", "Yet another text."],
        "label": [0, 1, 0],
        "user_info": [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30},
            {"name": "Charlie", "age": 35}
        ]
    }

    # 定义 features
    features = Features(
        {
            "text": Value("string"),  # 文本字段
            "label": ClassLabel(names=["class_0", "class_1"]),  # 分类标签字段
            "user_info": {  # 嵌套字段
                "name": Value("string"),
                "age": Value("int64")
            }
        }
    )

    # 使用 from_dict 方法创建 Dataset 对象，并指定 features
    dataset = Dataset.from_dict(data_dict, features=features)

    logger.debug(dataset)

    ft_dict = {col: [] for col in features}

    logger.debug(ft_dict)

    dataset = Dataset.from_dict(ft_dict, features=features)
    logger.debug(dataset)




if __name__ == "__main__":
	test()
