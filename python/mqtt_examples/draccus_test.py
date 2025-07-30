from dataclasses import dataclass
import draccus
from loguru import logger
from draccus import field
from typing import List,Dict

from draccus import parse_from_command_line

# Choice Registry lets you define a choice of implementations that can be selected at runtime
@dataclass
class ModelConfig(draccus.ChoiceRegistry):
    pass


@ModelConfig.register_subclass('gpt')
@dataclass
class GPTConfig(ModelConfig):
    """GPT Model Config"""
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768


@ModelConfig.register_subclass('bert')
@dataclass
class BERTConfig(ModelConfig):
    """BERT Model Config"""
    num_layers: int = 12
    num_heads: int = 12
    hidden_size: int = 768
    dropout: float = 0.1

@dataclass
class CCConfig:
    c:str = "ccc"

@dataclass
class BBConfig(BERTConfig,CCConfig):
    a:int=6


@dataclass
class Robot:
    robot_name:str=""
    ip : str="localhost"
    port : int =8000

@dataclass
class Robots:
    a:Robot=Robot()
    b:Robot=Robot()


@dataclass
class TrainConfig:
    """Training Config for Machine Learning"""
    workers: int = 8                  # The number of workers for training
    exp_name: str = 'default_exp'     # The experiment name

    model: ModelConfig = GPTConfig()  # The model configuration
    # robots: Dict[str,Robot] = field(default_factory=lambda:{"a":Robot()})
    # robots:  Dict[str,Robot] = {"a":Robot()}
    robots:Robots = Robots()
    

    def get_robot(self,name):
        return getattr(self.robots,name)
        return self.robots.__dict__[name]

def create_config(**overrides):
    return BERTConfig(**overrides)





@draccus.wrap()
def main(cfg: TrainConfig):
    logger.debug(cfg)
    logger.debug(cfg.get_robot('a'))
    # print(f"Training {cfg.exp_name} with {cfg.workers} workers...")

    # # 使用示例
    # bcfg = create_config(num_layers=30)
    # logger.debug(bcfg)

    # logger.debug(BBConfig())



if __name__=='__main__':
    main()