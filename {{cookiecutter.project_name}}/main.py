import logging, warnings, torch, os
from src.models import *
from src.datasets import *
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI
from datetime import datetime
from comet_ml import CometExperiment 

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)
        
def cli_main():
    
    project_name = "{{cookiecutter.project_name}}"
    if os.path.exists(".env"):
        load_dotenv(".env")
        logging.info("Loaded .env")
    elif os.path.exists(".env.example"):
        load_dotenv(".env.example")
        logging.info("Loaded .env.example")
    else:
        logging.error("No .env or .env.example file found")
    logging.getLogger("comet-ml").setLevel(logging.ERROR)

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Model Training - %(levelname)s - %(message)s'
    )
    
    cli = CustomLightningCLI(
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"}
    )
    
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logger = cli.trainer.logger
    # experiment_name = f"{cli.config.data.class_path}_{cli.config.model.class_path}_{timestamp}"
    # logger.experiment.set_name(experiment_name)
    # logger.experiment.log_env_details = False

if __name__ == "__main__":
    cli_main()