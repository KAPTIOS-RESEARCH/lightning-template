import logging, warnings, torch, mlflow, os
from src.models import *
from src.datasets import *
warnings.filterwarnings("ignore")
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf
from datetime import datetime

OmegaConf.register_new_resolver(
    "timestamp",
    lambda fmt="%Y%m%d_%H%M%S": datetime.now().strftime(fmt),
)

class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)
    
    def before_fit(self):
        """Hook called before training starts"""
        config_file = getattr(self.config, "config", None)
        
        if config_file and os.path.exists(config_file):
            if isinstance(self.trainer.logger, MLFlowLogger):
                mlflow.log_artifact(config_file, artifact_path="config")
                logging.info(f"✓ Logged config file: {config_file}")

        if isinstance(self.trainer.logger, MLFlowLogger):
            mlflow.log_params({
                "environment": os.getenv("ENVIRONMENT", "development"),
                "git_commit": os.getenv("GIT_COMMIT", "unknown"),
            })

    
    def after_fit(self):
        """Hook called after training"""
        if isinstance(self.trainer.logger, MLFlowLogger):
            run_id = self.trainer.logger.run_id
            
            mlflow.pytorch.log_model(
                self.model,
                artifact_path="model",
                registered_model_name=self.model.__class__.__name__.lower()
            )
            
            val_loss = self.trainer.callback_metrics.get('val_loss', None)
            val_acc = self.trainer.callback_metrics.get('val_acc', None)
            
            logging.info(f"✓ Model logged to MLflow")
            logging.info(f"  Run ID: {run_id}")
            logging.info(f"  Val Loss: {val_loss}")
            logging.info(f"  Val Acc: {val_acc}")
            
def cli_main():
    
    project_name = "{{cookiecutter.project_name}}"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Model Training - %(levelname)s - %(message)s'
    )
    
    cli = CustomLightningCLI(
        run=True,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"}
    )


if __name__ == "__main__":
    cli_main()