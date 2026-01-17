import logging, warnings, torch, mlflow, os
from src.models import *
from src.datasets import *
warnings.filterwarnings("ignore")
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf
from datetime import datetime
from dotenv import load_dotenv

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
            
                # logging.info("✓ Exporting ONNX model")

                # # Example input (REQUIRED)
                # # Option 1: hardcoded
                # example_input = torch.randn(1, *self.model.example_input_shape)

                # # Option 2: pull from datamodule (preferred)
                # # example_input = next(iter(self.trainer.datamodule.train_dataloader()))[0][:1]

                # onnx_path = "artifacts/model.onnx"

                # export_onnx(
                #     model=self.model,
                #     example_input=example_input,
                #     output_path=onnx_path,
                #     opset=export_cfg.get("opset", 17),
                # )

                # mlflow.log_artifact(onnx_path, artifact_path="onnx")

                # # ---------- QUANTIZED ONNX ----------
                # if export_cfg.get("quantized", False):
                #     quant_path = "artifacts/model_quant.onnx"

                #     export_quantized_onnx(
                #         fp32_onnx_path=onnx_path,
                #         output_path=quant_path,
                #     )

                #     mlflow.log_artifact(quant_path, artifact_path="onnx")

                #     logging.info("✓ Quantized ONNX exported")
            
            logging.info(f"✓ Model logged to MLflow")
            logging.info(f"  Run ID: {run_id}")
            logging.info(f"  Val Loss: {val_loss}")
            logging.info(f"  Val Acc: {val_acc}")
            
def cli_main():
    
    if os.path.exists(".env"):
        load_dotenv(".env")
        logging.info("Loaded .env")
    elif os.path.exists(".env.example"):
        load_dotenv(".env.example")
        logging.info("Loaded .env.example")
    else:
        logging.error("No .env or .env.example file found")
        
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