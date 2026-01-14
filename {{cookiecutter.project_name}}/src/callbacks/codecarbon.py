from lightning.pytorch.callbacks import Callback
from codecarbon import EmissionsTracker

class CodeCarbonCallback(Callback):
    def __init__(self, project_name: str, experiment_name: str, save_to_file: bool = False, log_level: str = 'error'):
        self.project_name = project_name
        self.save_to_file = save_to_file
        self.log_level = log_level
        self.experiment_name = experiment_name
        
        self.tracker = EmissionsTracker(
            project_name=self.project_name,
            save_to_file=self.save_to_file,
            log_level=self.log_level,
            experiment_name=self.experiment_name
        )

    def _start_tracker(self, phase: str):
        self.tracker.start_task(phase)

    def _stop_task(self, trainer, phase: str):
        emissions = self.tracker.stop_task(phase)
        if trainer.logger and hasattr(trainer.logger, "experiment"):
            
            trainer.logger.experiment.log_param(
                run_id=trainer.logger.run_id, 
                key=f"{phase}_energy",  
                value=emissions.energy_consumed)
            
            trainer.logger.experiment.log_param(
                run_id=trainer.logger.run_id, 
                key=f"{phase}_emissions", 
                value=emissions.emissions)
            
        return emissions

    def on_fit_start(self, trainer, pl_module):
        self._start_tracker("training")

    def on_fit_end(self, trainer, pl_module):
        self._stop_task(trainer, "training")
        self.tracker.stop()

    def on_test_start(self, trainer, pl_module):
        self._start_tracker("testing")

    def on_test_end(self, trainer, pl_module):
        self._stop_task(trainer, "testing")
        self.tracker.stop()