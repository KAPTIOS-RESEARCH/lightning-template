from lightning.pytorch.callbacks import Callback
from codecarbon import EmissionsTracker

class CodeCarbonCallback(Callback):
    def __init__(self, project_name: str, save_to_file: bool = False, log_level: str = 'error'):
        self.project_name = project_name
        self.tracker = EmissionsTracker(
            project_name=f"{self.project_name}",
            save_to_file=save_to_file,
            log_level=log_level,
        )

    def on_fit_start(self, trainer, pl_module):
        self.tracker.start("training")

    def on_fit_end(self, trainer, pl_module):
        self.tracker.stop("training")

    def on_test_start(self, trainer, pl_module):
        self.tracker.start("testing")

    def on_test_end(self, trainer, pl_module):
        self.tracker.stop("testing")