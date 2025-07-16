from src.utils import save_results
from importlib import import_module
from src.steps.my_icecream import Show
import logging
show = Show(condition=False)
class Pipeline:
    def __init__(self, config):
        self.config = config
        self.steps_config = config.get('pipeline', [])
        self.general_config = config.get('general', {})
        self.data = None  # puede ser un DataFrame o estructura compuesta

    def run(self):
        for step_conf in self.steps_config:
            show.show(f"tipo al inicio {type(self.data)}")
            name = step_conf['name']
            params = step_conf.get('params', {})
            print("....................")
            print(f"Running step: {name}")

            module = import_module(f"src.steps.{name}")
            step_class_name = ''.join(part.capitalize() for part in name.split('_'))
            StepClass = getattr(module, step_class_name)
            step_instance = StepClass(self.data, params, self.general_config)
            show.show(f"tipo al medio {type(self.data)}")
            self.data = step_instance.run()
            show.show(f"tipo al final {type(self.data)}")            

        # Guardar la salida final
        if self.data is not None:
            save_results(self.general_config['output_dir'], 'final_output.csv', self.data)
