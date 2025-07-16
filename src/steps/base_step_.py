from abc import ABC, abstractmethod

class BaseStep(ABC):
    """
    Clase base abstracta para cada paso del pipeline.
    Cada paso debe implementar el método run() que recibe y retorna un DataFrame.
    """

    def __init__(self, data=None, params=None, general_config=None):
        """
        Parámetros:
            data: pd.DataFrame - datos de entrada (metadatos + features)
            params: dict - parámetros específicos para este paso
            general_config: dict - configuración general (rutas, etc)
        """
        self.data = data
        self.params = params if params is not None else {}
        self.general_config = general_config if general_config is not None else {}

    @abstractmethod
    def run(self):
        """
        Método que debe implementar cada subclase para ejecutar su lógica.
        Debe retornar el DataFrame modificado o extendido.
        """
        pass
