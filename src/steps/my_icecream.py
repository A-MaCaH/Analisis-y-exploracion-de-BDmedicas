class Show:

    def __init__(self, default_message="...", condition=False):
        """
        Inicializa la clase Show
        
        Args:
            default_message (str): Mensaje por defecto a mostrar
        """
        self.default_message = default_message
        self.messages = {}
        self.condition = condition
    
#    def set_message(self, variable_name, message):
        """
        Establece un mensaje personalizado para una variable específica
        
        Args:
            variable_name (str): Nombre de la variable
            message (str): Mensaje a mostrar cuando la variable sea True
        """
 #       self.messages[variable_name] = message
    
#    def check_and_show(self, variable_name, value):
        """
        Verifica el valor de una variable y muestra un mensaje si es True
        
        Args:
            variable_name (str): Nombre de la variable
            value (bool): Valor de la variable
        """
#        if value is True:
#            message = self.messages.get(variable_name, self.default_message)
#            print(f"[{variable_name}] {message}")
#        else:
#            print(f"[{variable_name}] Condición no activada")
    
#    def show_if_true(self, **kwargs):
        """
        Verifica múltiples variables y muestra mensajes para las que sean True
        
        Args:
            **kwargs: Variables y sus valores (nombre=valor)
        """
#        for variable_name, value in kwargs.items():
#            if value is True:
#                message = self.messages.get(variable_name, self.default_message)
#                print(f"✓ [{variable_name}] {message}")
    
    def show(self, message=None):
        """
        Muestra un mensaje si la condición es True
        
        Args:
            condition (bool): Condición a evaluar
            message (str): Mensaje a mostrar (opcional)
        """
        if self.condition is True:
            msg = message if message else self.default_message
            print(msg)
    
    # def monitor_variables(self, variables_dict):
    #     """
    #     Monitorea un diccionario de variables y muestra mensajes para las que sean True
        
    #     Args:
    #         variables_dict (dict): Diccionario con variables y sus valores
    #     """
    #     print("=== Monitor de Variables ===")
    #     for var_name, value in variables_dict.items():
    #         if value is True:
    #             message = self.messages.get(var_name, self.default_message)
    #             print(f"🟢 {var_name}: {message}")
    #         else:
    #             print(f"🔴 {var_name}: Inactivo")
    #     print("=" * 30)


# Ejemplo de uso del programa
if __name__ == "__main__":
    # Crear una instancia de Show
    show = Show(condition=True)
    
    # Configurar mensajes personalizados
    show.show("sistema_activo")
