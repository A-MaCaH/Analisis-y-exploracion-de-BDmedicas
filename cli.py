import typer
from src.pipeline import Pipeline
import yaml
from pathlib import Path

app = typer.Typer()

@app.command()
def run(
    config_file: str = typer.Option(..., help="Ruta al archivo de configuración YAML"),
    output_dir: str = typer.Option(None, help="Sobrescribe el directorio de salida")
):
    """
    Ejecuta el pipeline de análisis de imágenes médicas usando un archivo de configuración.
    """
    config_path = Path(config_file)

    if not config_path.exists():
        typer.echo(f"❌ Config file {config_file} not found.")
        raise typer.Exit(code=1)

    with open(config_path, 'r') as f:
        typer.echo(f"Se abrio el archivo!!")
        config = yaml.safe_load(f)

    if output_dir:
        config['general']['output_dir'] = output_dir

    pipeline = Pipeline(config)
    pipeline.run()

if __name__ == "__main__":
    app()
