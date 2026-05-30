from __future__ import annotations
import json
import sys

try:
    import click
    _CLICK_AVAILABLE = True
except ImportError:
    _CLICK_AVAILABLE = False


if _CLICK_AVAILABLE:
    @click.group()
    def cli() -> None:
        """predikit — ML model utilities for LLM agents."""

    @cli.command()
    @click.argument("model_path", type=click.Path(exists=True))
    @click.option("--name", default="model", show_default=True, help="Tool name used in schema generation.")
    @click.option("--description", default="ML model prediction", show_default=True, help="Tool description.")
    def inspect(model_path: str, name: str, description: str) -> None:
        """Inspect a saved model file and print its metadata and OpenAI schema."""
        try:
            import joblib
        except ImportError:
            raise click.ClickException("joblib is required. Install it with: pip install predikit[cli]")

        from pydantic import create_model
        from .introspect import introspect
        from .tool import ModelTool

        model = joblib.load(model_path)
        meta = introspect(model)

        click.echo(f"Model:    {type(model).__name__}")
        click.echo(f"Task:     {meta['task']}")
        if meta["n_features"] is not None:
            click.echo(f"Features: {meta['n_features']}")

        if meta["feature_names"]:
            click.echo("Feature names:")
            for fname in meta["feature_names"]:
                click.echo(f"  {fname}")
        else:
            click.echo("Feature names: (none — fit the model with a named DataFrame to enable)")

        if meta["classes"] is not None:
            click.echo(f"Classes:  {meta['classes']}")

        if meta["feature_names"]:
            fields = {f: (float, ...) for f in meta["feature_names"]}
            input_schema = create_model("Input", **fields)
            tool = ModelTool(
                model=model,
                name=name,
                description=description,
                input_schema=input_schema,
                output_name="prediction",
                output_description="model output",
            )
            click.echo("\nOpenAI schema:")
            click.echo(json.dumps(tool.to_openai(), indent=2))
        else:
            click.echo("\nOpenAI schema: unavailable (fit model with a named DataFrame to enable)")

else:
    def cli() -> None:  # type: ignore[misc]
        """Fallback when click is not installed."""
        print("Error: 'click' is required. Install it with: pip install predikit[cli]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli()
