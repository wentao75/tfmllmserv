import uvicorn
import typer

app = typer.Typer()

@app.command()
def main(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """启动API服务"""
    uvicorn.run(
        "tfmllmserv.api:app",
        host=host,
        port=port,
        reload=reload
    )

if __name__ == "__main__":
    app() 