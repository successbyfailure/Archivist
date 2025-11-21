# Archivist

Archivist es un sistema de indexado de contenidos en bibliotecas para poder usarla con LLMs, sistemas rag, bots y automatizaciones.
## Development

Install dependencies and run the API:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir src
```

The service exposes ingestion endpoints for files, URLs, Git repositories, APIs, webhook payloads, and VHS video transcripts. Query endpoints `/rag/query` and `/rag/chat` support streaming, structured responses, filters, retrieval-only mode, and pipeline selection. Admin endpoints expose stats and query replay details. A lightweight web UI is available at `/ui` for uploads and query testing.

## Docker Compose

Para ejecutar el servicio con Docker Compose:

```bash
docker compose build
docker compose up -d
```

Variables de entorno opcionales pueden declararse en un archivo `.env` en la raíz del proyecto. Los volúmenes `./data` y `./logs` se montan para persistir los índices y los registros. El servicio quedará accesible en `http://localhost:8000` con la documentación interactiva en `http://localhost:8000/docs` y el panel ligero en `http://localhost:8000/ui`.
